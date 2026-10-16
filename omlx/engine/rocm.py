# SPDX-License-Identifier: Apache-2.0
"""External engine for ROCm (AMD) inference nodes.

oMLX runs its own models on MLX/Metal. MLX has no ROCm backend, so an AMD GPU
(Strix Halo / RDNA, gfx1151 and friends) can never be an MLX rank. What an AMD
box *can* do is serve an OpenAI-compatible API from a ROCm-native engine —
``gufo``/``llama-server`` are the known-good ones — and oMLX can then drive it
as a first-class backend: the same product surface (``/v1/chat/completions``,
``/v1/completions``, tools, streaming, admin, benchmarks) with the compute
living on the AMD node.

This module is the adapter. It mirrors the shape of the Splash external engine
(``omlx/engine/splash.py``) but speaks plain HTTP to an endpoint rather than
spawning a local binary, so the ROCm node may be another machine — reachable
over the LAN or, on this project, a Thunderbolt 4 link.

Selection is per model, through the existing ``external_engine`` model setting
(``"rocm"``). The endpoint that backs the model comes from
``model_settings.external_engine_config`` — a small, engine-specific JSON blob:

.. code-block:: json

    {
      "base_url": "http://192.168.0.142:8731",
      "model": "gufo-flash-next",
      "api_key": "",
      "tokenizer": "/path/to/tokenizer",
      "label": "valhalla (ROCm)"
    }

``base_url`` and ``model`` are the only required keys; ``api_key`` is sent as a
Bearer token when present. ``tokenizer`` is optional and, when given, is loaded
with ``transformers`` for exact prompt token counts (an estimate is used
otherwise).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from omlx.engine.base import BaseEngine, GenerationOutput
from omlx.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

# A long prefill on a big model can stall the first byte for a while; the
# read timeout must outlast it. connect stays short so a dead node fails fast.
DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_READ_TIMEOUT_S = 3600.0
DEFAULT_START_TIMEOUT_S = 60.0
DEFAULT_START_POLL_S = 0.5

# Reasoning controls an OpenAI-compatible ROCm server may understand. Only
# keys the caller actually set are forwarded, so a server that rejects an
# unknown field never sees one. oMLX spells the extremes "xhigh"/"max"; the
# OpenAI spelling is "high".
_EFFORT_ALIASES = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class RocmEngineError(RuntimeError):
    """A ROCm endpoint error, distinct from an oMLX-internal failure."""


@dataclass(frozen=True)
class RocmEngineConfig:
    """Where a ROCm model lives and how to talk to it."""

    base_url: str
    upstream_model: str
    # oMLX-side identity for the model. Defaults to the upstream model id.
    model_name: str = ""
    api_key: str = ""
    label: str = "ROCm"
    # Optional local tokenizer (HF repo id or path) for exact token counts.
    tokenizer: str | None = None
    model_type: str | None = None
    request_timeout_s: float = DEFAULT_READ_TIMEOUT_S
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S
    extra_headers: tuple[tuple[str, str], ...] = ()
    # TLS verification of the endpoint. False is useful for a self-signed
    # certificate on a private link (e.g. a Thunderbolt-attached node).
    verify: bool = True
    # Whether the endpoint accepts a `tools` array (llama.cpp/gufo do).
    supports_tools: bool = True
    # Whether the endpoint accepts image content parts.
    accepts_image_input: bool = False

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("ROCm engine base_url is required")
        if not self.upstream_model:
            raise ValueError("ROCm engine upstream_model is required")

    @property
    def display_name(self) -> str:
        return self.model_name or self.upstream_model

    def http_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout_s,
            read=self.request_timeout_s,
            write=self.request_timeout_s,
            pool=self.connect_timeout_s,
        )

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        default_model: str = "",
        default_base_url: str = "",
    ) -> RocmEngineConfig:
        """Build a config from the ``external_engine_config`` JSON blob.

        Accepts a few alias spellings so hand-written settings are forgiving:
        ``base_url``/``url``/``endpoint`` and ``model``/``model_name``.
        """
        data = dict(data or {})
        base_url = str(
            data.get("base_url") or data.get("url") or data.get("endpoint")
            or default_base_url
        ).rstrip("/")
        upstream = str(
            data.get("model") or data.get("model_name") or data.get("upstream_model")
            or default_model
        )
        extra = data.get("extra_headers") or {}
        extra_headers = tuple(
            (str(k), str(v)) for k, v in dict(extra).items()
        ) if isinstance(extra, dict) else ()
        return cls(
            base_url=base_url,
            upstream_model=upstream,
            model_name=str(data.get("served_model_name") or data.get("display_name") or ""),
            api_key=str(data.get("api_key") or ""),
            label=str(data.get("label") or "ROCm"),
            tokenizer=(str(data["tokenizer"]) if data.get("tokenizer") else None),
            model_type=(str(data["model_type"]) if data.get("model_type") else None),
            request_timeout_s=float(
                data.get("request_timeout_s", DEFAULT_READ_TIMEOUT_S)
            ),
            connect_timeout_s=float(
                data.get("connect_timeout_s", DEFAULT_CONNECT_TIMEOUT_S)
            ),
            start_timeout_s=float(data.get("start_timeout_s", DEFAULT_START_TIMEOUT_S)),
            extra_headers=extra_headers,
            verify=bool(data.get("verify", True)),
            supports_tools=bool(data.get("supports_tools", True)),
            accepts_image_input=bool(data.get("accepts_image_input", False)),
        )


@dataclass
class _Accumulator:
    """Accumulates a streamed OpenAI response into one GenerationOutput.

    Mirrors ``omlx.engine.splash._StreamState``: reasoning deltas are wrapped
    in ``<think>``/``</think>`` so oMLX's shared reasoning parser can split
    them into ``reasoning_content``, and tool calls are emitted in the flat
    ``{"id", "name", "arguments"}`` shape that
    :func:`omlx.api.parser_tool_calls.convert_parser_tool_calls` expects.
    """

    text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    first_token_at: float | None = None
    in_think: bool = False

    def add_delta(self, choice: dict[str, Any]) -> str:
        """Apply one streamed choice; returns the text it produced (may be "")."""
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        piece = ""
        # Servers that stream chain-of-thought (vLLM, llama.cpp reasoning,
        # gufo) send it as `reasoning_content`, not `content`. Wrap it in the
        # <think> tags oMLX's parser understands, exactly as Splash does.
        if delta.get("reasoning_content"):
            piece += ("" if self.in_think else "<think>") + delta["reasoning_content"]
            self.in_think = True
        if delta.get("content"):
            piece += ("</think>" if self.in_think else "") + delta["content"]
            self.in_think = False
        for call in delta.get("tool_calls") or []:
            self._merge_tool_call(call)
        if piece:
            self.text += piece
        return piece

    def _merge_tool_call(self, call: dict[str, Any]) -> None:
        index = int(call.get("index", 0))
        slot = self.tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if call.get("id"):
            slot["id"] = call["id"]
        function = call.get("function") or {}
        # Name and arguments may both be split across chunks; append.
        slot["name"] += function.get("name") or ""
        slot["arguments"] += function.get("arguments") or ""

    def close_think(self) -> str:
        closing = "</think>" if self.in_think else ""
        self.in_think = False
        self.text += closing
        return closing

    def final(self) -> GenerationOutput:
        # If the budget ran out mid-reasoning, close the block so the parser
        # does not treat the tail as unclosed thinking.
        closing = self.close_think()
        details = (self.usage or {}).get("prompt_tokens_details") or {}
        tool_calls = None
        if self.tool_calls:
            tool_calls = [
                {
                    "id": slot["id"],
                    "name": slot["name"],
                    "arguments": slot["arguments"] or "{}",
                }
                for _, slot in sorted(self.tool_calls.items())
            ]
        return GenerationOutput(
            text=self.text,
            new_text=closing,
            finished=True,
            finish_reason=self.finish_reason or "stop",
            prompt_tokens=int((self.usage or {}).get("prompt_tokens", 0) or 0),
            completion_tokens=int((self.usage or {}).get("completion_tokens", 0) or 0),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            tool_calls=tool_calls,
            prompt_tps=float((self.usage or {}).get("prompt_tokens_per_second", 0.0) or 0.0),
            generation_tps=float(
                (self.usage or {}).get("completion_tokens_per_second", 0.0) or 0.0
            ),
            first_token_at=self.first_token_at,
            generated_until=time.perf_counter(),
        )


class RocmEngine(BaseEngine):
    """Serves one model from a remote (or local) ROCm OpenAI-compatible node.

    Unlike Splash, this engine never spawns a process: the ROCm node manages
    its own engine lifecycle (llama-swap/gufo on the box). ``start()`` therefore
    only verifies reachability and that the endpoint advertises the model, so a
    misconfigured endpoint fails the load with a clear reason instead of a
    silent fallback to MLX.
    """

    # The ROCm node budgets its own GPU memory; there is no oMLX scheduler here.
    _prefill_memory_guard_managed_externally = True
    supports_native_response_format = True
    # The weights (and therefore the tokenizer) live on the remote node. A
    # local tokenizer is optional, so the pool must not reject this engine for
    # lacking one -- token counts fall back to an estimate.
    local_tokenizer_optional = True

    def __init__(self, config: RocmEngineConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._tokenizer: Any = None
        self._tokenizer_loaded = False
        self._active = 0
        self._requests = 0
        self._last_models: dict[str, Any] = {}

    # -- identity ------------------------------------------------------------------

    @property
    def config(self) -> RocmEngineConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config.display_name

    @property
    def model_type(self) -> str | None:
        return self._config.model_type

    @property
    def accepts_image_input(self) -> bool:
        return self._config.accepts_image_input

    @property
    def tokenizer(self) -> Any:
        """A local tokenizer for exact counts, when one was configured.

        Returns ``None`` when unavailable; callers must tolerate that, which is
        why :meth:`count_chat_tokens` falls back to an estimate.
        """
        if not self._tokenizer_loaded:
            self._tokenizer_loaded = True
            path = self._config.tokenizer
            if path:
                try:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(path)
                except Exception:  # noqa: BLE001 - tokenizer is best-effort
                    logger.warning(
                        "ROCm engine: could not load tokenizer %r; falling back "
                        "to estimates",
                        path,
                        exc_info=True,
                    )
                    self._tokenizer = None
        return self._tokenizer

    def has_active_requests(self) -> bool:
        return self._active > 0

    # -- lifecycle -----------------------------------------------------------------

    async def start(self) -> None:
        if self._client is not None:
            return
        config = self._config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=config.headers(),
            timeout=config.http_timeout(),
            verify=config.verify,
        )
        try:
            await self._wait_until_ready()
        except BaseException:
            await self.stop()
            raise
        logger.info(
            "ROCm engine: serving %s from %s (%s)",
            config.display_name,
            config.base_url,
            config.label,
        )

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._config.start_timeout_s
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                models = await self.list_models()
                self._last_models = models
                if self._advertises_model(models):
                    return
                last_error = (
                    f"endpoint at {self._config.base_url} does not advertise "
                    f"model '{self._config.upstream_model}'"
                )
                if models:
                    # The endpoint is up but the model is absent: no point
                    # waiting, it is a configuration error.
                    raise RocmEngineError(
                        f"{last_error} (available: {self._model_ids(models)})"
                    )
            except RocmEngineError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(DEFAULT_START_POLL_S)
        raise RocmEngineError(
            f"ROCm endpoint {self._config.base_url} not ready within "
            f"{self._config.start_timeout_s}s ({last_error})"
        )

    @staticmethod
    def _model_ids(models: dict[str, Any]) -> str:
        return ", ".join(
            str(m.get("id")) for m in (models.get("data") or []) if m.get("id")
        )

    def _advertises_model(self, models: dict[str, Any]) -> bool:
        ids = {str(m.get("id")) for m in (models.get("data") or [])}
        return self._config.upstream_model in ids

    # -- endpoint helpers ----------------------------------------------------------

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RocmEngineError(
                f"ROCm engine for {self._config.display_name} is not started"
            )
        return self._client

    async def list_models(self) -> dict[str, Any]:
        """GET /v1/models from the endpoint (used by health and stats)."""
        client = self._require_client()
        response = await client.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def health(self) -> dict[str, Any]:
        """Probe the endpoint's model list plus a couple of common health paths."""
        client = self._require_client()
        result: dict[str, Any] = {"base_url": self._config.base_url, "ok": False}
        for path in ("/health", "/v1/models"):
            try:
                response = await client.get(path, timeout=5.0)
                result[path] = response.status_code
                if response.status_code == 200 and path == "/v1/models":
                    result["ok"] = True
            except httpx.HTTPError as exc:
                result[path] = f"{type(exc).__name__}: {exc}"
        return result

    # -- requests ------------------------------------------------------------------

    def _base_body(self, max_tokens: int, temperature: float, top_p: float,
                   top_k: int, min_p: float, repetition_penalty: float,
                   presence_penalty: float, kwargs: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._config.upstream_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # llama.cpp/gufo-style extensions; harmless to omit when zero.
        if top_k:
            body["top_k"] = top_k
        if min_p:
            body["min_p"] = min_p
        if repetition_penalty and repetition_penalty != 1.0:
            body["repetition_penalty"] = repetition_penalty
        if presence_penalty:
            body["presence_penalty"] = presence_penalty
        for key in ("frequency_penalty", "stop", "seed", "response_format",
                    "logprobs", "top_logprobs"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]
        effort = kwargs.get("reasoning_effort")
        if not effort:
            effort = _EFFORT_ALIASES.get(
                str((kwargs.get("chat_template_kwargs") or {}).get("reasoning_effort"))
            )
        if effort:
            body["reasoning_effort"] = effort
        # llama.cpp/gufo pass `chat_template_kwargs` straight to the Jinja chat
        # template, which is how `enable_thinking` is toggled on Qwen. Forward
        # it (minus reasoning_effort, which is not a template variable) so the
        # caller can turn thinking on or off.
        template = kwargs.get("chat_template_kwargs")
        if isinstance(template, dict):
            forwarded = {
                k: v for k, v in template.items() if k != "reasoning_effort"
            }
            if forwarded:
                body["chat_template_kwargs"] = forwarded
        return body

    async def _stream(
        self,
        path: str,
        body: dict[str, Any],
    ) -> AsyncIterator[GenerationOutput]:
        client = self._require_client()
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        accumulator = _Accumulator()
        self._active += 1
        self._requests += 1
        try:
            async with client.stream("POST", path, json=body) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace")
                    if 400 <= response.status_code < 500:
                        raise InvalidRequestError(f"ROCm: {detail[:500]}")
                    raise RocmEngineError(
                        f"ROCm endpoint returned HTTP {response.status_code}: "
                        f"{detail[:500]}"
                    )
                async for chunk in self._chunks(response):
                    if isinstance(chunk.get("usage"), dict):
                        accumulator.usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        piece = accumulator.add_delta(choice)
                        if not piece:
                            continue
                        now = time.perf_counter()
                        accumulator.first_token_at = accumulator.first_token_at or now
                        yield GenerationOutput(
                            text=accumulator.text,
                            new_text=piece,
                            finished=False,
                            first_token_at=accumulator.first_token_at,
                            generated_at=now,
                        )
            yield accumulator.final()
        finally:
            self._active -= 1

    @staticmethod
    async def _chunks(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    return
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                logger.debug("ROCm engine: skipping malformed SSE line: %r", line[:120])
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                raise RocmEngineError(f"ROCm endpoint error: {chunk['error']}")
            yield chunk

    @staticmethod
    async def _collect(outputs: AsyncIterator[GenerationOutput]) -> GenerationOutput:
        final: GenerationOutput | None = None
        async for output in outputs:
            final = output
        if final is None:
            raise RocmEngineError("ROCm endpoint produced no output")
        final.new_text = final.text
        return final

    # -- chat ----------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if tools and not self._config.supports_tools:
            raise InvalidRequestError(
                f"ROCm model '{self._config.upstream_model}' does not accept tools"
            )
        body = self._base_body(
            max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            presence_penalty, kwargs,
        )
        body["messages"] = messages
        if tools:
            body["tools"] = tools
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        async for output in self._stream("/v1/chat/completions", body):
            yield output

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> GenerationOutput:
        return await self._collect(self.stream_chat(messages, **kwargs))

    # -- plain completions ---------------------------------------------------------

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if isinstance(prompt, list):
            raise InvalidRequestError(
                "ROCm engine expects a text prompt, not token ids"
            )
        body = self._base_body(
            max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            presence_penalty, {"stop": stop, **kwargs} if stop else kwargs,
        )
        body["prompt"] = prompt
        async for output in self._stream("/v1/completions", body):
            yield output

    async def generate(self, prompt: str | list[int], **kwargs) -> GenerationOutput:
        return await self._collect(self.stream_generate(prompt, **kwargs))

    # -- token counts --------------------------------------------------------------

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        tokenizer = self.tokenizer
        if tokenizer is not None:
            with contextlib.suppress(Exception):
                ids = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True, tokenize=True,
                    **(chat_template_kwargs or {}),
                )
                return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        # Estimate: ~4 characters per token is the usual English average. The
        # server treats this as an admission hint, never as ground truth.
        chars = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chars += len(part["text"])
        return max(1, chars // 4)

    # -- stats ---------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "rocm",
            "label": self._config.label,
            "model_name": self.model_name,
            "upstream_model": self._config.upstream_model,
            "base_url": self._config.base_url,
            "running": self._client is not None,
            "requests": self._requests,
            "active_requests": self._active,
            "supports_tools": self._config.supports_tools,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        # The ROCm node owns its own prefix/snapshot cache; oMLX's hot-cache
        # budget does not apply.
        return None

    async def runtime_stats(self) -> dict[str, Any]:
        """Live endpoint stats for admin display (models + health)."""
        if self._client is None:
            return {"running": False}
        stats = self.get_stats()
        with contextlib.suppress(Exception):
            stats["models"] = self._model_ids(await self.list_models())
        with contextlib.suppress(Exception):
            stats["health"] = await self.health()
        return stats


async def probe_rocm_endpoint(
    base_url: str,
    *,
    api_key: str = "",
    timeout_s: float = 10.0,
    verify: bool = True,
) -> dict[str, Any]:
    """Reachability check for a ROCm endpoint.

    Returns ``{"ok": bool, "models": [...], "detail": str}``. Never raises: a
    probe is used for discovery, where an unreachable node is data, not an
    error.
    """

    result: dict[str, Any] = {"ok": False, "models": [], "detail": ""}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            verify=verify,
            timeout=httpx.Timeout(connect=min(timeout_s, 5.0), read=timeout_s,
                                  write=timeout_s, pool=timeout_s),
        ) as client:
            response = await client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [
                {"id": m.get("id"), "owned_by": m.get("owned_by")}
                for m in (data.get("data") or [])
            ]
            result["ok"] = True
            result["models"] = models
            result["detail"] = f"{len(models)} model(s)"
    except httpx.HTTPStatusError as exc:
        result["detail"] = f"HTTP {exc.response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result
