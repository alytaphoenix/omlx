# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm external engine (``omlx/engine/rocm.py``).

A ROCm node (AMD/Strix Halo) publishes an OpenAI-compatible API; oMLX proxies
it as a first-class backend. These tests drive :class:`RocmEngine` against a
small stdlib HTTP server that speaks that protocol, so no AMD hardware and no
``fastapi``/``pytest`` runtime is required to exercise the adapter.

The module is deliberately dependency-light: it imports only ``httpx`` and the
engine itself. Every test is a plain sync function that wraps an ``asyncio``
coroutine, so the file also runs directly::

    python tests/test_engine_rocm.py

(the harness at the bottom executes every ``test_*`` function).
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from typing import Any

from omlx.engine.rocm import (
    RocmEngine,
    RocmEngineConfig,
    RocmEngineError,
    probe_rocm_endpoint,
)
from omlx.exceptions import InvalidRequestError

UPSTREAM_MODEL = "gufo-flash-next"


# --------------------------------------------------------------------------- #
# A scriptable fake of an OpenAI-compatible ROCm endpoint.
# --------------------------------------------------------------------------- #


class _FakeUpstream:
    """Behaviour of the fake server; tests mutate these between requests."""

    def __init__(self, models: tuple[str, ...] = (UPSTREAM_MODEL,)) -> None:
        self.models = list(models)
        self.models_status = 200
        self.chat_error: tuple[int, str] | None = None
        self.completion_error: tuple[int, str] | None = None
        self.pieces = ["Hello", ", ", "world"]
        self.reasoning_pieces: list[str] = []
        self.tool_call_fragments: list[list[dict]] = []
        self.finish_reason = "stop"
        self.usage: dict[str, Any] | None = {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "prompt_tokens_per_second": 120.5,
            "completion_tokens_per_second": 39.7,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        # Recorded request bodies, in order.
        self.requests: list[tuple[str, dict]] = []

    @property
    def last_body(self) -> dict:
        return self.requests[-1][1] if self.requests else {}

    def sse(self) -> bytes:
        lines: list[str] = []

        def emit(obj: dict) -> None:
            lines.append("data: " + json.dumps(obj) + "\n\n")

        for piece in self.reasoning_pieces:
            emit(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        for piece in self.pieces:
            emit(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
            )
        for group in self.tool_call_fragments:
            for fragment in group:
                emit(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [fragment]},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
        emit(
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": self.finish_reason}
                ]
            }
        )
        if self.usage is not None:
            emit({"choices": [], "usage": self.usage})
        lines.append("data: [DONE]\n\n")
        return "".join(lines).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "fake-rocm/1.0"

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        return

    # -- helpers -------------------------------------------------------------
    def _fake(self) -> _FakeUpstream:
        return self.server.fake  # type: ignore[attr-defined]

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw)

    def _send(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # -- routes --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        fake = self._fake()
        if self.path == "/v1/models":
            if fake.models_status != 200:
                self._send(fake.models_status, {"error": "unavailable"})
                return
            self._send(200, {"object": "list", "data": [{"id": m} for m in fake.models]})
            return
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        fake = self._fake()
        body = self._read_body()
        fake.requests.append((self.path, body))
        if self.path == "/v1/chat/completions":
            if fake.chat_error is not None:
                self._send(fake.chat_error[0], {"error": fake.chat_error[1]})
                return
            self._send_sse(fake.sse())
            return
        if self.path == "/v1/completions":
            if fake.completion_error is not None:
                self._send(fake.completion_error[0], {"error": fake.completion_error[1]})
                return
            self._send_sse(fake.sse())
            return
        self._send(404, {"error": "not found"})


class FakeRocmServer:
    """A running fake endpoint. Use as a context manager (sync or async)."""

    def __init__(self, models: tuple[str, ...] = (UPSTREAM_MODEL,)) -> None:
        self.fake = _FakeUpstream(models=models)
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.fake = self.fake  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeRocmServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _config(server: FakeRocmServer, **over: Any) -> RocmEngineConfig:
    # verify=False: the fake speaks plain HTTP and this test host has no CA
    # bundle; the engine's default stays True for real deployments.
    data = {"base_url": server.base_url, "model": UPSTREAM_MODEL, "verify": False, **over}
    return RocmEngineConfig.from_dict(data)


async def _collect(engine: RocmEngine, **kw: Any):
    outputs = [o async for o in engine.stream_chat(**kw)]
    return outputs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_from_dict_aliases() -> None:
    cfg = RocmEngineConfig.from_dict(
        {"url": "http://box:8731/", "model_name": "m", "api_key": "k"}
    )
    assert cfg.base_url == "http://box:8731"
    assert cfg.upstream_model == "m"
    assert cfg.api_key == "k"
    assert cfg.headers()["Authorization"] == "Bearer k"


def test_config_requires_base_url_and_model() -> None:
    for bad in ({}, {"base_url": "http://x"}, {"model": "m"}):
        try:
            RocmEngineConfig.from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_config_accepts_extra_headers() -> None:
    cfg = RocmEngineConfig.from_dict(
        {
            "base_url": "http://x",
            "model": "m",
            "extra_headers": {"X-Node": "valhalla"},
        }
    )
    assert dict(cfg.extra_headers) == {"X-Node": "valhalla"}
    assert cfg.headers()["X-Node"] == "valhalla"


# --------------------------------------------------------------------------- #
# Lifecycle / readiness
# --------------------------------------------------------------------------- #


def test_start_advertises_model() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                assert engine.has_active_requests() is False
                assert engine.get_stats()["running"] is True
                models = await engine.list_models()
                assert UPSTREAM_MODEL in engine._model_ids(models)
            finally:
                await engine.stop()
            assert engine.get_stats()["running"] is False

    asyncio.run(main())


def test_start_rejects_missing_model() -> None:
    async def main() -> None:
        with FakeRocmServer(models=("other-model",)) as server:
            engine = RocmEngine(_config(server))
            try:
                await engine.start()
            except RocmEngineError as exc:
                assert "does not advertise" in str(exc)
                assert "other-model" in str(exc)
            else:
                raise AssertionError("start() should have failed")

    asyncio.run(main())


def test_start_rejects_unreachable() -> None:
    async def main() -> None:
        # Port 1 is reserved and nothing listens there.
        cfg = RocmEngineConfig(
            base_url="http://127.0.0.1:1", upstream_model=UPSTREAM_MODEL,
            start_timeout_s=1.0, connect_timeout_s=0.5, verify=False,
        )
        engine = RocmEngine(cfg)
        try:
            await engine.start()
        except RocmEngineError as exc:
            assert "not ready" in str(exc)
        else:
            raise AssertionError("start() should have failed")

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Chat streaming
# --------------------------------------------------------------------------- #


def test_stream_chat_relays_pieces_and_finalized_stats() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                messages = [{"role": "user", "content": "hi"}]
                outputs = [
                    o async for o in engine.stream_chat(messages, max_tokens=16)
                ]
                # 3 content pieces + the terminal output.
                assert [o.new_text for o in outputs[:-1]] == ["Hello", ", ", "world"]
                assert all(o.finished is False for o in outputs[:-1])
                final = outputs[-1]
                assert final.finished is True
                assert final.text == "Hello, world"
                assert final.finish_reason == "stop"
                assert final.prompt_tokens == 11
                assert final.completion_tokens == 3
                assert final.cached_tokens == 4
                assert final.prompt_tps == 120.5
                assert final.generation_tps == 39.7
                assert final.first_token_at is not None
            finally:
                await engine.stop()

    asyncio.run(main())


def test_chat_collects_full_text() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                out = await engine.chat([{"role": "user", "content": "hi"}])
                assert out.text == "Hello, world"
                assert out.new_text == "Hello, world"
                assert out.finished is True
                assert engine.get_stats()["requests"] == 1
            finally:
                await engine.stop()

    asyncio.run(main())


def test_tool_calls_are_accumulated_across_chunks() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            server.fake.pieces = []
            server.fake.tool_call_fragments = [
                [
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "lookup", "arguments": '{"q":'}},
                    {"index": 0, "function": {"arguments": ' "cats"}'}},
                ]
            ]
            server.fake.finish_reason = "tool_calls"
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                out = await engine.chat(
                    [{"role": "user", "content": "weather?"}],
                    tools=[{"type": "function", "function": {"name": "lookup"}}],
                )
                assert out.finish_reason == "tool_calls"
                assert out.tool_calls is not None and len(out.tool_calls) == 1
                call = out.tool_calls[0]
                # oMLX's parser wants the flat {"id","name","arguments"} shape.
                assert call["id"] == "call_1"
                assert call["name"] == "lookup"
                assert json.loads(call["arguments"]) == {"q": "cats"}
            finally:
                await engine.stop()

    asyncio.run(main())


def test_tools_rejected_when_unsupported() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server, supports_tools=False))
            await engine.start()
            try:
                try:
                    await engine.chat(
                        [{"role": "user", "content": "hi"}],
                        tools=[{"type": "function", "function": {"name": "x"}}],
                    )
                except InvalidRequestError as exc:
                    assert "does not accept tools" in str(exc)
                else:
                    raise AssertionError("expected InvalidRequestError")
            finally:
                await engine.stop()

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Reasoning / thinking
# --------------------------------------------------------------------------- #


def test_reasoning_content_is_wrapped_in_think_tags() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            server.fake.reasoning_pieces = ["Let me ", "think."]
            server.fake.pieces = ["An", "swer"]
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                outputs = [
                    o async for o in engine.stream_chat(
                        [{"role": "user", "content": "hi"}]
                    )
                ]
                streamed = [o.new_text for o in outputs[:-1]]
                # Reasoning is wrapped once, then content closes the block.
                assert streamed == [
                    "<think>Let me ",
                    "think.",
                    "</think>An",
                    "swer",
                ]
                final = outputs[-1]
                assert final.finished is True
                assert final.text == "<think>Let me think.</think>Answer"
            finally:
                await engine.stop()

    asyncio.run(main())


def test_reasoning_only_response_is_closed_at_end() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            server.fake.reasoning_pieces = ["still ", "thinking"]
            server.fake.pieces = []
            server.fake.finish_reason = "length"
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                final = await engine.chat([{"role": "user", "content": "hi"}])
                # The budget ran out mid-reasoning; the block must be closed.
                assert final.finish_reason == "length"
                assert final.text == "<think>still thinking</think>"
            finally:
                await engine.stop()

    asyncio.run(main())


def test_chat_template_kwargs_are_forwarded_without_reasoning_effort() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                async for _ in engine.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    chat_template_kwargs={
                        "enable_thinking": False,
                        "reasoning_effort": "xhigh",
                    },
                ):
                    pass
                body = server.fake.last_body
                assert body["chat_template_kwargs"] == {"enable_thinking": False}
                assert body["reasoning_effort"] == "high"
            finally:
                await engine.stop()

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Plain completions
# --------------------------------------------------------------------------- #


def test_stream_generate_uses_completions_endpoint() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                out = await engine.generate("write a poem", max_tokens=8)
                assert out.text == "Hello, world"
                path, body = server.fake.requests[-1]
                assert path == "/v1/completions"
                assert body["prompt"] == "write a poem"
                assert body["stream"] is True
            finally:
                await engine.stop()

    asyncio.run(main())


def test_generate_rejects_token_ids() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                try:
                    await engine.generate([1, 2, 3])
                except InvalidRequestError as exc:
                    assert "text prompt" in str(exc)
                else:
                    raise AssertionError("expected InvalidRequestError")
            finally:
                await engine.stop()

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #


def test_request_body_forwards_sampling_and_extensions() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                async for _ in engine.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=32,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    min_p=0.05,
                    repetition_penalty=1.1,
                    presence_penalty=0.4,
                    stop=["</s>"],
                    seed=7,
                ):
                    pass
                body = server.fake.last_body
                assert body["model"] == UPSTREAM_MODEL
                assert body["max_tokens"] == 32
                assert body["temperature"] == 0.3
                assert body["top_p"] == 0.8
                assert body["top_k"] == 40
                assert body["min_p"] == 0.05
                assert body["repetition_penalty"] == 1.1
                assert body["presence_penalty"] == 0.4
                assert body["stop"] == ["</s>"]
                assert body["seed"] == 7
                assert body["stream"] is True
            finally:
                await engine.stop()

    asyncio.run(main())


def test_request_body_omits_neutral_defaults() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                async for _ in engine.stream_chat([{"role": "user", "content": "hi"}]):
                    pass
                body = server.fake.last_body
                # Neutral values must not be sent as llama.cpp extensions.
                for key in ("top_k", "min_p", "repetition_penalty", "presence_penalty"):
                    assert key not in body
            finally:
                await engine.stop()

    asyncio.run(main())


def test_reasoning_effort_is_mapped_and_forwarded() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                async for _ in engine.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    chat_template_kwargs={"reasoning_effort": "xhigh"},
                ):
                    pass
                assert server.fake.last_body["reasoning_effort"] == "high"
            finally:
                await engine.stop()

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_upstream_4xx_becomes_invalid_request() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            server.fake.chat_error = (400, "context length exceeded")
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                try:
                    await engine.chat([{"role": "user", "content": "hi"}])
                except InvalidRequestError as exc:
                    assert "context length exceeded" in str(exc)
                else:
                    raise AssertionError("expected InvalidRequestError")
            finally:
                await engine.stop()

    asyncio.run(main())


def test_upstream_5xx_becomes_rocm_error() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            server.fake.chat_error = (503, "server busy")
            engine = RocmEngine(_config(server))
            await engine.start()
            try:
                try:
                    await engine.chat([{"role": "user", "content": "hi"}])
                except RocmEngineError as exc:
                    assert "503" in str(exc)
                else:
                    raise AssertionError("expected RocmEngineError")
            finally:
                await engine.stop()

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# Token counting / stats / probe
# --------------------------------------------------------------------------- #


def test_count_chat_tokens_estimates_without_tokenizer() -> None:
    engine = RocmEngine(
        RocmEngineConfig(base_url="http://x", upstream_model=UPSTREAM_MODEL)
    )
    n = engine.count_chat_tokens([{"role": "user", "content": "a" * 40}])
    assert n == 10  # ~4 chars per token


def test_active_request_tracking_and_stats() -> None:
    async def main() -> None:
        with FakeRocmServer() as server:
            engine = RocmEngine(_config(server, label="valhalla (ROCm)"))
            await engine.start()
            try:
                assert engine.has_active_requests() is False
                stats = engine.get_stats()
                assert stats["engine_type"] == "rocm"
                assert stats["label"] == "valhalla (ROCm)"
                assert stats["base_url"] == server.base_url
                assert stats["upstream_model"] == UPSTREAM_MODEL
                assert engine.get_cache_stats() is None
            finally:
                await engine.stop()

    asyncio.run(main())


def test_probe_endpoint_reports_models() -> None:
    with FakeRocmServer(models=(UPSTREAM_MODEL, "qwen3-27b")) as server:
        result = asyncio.run(probe_rocm_endpoint(server.base_url, verify=False))
    assert result["ok"] is True
    assert {m["id"] for m in result["models"]} == {UPSTREAM_MODEL, "qwen3-27b"}


def test_probe_endpoint_never_raises() -> None:
    result = asyncio.run(
        probe_rocm_endpoint("http://127.0.0.1:1", timeout_s=0.5, verify=False)
    )
    assert result["ok"] is False
    assert result["models"] == []
    assert result["detail"]


# --------------------------------------------------------------------------- #
# Standalone harness (no pytest needed)
# --------------------------------------------------------------------------- #


def _run_standalone() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
