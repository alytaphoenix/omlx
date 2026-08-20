# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N806, SIM300 -- the ported reference section below (down to
# _stream_generate) keeps z-lab's original single-letter tensor-dimension
# names (B, L, S, K, A_log, ...) and comparison order verbatim for upstream
# diffability, matching the vendored-file convention used elsewhere in this
# repo (e.g. omlx/patches/qwen35_q4_mlp.py).
"""
DFlash 2 engine: independent block-diffusion speculative decoding.

DFlash 2 (z-lab, released two days after DFlash 1) adds a Path Selector +
Two-Tap Dynamic Convolution head (~3% of the drafter's size) on top of the
DFlash 1 draft architecture. That head makes each block's per-position draft
predictions mutually consistent instead of independent, which the z-lab
paper reports as a 21% acceptance-length gain over DFlash 1 (2.7-3.4x vs
autoregressive) on Qwen3.8-27B.

Unlike ``DFlashEngine`` (``omlx/engine/dflash.py``), this engine does **not**
wrap the pinned ``dflash-mlx`` package. DFlash 2's draft-model protocol
(hidden-state fusion via ``target_layer_ids`` hooks, Path Selector, grouped
dynamic convolution) is incompatible with dflash-mlx's draft-model contract,
and z-lab's own reference implementation
(``z-lab/dflash``, ``dflash/model_mlx.py``) is a complete, self-contained
generation engine built directly on ``mlx_lm`` primitives (``KVCache``,
``RotatingKVCache``, ``create_causal_mask``, ...) rather than a plugin for
dflash-mlx's ``spec_epoch`` runtime.

The classes and functions below (down to ``_stream_generate``) are a close
port of that reference implementation. They intentionally do not get
omlx's prefix L1/L2 cache, dflash-mlx's runtime cache manager, or any of
DFlashEngine's fallback-to-VLM machinery — this is model-level KV/GDN
cache state on ``self._target_model`` shared across a whole ``generate()``
call, which (like DFlashEngine) makes this engine single-stream only: one
in-flight request per loaded ``DFlash2Engine`` instance.
"""

import asyncio
import copy
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import mlx.core as mx
import mlx_lm.models.gated_delta as _gd_mod
from huggingface_hub import snapshot_download
from mlx import nn
from mlx_lm.generate import generation_stream, wired_limit
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
)
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import detect_and_strip_partial
from ..memory_monitor import (
    MemoryMonitor,
    raise_if_prefill_exceeds,
    set_model_info_from_model,
)
from ..reasoning_effort import apply_chat_template_with_reasoning_effort_fallback
from ..utils.model_loading import (
    apply_post_load_transforms,
    lm_load_compat,
    materialize_lazy_state,
    maybe_apply_pre_load_patches,
)
from ..utils.proc_memory import get_phys_footprint
from ..utils.tokenizer import get_tokenizer_config
from .base import (
    ActivityTrackingMixin,
    BaseEngine,
    GenerationOutput,
    _warn_scheduler_unreachable_once,
)

logger = logging.getLogger(__name__)

_GDN_PATCH_LOCK = RLock()


# =============================================================================
# Ported from z-lab/dflash's dflash/model_mlx.py (DFlash 2 reference impl).
# Kept close to the original so future upstream diffs are easy to re-apply;
# resist "cleaning up" the arithmetic below (offset bookkeeping in
# _prefill_target / _stream_generate is the correctness core).
# =============================================================================


def _sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    scores = logits.astype(mx.float32) / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        indices = mx.argpartition(-scores, top_k - 1, axis=-1)[..., :top_k]
        scores = mx.take_along_axis(scores, indices, axis=-1)
    else:
        indices = None

    probs = mx.softmax(scores, axis=-1)
    if top_p < 1.0:
        order = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, order, axis=-1)
        keep = mx.cumsum(sorted_probs, axis=-1) - sorted_probs < top_p
        sorted_probs = mx.where(keep, sorted_probs, 0)
        probs = mx.put_along_axis(mx.zeros_like(probs), order, sorted_probs, axis=-1)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)

    if indices is not None:
        probs = mx.put_along_axis(
            mx.zeros(logits.shape, dtype=probs.dtype), indices, probs, axis=-1
        )
    return probs


def _sample_probs(probs):
    return mx.random.categorical(mx.log(probs))


def _sample_logits(logits, temperature=0.0, top_p=1.0, top_k=0):
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    return _sample_probs(_sampling_probs(logits, temperature, top_p, top_k))


def make_sampler(temperature=0.0, top_p=1.0, top_k=0):
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("temperature and top_k must be non-negative, and top_p in (0, 1].")
    return lambda logits: _sample_logits(logits, temperature, top_p, top_k)


def _rejection_sample(draft_tokens, target_probs, draft_probs, draft_indices=None):
    gamma = draft_tokens.shape[1]
    p = mx.take_along_axis(
        target_probs[:, :gamma], draft_tokens[..., None], axis=-1
    )[..., 0]
    if draft_indices is None:
        q = mx.take_along_axis(
            draft_probs, draft_tokens[..., None], axis=-1
        )[..., 0]
    else:
        q = mx.sum(
            draft_probs * (draft_indices == draft_tokens[..., None]), axis=-1
        )
    accepted = mx.sum(
        mx.cumprod(
            (mx.random.uniform(shape=q.shape) * q < p).astype(mx.int32), axis=-1
        ),
        axis=-1,
    )[0].item()
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0].item()

    residual = target_probs[0, accepted]
    if draft_indices is None:
        residual = residual - draft_probs[0, accepted]
    else:
        indices = draft_indices[0, accepted]
        values = mx.take(residual, indices) - draft_probs[0, accepted]
        residual = mx.put_along_axis(
            residual[None], indices[None], values[None], axis=-1
        )[0]
    residual = mx.maximum(residual, 0)
    total = mx.sum(residual)
    residual = mx.where(
        total > 0,
        residual / mx.maximum(total, 1e-30),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual[None])[0].item()


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    is_causal: bool | None = None


def _build_rope(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: dict[str, Any] | None,
):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = self.is_sliding if config.is_causal is None else config.is_causal
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        mask = create_causal_mask(L, offset=ctx_len) if self.is_causal else None
        if self.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            key = mx.arange(ctx_len + L)[None]
            context = (key < ctx_len) & (query - key < self.sliding_window)
            block = key >= ctx_len
            if self.is_causal:
                block = block & (key <= query)
            mask = context | block
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else mx.concatenate(
                (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]), axis=1
            )
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(
                hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache):
        residual = x
        x, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(
            self.self_attn(x, x_ctx, rope, cache), kernel
        )
        residual = x
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(config.hidden_size, config.selector_rank, bias=False)

    def select(self, hidden, logits, anchor_ids, temperature):
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path, q_rows = [], []
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            if temperature > 0:
                q = _sampling_probs(scores[:, None], temperature)[:, 0]
                selected = _sample_probs(q)
                q_rows.append(q)
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return (
            mx.stack(path, axis=1),
            candidates,
            mx.stack(q_rows, axis=1) if q_rows else None,
        )


class DFlashDraftModel(nn.Module):
    layer_class = DFlashDecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ("full_attention",) * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [self.layer_class(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = (
            getattr(self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0))
            * self.config.input_embedding_scale
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
                caches.append(RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0))
            else:
                caches.append(KVCache())
        return caches

    def hidden_states(
        self,
        inputs,
        target_hidden,
        cache,
        logits_start: int = 0,
    ):
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        return self.norm(h)

    def __call__(self, inputs, target_hidden, cache, logits_start: int = 0):
        return self.compute_logits(
            self.hidden_states(inputs, target_hidden, cache, logits_start)
        )

    def compute_logits(self, hidden):
        logits = self.lm_head(hidden)
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None and self.config.final_logit_softcapping > 0:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def propose(self, inputs, target_hidden, cache, temperature, logits_start: int = 0):
        hidden = self.hidden_states(inputs, target_hidden, cache, logits_start)
        return self.candidate_selector.select(
            hidden, self.compute_logits(hidden), inputs[:, 0], temperature
        )


def _resolve_draft_snapshot_dir(draft_id: str) -> Path:
    """Resolve ``draft_id`` to a local directory with a ``config.json``.

    Mirrors ``mlx_lm.load()``'s own dual behaviour: a local directory is used
    directly (the common case — the admin UI's draft-model picker always
    supplies an already-downloaded local ``model_path``), and anything else
    is treated as a Hub repo id and fetched via ``snapshot_download``. z-lab's
    own reference ``load_draft`` always calls ``snapshot_download``, which
    rejects arbitrary local directories, so this indirection is the one
    deliberate deviation from the reference port.
    """
    local = Path(draft_id).expanduser()
    if local.is_dir() and (local / "config.json").is_file():
        return local
    return Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))


def load_draft(draft_id: str) -> DFlashDraftModel:
    path = _resolve_draft_snapshot_dir(draft_id)
    cfg = json.loads((path / "config.json").read_text())
    dflash = cfg.get("dflash_config", {})
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    if len(layer_types) != cfg["num_hidden_layers"]:
        raise ValueError("Draft config layer_types length must match num_hidden_layers.")
    unknown_layer_types = set(layer_types) - {"full_attention", "sliding_attention"}
    if unknown_layer_types:
        raise ValueError(f"Unsupported draft layer_types: {sorted(unknown_layer_types)}.")
    if "sliding_attention" in layer_types and cfg.get("sliding_window") is None:
        raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg.get("rope_theta", (rope or {}).get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=int(dflash.get("block_size", cfg.get("block_size", 16))),
        target_layer_ids=tuple(dflash["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dflash["mask_token_id"],
        rope_scaling=rope,
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=dflash.get(
            "final_logit_softcapping", cfg.get("final_logit_softcapping")
        ),
        input_embedding_scale=float(dflash.get("input_embedding_scale", 1.0)),
        output_multiplier=float(dflash.get("output_multiplier", 1.0)),
        conv_kernel_size=int(dflash.get("conv_kernel_size", 0)),
        conv_group_size=int(dflash.get("conv_group_size", 0)),
        selector_rank=int(dflash.get("selector_rank", 0)),
        selector_top_k=int(dflash.get("selector_top_k", 0)),
        is_causal=cfg.get("is_causal"),
    )
    weights = {k: v for f in path.glob("*.safetensors") for k, v in mx.load(str(f)).items()}
    model_class = (
        DFlash2DraftModel
        if "DFlash2DraftModel" in (cfg.get("architectures") or [])
        else DFlashDraftModel
    )
    if model_class is DFlash2DraftModel:
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            weights[f"{key}.weight"] = weights.pop(key)
    model = model_class(config)
    model.eval()
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def _trim_recent_cache(cache: list[Any], num_tokens: int) -> None:
    if num_tokens <= 0:
        return
    for c in cache:
        n = min(getattr(c, "offset", num_tokens), num_tokens)
        if n <= 0:
            continue
        if isinstance(c, RotatingKVCache) and c.keys is not None:
            c.keys = c._temporal_order(c.keys)
            c.values = c._temporal_order(c.values)
            c.keys = c.keys[..., :-n, :]
            c.values = c.values[..., :-n, :]
            c.offset -= n
            c._idx = c.keys.shape[2]
        elif hasattr(c, "trim"):
            c.trim(n)


class _LayerHook:
    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out[0] if isinstance(out, tuple) else out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _patch_model(model, layer_ids):
    if hasattr(model, "_hidden_states"):
        return
    model._hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._hidden_states)


class _GDNStateCapture:
    def __init__(self):
        self.conv_data = []
        self._gdn_inputs = []
        self._gdn_cls = None
        self._orig_call = None
        self._patched_call = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            _GDN_PATCH_LOCK.release()
            raise

    def _patch(self):
        from mlx_lm.models.qwen3_5 import GatedDeltaNet

        self._gdn_cls = GatedDeltaNet
        self._orig_call = GatedDeltaNet.__call__
        capture = self

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            if self_layer.sharding_group is not None:
                from mlx_lm.models.qwen3_5 import sum_gradients

                inputs = sum_gradients(self_layer.sharding_group)(inputs)
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(B, S, self_layer.num_v_heads, self_layer.head_v_dim)
            b, a = self_layer.in_proj_b(inputs), self_layer.in_proj_a(inputs)
            conv_state = (
                cache[0]
                if (cache is not None and cache[0] is not None)
                else mx.zeros((B, self_layer.conv_kernel_size - 1, self_layer.conv_dim), dtype=inputs.dtype)
            )
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                n_keep = self_layer.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:])
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            capture._gdn_inputs.append((q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask))
            out, new_state = _gd_mod.gated_delta_update(
                q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask,
                use_kernel=not self_layer.training,
            )
            if cache is not None:
                cache[1] = new_state
                cache.advance(S)
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if self_layer.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        self._patched_call = _capturing_gdn_call
        GatedDeltaNet.__call__ = _capturing_gdn_call

    def clear(self):
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self):
        if self._closed:
            return
        try:
            if self._gdn_cls is not None and self._gdn_cls.__call__ is self._patched_call:
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._gdn_cls = None
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def rollback(self, cache, accepted, trim):
        n_non_trimmable = sum(1 for c in cache if not c.is_trimmable())
        assert n_non_trimmable == len(self._gdn_inputs), (
            f"non-trimmable cache count ({n_non_trimmable}) != "
            f"captured GDN inputs ({len(self._gdn_inputs)}); "
            "DFlash 2 rollback assumes every non-trimmable cache is a GatedDeltaNet layer"
        )
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                q, k, v, a, b, A_log, dt_bias, init_state, mask = self._gdn_inputs[j]
                n = accepted + 1
                _, state = _gd_mod.gated_delta_update(
                    q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                    A_log, dt_bias, init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=True,
                )
                c.cache[1] = state
                conv_input, K = self.conv_data[j]
                c.cache[0] = mx.contiguous(
                    conv_input[:, accepted + 1 : accepted + K]
                )
                j += 1


@dataclass
class GenerationResponse:
    text: str
    tokens: list[int]
    accepted: int | None
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: str | None = None


def _make_response(
    text,
    tokens,
    accepted,
    prompt_size,
    prompt_tps,
    n,
    tic,
    finish_reason=None,
):
    return GenerationResponse(
        text, tokens, accepted, prompt_size, prompt_tps,
        n, n / (time.perf_counter() - tic), mx.get_peak_memory() / 1e9, finish_reason,
    )


def _prefill_target(model, prompt, cache, hidden_limit, step_size):
    if step_size <= 0:
        raise ValueError("prefill_step_size must be positive.")
    hidden_chunks = []
    start = 0
    while start < prompt.size:
        remaining = prompt.size - start
        end = start + (1 if remaining == 1 else min(step_size, remaining - 1))
        logits = model(prompt[None, start:end], cache)
        hidden = mx.concatenate(model._hidden_states, axis=-1)
        if hidden_limit is None:
            hidden_chunks.append(hidden)
        else:
            if hidden_chunks:
                hidden = mx.concatenate((hidden_chunks[0], hidden), axis=1)
            hidden_chunks = [hidden[:, -hidden_limit:]]
        if end < prompt.size:
            mx.eval([c.state for c in cache], hidden_chunks[-1])
            mx.clear_cache()
        start = end
    hidden = (
        hidden_chunks[0]
        if len(hidden_chunks) == 1
        else mx.concatenate(hidden_chunks, axis=1)
    )
    return logits[:, -1:], hidden, prompt.size - hidden.shape[1]


def stream_generate(
    model, draft, tokenizer, prompt,
    block_size=None, max_tokens=256, temperature=0.0, top_p=1.0, top_k=0,
    prefill_step_size=2048,
):
    with wired_limit(model, [generation_stream]):
        yield from _stream_generate(
            model, draft, tokenizer, prompt,
            block_size, max_tokens, temperature, top_p, top_k,
            prefill_step_size,
        )


def _stream_generate(
    model, draft, tokenizer, prompt,
    block_size=None, max_tokens=256, temperature=0.0, top_p=1.0, top_k=0,
    prefill_step_size=2048,
):
    _patch_model(model, draft.config.target_layer_ids)
    block_size = block_size if block_size is not None else int(draft.config.block_size)
    sampler = make_sampler(temperature, top_p, top_k)

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    tokens = []

    target_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft)
    draft.bind(model)
    _target_can_trim = can_trim_prompt_cache(target_cache)
    _capture = None

    try:
        tic = time.perf_counter()
        with mx.stream(generation_stream):
            hidden_limit = (
                draft.config.sliding_window - 1
                if all(t == "sliding_attention" for t in draft.config.layer_types)
                else None
            )
            logits, hidden, hidden_offset = _prefill_target(
                model, prompt, target_cache, hidden_limit, prefill_step_size
            )
            for cache in draft_cache:
                cache.offset = hidden_offset
        mx.eval(logits, hidden)
        prompt_tps = prompt.size / (time.perf_counter() - tic)

        tic = time.perf_counter()
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n = 1

        if token in tokenizer.eos_token_ids:
            detokenizer.add_token(token)
            detokenizer.finalize()
            yield _make_response(detokenizer.last_segment, [token], None, prompt.size, prompt_tps, n, tic, "stop")
            return

        detokenizer.add_token(token)
        yield _make_response(
            detokenizer.last_segment,
            [token],
            None,
            prompt.size,
            prompt_tps,
            n,
            tic,
        )

        if not _target_can_trim:
            _capture = _GDNStateCapture()

        while n < max_tokens:
            bs = min(block_size, max_tokens - n + 1)
            if bs <= 1:
                break

            with mx.stream(generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                if isinstance(draft, DFlash2DraftModel):
                    draft_tokens, draft_indices, draft_probs = draft.propose(
                        block, hidden, draft_cache, temperature, logits_start=1
                    )
                else:
                    draft_logits = draft(block, hidden, draft_cache, logits_start=1)
                    if temperature > 0:
                        draft_probs = _sampling_probs(
                            draft_logits, temperature, top_p, top_k
                        )
                        draft_tokens = _sample_probs(draft_probs)
                        draft_indices = None
                    else:
                        draft_tokens = mx.argmax(draft_logits, axis=-1)
                if (trim_n := draft_cache[0].offset - (prompt.size + n - 1)) > 0:
                    _trim_recent_cache(draft_cache, trim_n)
            mx.async_eval(draft_tokens)

            if _capture is not None:
                _capture.clear()
            with mx.stream(generation_stream):
                verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
                logits = model(verify_input, target_cache)
                hidden = mx.concatenate(model._hidden_states, axis=-1)
                if temperature > 0:
                    target_probs = _sampling_probs(logits, temperature, top_p, top_k)
                else:
                    target_tokens = mx.argmax(logits, axis=-1)
            mx.async_eval(target_probs if temperature > 0 else target_tokens, hidden)

            d_list = draft_tokens[0].tolist()
            if temperature > 0:
                accepted, bonus = _rejection_sample(
                    draft_tokens, target_probs, draft_probs, draft_indices
                )
            else:
                t_list = target_tokens[0].tolist()
                accepted = next(
                    (i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                    len(d_list),
                )
                bonus = t_list[accepted]
            new_tokens = d_list[:accepted] + [bonus]
            new_tokens = new_tokens[: max_tokens - n]

            eos_idx = next((i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids), None)
            if eos_idx is not None:
                new_tokens = new_tokens[: eos_idx + 1]
                for t in new_tokens:
                    detokenizer.add_token(t)
                detokenizer.finalize()
                tokens.extend(new_tokens)
                n += len(new_tokens)
                yield _make_response(
                    detokenizer.last_segment,
                    new_tokens,
                    len(new_tokens),
                    prompt.size,
                    prompt_tps,
                    n,
                    tic,
                    "stop",
                )
                return

            for t in new_tokens:
                detokenizer.add_token(t)
            tokens.extend(new_tokens)
            previous_n = n
            n += len(new_tokens)

            if n // 256 > previous_n // 256:
                mx.clear_cache()

            yield _make_response(
                detokenizer.last_segment,
                new_tokens,
                len(new_tokens),
                prompt.size,
                prompt_tps,
                n,
                tic,
            )

            trim = bs - accepted - 1
            if trim > 0:
                if _target_can_trim:
                    _trim_recent_cache(target_cache, trim)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, : accepted + 1, :]

        detokenizer.finalize()
        yield _make_response(
            detokenizer.last_segment,
            [],
            None,
            prompt.size,
            prompt_tps,
            n,
            tic,
            "length",
        )
    finally:
        if _capture is not None:
            _capture.close()


# =============================================================================
# End of ported reference code. Everything below is omlx-specific engine
# scaffolding.
# =============================================================================


class DFlash2Engine(ActivityTrackingMixin, BaseEngine):
    """DFlash 2 speculative decoding engine.

    Loads the target model through omlx's normal LM loading path
    (``lm_load_compat`` + the same post-load transforms ``BatchedEngine``
    applies) and the draft model through the ported ``load_draft`` above.
    Generation runs the ported ``stream_generate`` loop on the shared MLX
    executor thread, exactly like ``DFlashEngine`` runs dflash-mlx's event
    iterator: the whole request is driven synchronously on that one thread
    and results are handed back to the event loop via an ``asyncio.Queue``.

    Single-stream only, for the same structural reason as ``DFlashEngine``:
    ``_patch_model`` stores per-target-model hook state
    (``model._hidden_states``) directly on ``self._target_model``, and when
    the target has GatedDeltaNet layers ``_GDNStateCapture`` monkey-patches
    ``GatedDeltaNet.__call__`` process-wide for the duration of a request.
    Both are safe for one in-flight request per loaded instance, not for
    concurrent requests sharing the same target model.
    """

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        model_settings: Any | None = None,
        scheduler_config: Any | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._model_settings = model_settings
        # Snapshot now: engine_pool mutates the shared scheduler_config
        # instance (model_name/model_path) on every model load.
        self._scheduler_config = (
            copy.copy(scheduler_config) if scheduler_config else scheduler_config
        )

        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        self._active_request = False
        self._prefill_guard = None
        self._prefill_step_size = 2048

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer_obj

    @property
    def model_type(self) -> str | None:
        """Get the model type from config (e.g. 'qwen3', 'qwen3_5')."""
        model = self._target_model
        if model is None:
            return None
        try:
            if hasattr(model, "config"):
                config = model.config
                if hasattr(config, "model_type"):
                    model_type = config.model_type
                    return model_type if isinstance(model_type, str) else None
                elif isinstance(config, dict):
                    model_type = config.get("model_type")
                    return model_type if isinstance(model_type, str) else None
            if hasattr(model, "args"):
                args = model.args
                if hasattr(args, "model_type"):
                    model_type = args.model_type
                    return model_type if isinstance(model_type, str) else None
        except Exception as e:
            logger.debug(f"DFlash2Engine: error getting model_type: {e}")
        return None

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        trust_remote_code = bool(
            getattr(self._model_settings, "trust_remote_code", False)
        )

        def _load_models():
            maybe_apply_pre_load_patches(
                self._model_name, model_settings=self._model_settings
            )
            tokenizer_config = get_tokenizer_config(
                self._model_name, trust_remote_code=trust_remote_code
            )
            model, tokenizer = lm_load_compat(
                self._model_name,
                tokenizer_config=tokenizer_config,
                trust_remote_code=trust_remote_code,
            )
            model = apply_post_load_transforms(model, self._model_settings)
            materialize_lazy_state(model)

            # Same MoE gate+up fusion BatchedEngine/DFlashEngine apply to
            # targets: fuses gate+up projections for routed experts so
            # DFlash2's block-verify forward pass (a decode-shaped batch)
            # runs 2 gather_qmm launches per MoE layer instead of 3.
            if (
                getattr(self._model_settings, "moe_gate_up_fusion_enabled", True)
                is not False
            ):
                try:
                    from ..patches.qwen35_moe_gate_up import (
                        apply_qwen35_moe_gate_up_fusion,
                    )

                    apply_qwen35_moe_gate_up_fusion(model)
                except Exception:
                    logger.debug(
                        "DFlash2 target MoE gate+up fusion not applied",
                        exc_info=True,
                    )

            draft = load_draft(self._draft_model_path)
            return model, tokenizer, draft

        self._target_model, self._tokenizer_obj, self._draft_model = (
            await loop.run_in_executor(get_mlx_executor(), _load_models)
        )

        # Deep-copy tokenizer for executor-thread usage (generation), mirrors
        # DFlashEngine — the original self._tokenizer_obj stays for
        # event-loop operations (encode, apply_chat_template).
        # See: https://github.com/huggingface/tokenizers/issues/537
        self._executor_tokenizer = copy.deepcopy(self._tokenizer_obj)

        self._prefill_step_size = int(
            getattr(self._scheduler_config, "prefill_step_size", 2048) or 2048
        )

        # Primary-mode prefill memory guard. DFlash2 bypasses the scheduler
        # entirely, same as DFlashEngine, so it needs its own guard.
        self._prefill_guard = None
        try:
            monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
            set_model_info_from_model(monitor, self._target_model)
            self._prefill_guard = _DFlash2PrefillGuard(monitor, self._prefill_step_size)
        except Exception as exc:
            logger.warning(f"DFlash2 prefill guard init failed: {exc}")
            self._prefill_guard = None

        self._loaded = True
        logger.info(
            f"DFlash2Engine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"block_size={int(self._draft_model.config.block_size)}"
        )

    async def stop(self) -> None:
        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._prefill_guard = None
        self._loaded = False
        logger.info("DFlash2Engine stopped")

    def _record_prefill_guard_active_memory(self) -> None:
        guard = self._prefill_guard
        if guard is None:
            return
        try:
            guard.record_mlx_active_memory(mx.get_active_memory())
        except Exception as exc:
            logger.debug(f"DFlash2 active-memory sample failed: {exc}")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> str:
        """Apply chat template to messages. Mirrors DFlashEngine's version."""
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            if is_partial is None:
                is_partial = detect_and_strip_partial(messages)
            else:
                for msg in messages:
                    msg.pop("partial", None)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            try:
                return apply_chat_template_with_reasoning_effort_fallback(
                    self._tokenizer_obj,
                    messages,
                    template_kwargs,
                    is_harmony=self.model_type == "gpt_oss",
                )
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
        return len(self._tokenizer_obj.encode(prompt))

    async def preflight_chat(
        self,
        messages: list,
        tools: list | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> None:
        """Prefill-memory preflight for chat requests. Mirrors DFlashEngine."""
        if not self._loaded:
            await self.start()
        if self._prefill_guard is None:
            _warn_scheduler_unreachable_once(
                self, "preflight_chat", "primary-mode prefill guard unavailable"
            )
            return
        try:
            num_tokens = self.count_chat_tokens(
                messages,
                tools,
                chat_template_kwargs=kwargs.get("chat_template_kwargs"),
                is_partial=kwargs.get("is_partial"),
            )
        except Exception as e:
            logger.warning(
                "DFlash2Engine.preflight_chat: token count raised %s; skipping "
                "prefill memory check, real chat path will surface the error",
                type(e).__name__,
            )
            return
        self._prefill_guard.preflight_or_raise(
            num_prompt_tokens=num_tokens, request_id=request_id
        )

    async def preflight_completion(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs,
    ) -> None:
        """Prefill-memory preflight for plain completions. See ``preflight_chat``."""
        if not self._loaded:
            await self.start()
        if self._prefill_guard is None:
            _warn_scheduler_unreachable_once(
                self, "preflight_completion", "primary-mode prefill guard unavailable"
            )
            return
        try:
            num_tokens = len(self._tokenizer_obj.encode(prompt))
        except Exception as e:
            logger.warning(
                "DFlash2Engine.preflight_completion: tokenizer.encode raised %s; "
                "skipping prefill memory check, real completion path will "
                "surface the error",
                type(e).__name__,
            )
            return
        self._prefill_guard.preflight_or_raise(
            num_prompt_tokens=num_tokens, request_id=request_id
        )

    def _get_think_token_id(self, attr: str) -> int | None:
        try:
            return getattr(self._tokenizer_obj, attr, None)
        except (ValueError, TypeError):
            return None

    def _detect_needs_think_prefix(self, prompt_tokens: list[int]) -> bool:
        """Detect if prompt ends with an open <think> tag (thinking enabled).

        DFlash2 bypasses the scheduler, so the ``<think>\\n`` prefix the
        scheduler normally prepends to the first chunk for reasoning models
        must be reproduced here. Mirrors ``DFlashEngine._detect_needs_think_prefix``.
        """
        if not prompt_tokens:
            return False

        think_start_id = self._get_think_token_id("think_start_id")
        if think_start_id is None and self._tokenizer_obj is not None:
            try:
                tid = self._tokenizer_obj.convert_tokens_to_ids("<think>")
                if tid == getattr(self._tokenizer_obj, "unk_token_id", None):
                    return False
                think_start_id = tid
            except (AttributeError, KeyError, TypeError):
                return False

        if not think_start_id:
            return False

        last_tokens = list(prompt_tokens[-3:])
        if think_start_id not in last_tokens:
            return False

        last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
        after_start = last_tokens[last_idx + 1 :]

        if after_start:
            think_end_id = self._get_think_token_id("think_end_id")
            if think_end_id is not None and think_end_id in after_start:
                return False
            if self._tokenizer_obj is not None:
                try:
                    tid = self._tokenizer_obj.convert_tokens_to_ids("</think>")
                    unk = getattr(self._tokenizer_obj, "unk_token_id", None)
                    if tid != unk and tid in after_start:
                        return False
                except (AttributeError, KeyError, TypeError):
                    pass

        return True

    def _think_prefix_text(self) -> str:
        tag = getattr(self._tokenizer_obj, "think_start", "<think>")
        return f"{tag}\n"

    def _tokenize_prompt(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, list):
            return list(prompt)
        return list(self._tokenizer_obj.encode(prompt))

    def _run_stream_sync(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        stop_event: threading.Event,
    ) -> None:
        """Drive the ported ``stream_generate`` loop on the MLX executor thread.

        Mirrors ``DFlashEngine._run_generate_streaming``: everything runs
        synchronously on this one thread (required — the ``_GDNStateCapture``
        RLock is acquired/released on this thread only, and the generator is
        always explicitly closed here rather than left to GC).
        """
        gen = None
        try:
            self._record_prefill_guard_active_memory()
            gen = stream_generate(
                self._target_model,
                self._draft_model,
                self._executor_tokenizer,
                prompt_tokens,
                block_size=None,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prefill_step_size=self._prefill_step_size,
            )
            for resp in gen:
                if stop_event.is_set():
                    logger.info("DFlash2 generation aborted by client")
                    break
                metrics = None
                if resp.finish_reason is not None:
                    metrics = {
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.generation_tokens,
                        "finish_reason": resp.finish_reason,
                        "prompt_tps": resp.prompt_tps,
                        "generation_tps": resp.generation_tps,
                        "cached_tokens": 0,
                    }
                asyncio.run_coroutine_threadsafe(
                    queue.put((resp.text, list(resp.tokens), metrics is not None, metrics)),
                    loop,
                )
        except Exception as e:
            logger.error(f"DFlash2 streaming generation error: {e}")
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"error": str(e)})), loop
            )
        finally:
            self._record_prefill_guard_active_memory()
            if gen is not None:
                try:
                    gen.close()
                except Exception as exc:
                    logger.debug(f"DFlash2 generator close() raised: {exc}")
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"aborted": stop_event.is_set()})),
                loop,
            )
            self._active_request = False

    async def generate(
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
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenize_prompt(prompt)
        kwargs.pop("tools", None)

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        activity_id = self._begin_activity("generate", detail="generating")

        def _run():
            gen = None
            try:
                self._record_prefill_guard_active_memory()
                gen = stream_generate(
                    self._target_model,
                    self._draft_model,
                    self._executor_tokenizer,
                    prompt_tokens,
                    block_size=None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    prefill_step_size=self._prefill_step_size,
                )
                tokens: list[int] = []
                text = ""
                finish_reason = "stop"
                prompt_token_count = len(prompt_tokens)
                first_token_at: float | None = None
                for resp in gen:
                    if stop_event.is_set():
                        logger.info("DFlash2 generation aborted by client")
                        break
                    if first_token_at is None and resp.tokens:
                        first_token_at = time.perf_counter()
                    tokens.extend(resp.tokens)
                    text += resp.text
                    prompt_token_count = resp.prompt_tokens
                    self._update_activity(activity_id, token_count=len(tokens))
                    if resp.finish_reason is not None:
                        finish_reason = resp.finish_reason
                return tokens, text, finish_reason, prompt_token_count, first_token_at
            finally:
                self._record_prefill_guard_active_memory()
                if gen is not None:
                    try:
                        gen.close()
                    except Exception as exc:
                        logger.debug(f"DFlash2 generator close() raised: {exc}")
                self._active_request = False

        self._active_request = True
        future = loop.run_in_executor(get_mlx_executor(), _run)
        try:
            try:
                (
                    tokens,
                    text,
                    finish_reason,
                    prompt_token_count,
                    first_token_at,
                ) = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                stop_event.set()
                logger.info("DFlash2 generate cancelled, waiting for executor to drain")
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
                except TimeoutError:
                    logger.warning("DFlash2 executor did not exit within 10s after abort")
                except Exception:
                    pass
                raise
        finally:
            self._end_activity(activity_id)

        if self._detect_needs_think_prefix(prompt_tokens):
            text = self._think_prefix_text() + text

        return GenerationOutput(
            text=text,
            tokens=tokens,
            prompt_tokens=prompt_token_count,
            completion_tokens=len(tokens),
            cached_tokens=0,
            finish_reason=finish_reason,
            tool_calls=None,
            first_token_at=first_token_at,
        )

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
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenize_prompt(prompt)
        kwargs.pop("tools", None)

        prompt_len = len(prompt_tokens)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        # Reasoning models put <think>\n at the end of the prompt; mirrors
        # DFlashEngine.stream_generate's think-prefix handling.
        think_prefix_pending = self._detect_needs_think_prefix(prompt_tokens)

        from ..engine_core import get_mlx_executor

        activity_id = self._begin_activity("generate", detail="generating")
        self._active_request = True
        future = loop.run_in_executor(
            get_mlx_executor(),
            self._run_stream_sync,
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            top_k,
            queue,
            loop,
            stop_event,
        )

        total_text = ""
        total_completion = 0
        finished_normally = False

        try:
            while True:
                new_text, new_tokens, finished, metrics = await queue.get()

                if think_prefix_pending and new_text:
                    new_text = self._think_prefix_text() + new_text
                    think_prefix_pending = False

                total_text += new_text
                total_completion += len(new_tokens)
                self._update_activity(activity_id, token_count=total_completion)

                finish_reason = None
                if finished:
                    if metrics and metrics.get("completion_tokens") is not None:
                        total_completion = int(metrics["completion_tokens"])
                    if metrics and metrics.get("error"):
                        finish_reason = "error"
                    else:
                        finish_reason = (metrics or {}).get("finish_reason", "stop")
                    finished_normally = True

                yield GenerationOutput(
                    text=total_text,
                    new_text=new_text,
                    tokens=new_tokens,
                    prompt_tokens=prompt_len,
                    completion_tokens=total_completion,
                    cached_tokens=0,
                    prompt_tps=(metrics or {}).get("prompt_tps", 0.0) or 0.0,
                    generation_tps=(metrics or {}).get("generation_tps", 0.0) or 0.0,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break
        finally:
            self._end_activity(activity_id)
            if not finished_normally:
                stop_event.set()
                logger.info("DFlash2 stream cancelled, waiting for executor to drain")
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
            except TimeoutError:
                logger.warning(
                    "DFlash2 executor did not exit within 10s after abort; "
                    "next request may still be queued"
                )
            except Exception as exc:
                logger.debug(f"DFlash2 executor future raised: {exc}")

    async def chat(
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
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=ct_kwargs,
            is_partial=is_partial,
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

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
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=ct_kwargs,
            is_partial=is_partial,
        )

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        """True while a request is mid-flight, including the drain window.

        ``_active_request`` covers the gap between ``_end_activity`` (called
        before the executor future is awaited in ``stream_generate``'s
        ``finally``) and the executor thread actually finishing the ported
        ``stream_generate`` generator's own ``finally`` (which can still be
        inside ``_GDNStateCapture.close()``). Without this, TTL eviction
        could tear down the engine mid-drain. Mirrors ``DFlashEngine``.
        """
        if self._active_request:
            return True
        return super().has_active_requests()

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "dflash2",
            "model_name": self._model_name,
            "draft_model": self._draft_model_path,
            "loaded": self._loaded,
            "block_size": (
                int(self._draft_model.config.block_size)
                if self._draft_model is not None
                else None
            ),
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None


class _DFlash2PrefillGuard:
    """Prefill-memory guard target for DFlash2's primary path.

    Identical shape to ``dflash.py``'s ``_DFlashPrefillGuard`` (holds a
    ``MemoryMonitor`` plus the watermark attrs ``ProcessMemoryEnforcer``
    writes each tick) — DFlash2 bypasses the Scheduler entirely, same as
    DFlash1, so it needs the same self-contained preflight target.
    """

    def __init__(self, memory_monitor: MemoryMonitor, prefill_step_size: int):
        self.memory_monitor = memory_monitor
        self._prefill_step_size = prefill_step_size
        self._last_mlx_active_memory_bytes: int = 0
        self._prefill_memory_guard: bool = False
        self._memory_hard_limit_bytes: int = 0
        self._memory_hot_cache_used_bytes: int = 0
        self._memory_static_ceiling_bytes: int = 0
        self._memory_dynamic_ceiling_bytes: int = 0
        self._memory_metal_cap_bytes: int = 0
        self._memory_guard_tier: str = ""

    def record_mlx_active_memory(self, active_bytes: int) -> None:
        self._last_mlx_active_memory_bytes = max(0, int(active_bytes))

    def _current_usage_bytes(self) -> int:
        phys = max(
            0, get_phys_footprint() - max(0, int(self._memory_hot_cache_used_bytes))
        )
        return max(self._last_mlx_active_memory_bytes, phys)

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens: int,
        request_id: str | None = None,
    ) -> None:
        raise_if_prefill_exceeds(
            self.memory_monitor,
            prefill_memory_guard=self._prefill_memory_guard,
            hard_limit_bytes=self._memory_hard_limit_bytes,
            current_usage_bytes=self._current_usage_bytes(),
            prefill_step_size=self._prefill_step_size,
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
            static_ceiling_bytes=self._memory_static_ceiling_bytes,
            dynamic_ceiling_bytes=self._memory_dynamic_ceiling_bytes,
            metal_cap_bytes=self._memory_metal_cap_bytes,
            memory_guard_tier=self._memory_guard_tier,
        )
