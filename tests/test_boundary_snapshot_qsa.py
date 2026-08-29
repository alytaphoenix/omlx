# SPDX-License-Identifier: Apache-2.0
"""Boundary-snapshot capture must blank block-sliceable cache layers.

Qwen4-Exp's QSAKVCache was missing from ``_KNOWN_SLICEABLE_CACHE_TYPES``,
so every boundary snapshot carried the full K/V+index prefix
(~27.9KB/token x token_count) — the #2551 quadratic-across-snapshots trap
resurfacing for a new cache class. Measured live: ~5GB of request-tied RAM
at 47k tokens and a pre-chunk guard rejection, plus 532MB GDN sidecars
that should be ~113MB. These tests pin the fix and add a registry-lockstep
guard so the NEXT sliceable cache class cannot silently regress the same
way.
"""

from types import SimpleNamespace

import pytest

from omlx.scheduler import (
    _KNOWN_SLICEABLE_CACHE_TYPES,
    _snapshot_value_nbytes,
    Scheduler,
)


def test_qsa_kvcache_is_snapshot_sliceable():
    # Both handlers declare supports_block_slicing=True (verified against
    # the live registry, not source grep — an earlier source scan
    # misattributed the quantized handler's flag).
    assert "QSAKVCache" in _KNOWN_SLICEABLE_CACHE_TYPES
    assert "QSAQuantizedKVCache" in _KNOWN_SLICEABLE_CACHE_TYPES
    # Batch wrapper never appears in per-request prefill capture, and its
    # handler declares slicing=False.
    assert "BatchQSAKVCache" not in _KNOWN_SLICEABLE_CACHE_TYPES


def test_registry_sliceable_handlers_locked_to_snapshot_skip_set():
    """Every registered handler that declares supports_block_slicing=True
    must be in the boundary-snapshot skip set. A True handler missing from
    the set means every boundary snapshot for that model carries the
    layer's full KV prefix — quadratic snapshot cost in context length
    (how the Qwen4-Exp QSAKVCache 5GB-at-47k regression shipped).

    Exemptions, both deliberate:
    - Batch-side classes: prefill capture sees per-request cache objects,
      never batch wrappers.
    - PoolingCache: captured ON PURPOSE but compacted to a single-block
      delta (`_compact_boundary_snapshot_value` /
      `_decode_boundary_snapshot_value`), so its snapshot cost is linear,
      not quadratic — blanking it would break the pooling store path.
      (Its "True" comes from the DefaultCacheHandler fallback, not a
      dedicated handler.)
    """
    # Handlers self-register on module import (_initialize_default_handlers).
    from omlx.cache.type_registry import CacheTypeRegistry

    compacted_by_design = {"PoolingCache"}

    for class_name in CacheTypeRegistry.list_known_class_names():
        if class_name.startswith("Batch") or class_name in compacted_by_design:
            continue
        handler = CacheTypeRegistry.get_handler_by_class_name(class_name)
        if handler.supports_block_slicing:
            assert class_name in _KNOWN_SLICEABLE_CACHE_TYPES, (
                f"{class_name} declares supports_block_slicing=True but is "
                "missing from scheduler._KNOWN_SLICEABLE_CACHE_TYPES — its "
                "full state will be captured into every boundary snapshot "
                "(quadratic in context length)."
            )


class QSAKVCache:  # test double: classification is by class NAME
    pass


class ArraysCache:
    pass


class CacheList:
    def __init__(self):
        self.caches = ()


class SomeUnknownFutureCache:
    pass


def test_emit_prefill_boundary_snapshot_blanks_sliceable_layers():
    captured = {}

    stub = SimpleNamespace(
        _on_prefill_boundary_snapshot=(
            lambda request_id, snapshot_cache, token_count: captured.update(
                request_id=request_id,
                snapshot_cache=snapshot_cache,
                token_count=token_count,
            )
        )
    )
    request = SimpleNamespace(request_id="req-1")
    qsa, arrays, clist, unknown = (
        QSAKVCache(),
        ArraysCache(),
        CacheList(),
        SomeUnknownFutureCache(),
    )
    prompt_cache = [qsa, arrays, clist, unknown]

    Scheduler._emit_prefill_boundary_snapshot(stub, request, prompt_cache, 2048)

    snap = captured["snapshot_cache"]
    assert snap[0] is None, "sliceable QSAKVCache must be blanked"
    assert snap[1] is arrays, "recurrent ArraysCache must be captured"
    assert snap[2] is clist, "composite CacheList keeps capture behavior"
    assert snap[3] is unknown, "unknown classes stay captured (safe default)"
    assert captured["request_id"] == "req-1"
    assert captured["token_count"] == 2048


def test_snapshot_value_nbytes_walks_nested_structures():
    class FakeArray:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    value = [
        {"state": (FakeArray(100), [FakeArray(20)]), "meta_state": "x"},
        None,
        FakeArray(3),
    ]
    assert _snapshot_value_nbytes(value) == 123
    assert _snapshot_value_nbytes(None) == 0
    assert _snapshot_value_nbytes(object()) == 0


def test_oversized_snapshot_warning_threshold_exists():
    from omlx.cache import boundary_snapshot_store as bss

    # ~100-200MB is legitimate recurrent state; the guard must not fire on
    # it, and must exist to fire on full-KV-prefix snapshots (>1GB at 40k).
    assert bss._OVERSIZED_SNAPSHOT_WARN_BYTES >= 200 * 1024 * 1024
    assert bss._OVERSIZED_SNAPSHOT_WARN_BYTES <= 1024 * 1024 * 1024
