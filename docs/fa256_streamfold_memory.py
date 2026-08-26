#!/usr/bin/env python3
"""Phase 3.1 / Step 3 (partial) — peak-memory A/B: stream_fold ON vs OFF.

Confirms the per-op transient drops from the ~2GiB n_chunks-scaled slab to an
n_chunks-independent ~one-slot figure. Run against the worktree _ext (same env
as the numerics harness).
"""
from __future__ import annotations
import gc, math
import mlx.core as mx
from omlx.custom_kernels.qwen35_prefill import fast

assert fast.fa256_supports_stream_fold(), "rebuilt _ext not loaded"

H_Q, H_KV, D = 24, 4, 256
DT = mx.bfloat16
SCALE = 1.0 / math.sqrt(D)

def qkv(qL, kL):
    mx.random.seed(3)
    q = mx.random.normal((1, H_Q, qL, D)).astype(DT)
    k = mx.random.normal((1, H_KV, kL, D)).astype(DT)
    v = mx.random.normal((1, H_KV, kL, D)).astype(DT)
    mx.eval(q, k, v)
    return q, k, v

def transient(q, k, v, budget, stream_fold):
    mx.eval(q, k, v); mx.clear_cache(); gc.collect(); mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fast.qwen35_fa256_attention(
        q, k, v, SCALE, causal=True, dispatch_budget=budget, stream_fold=stream_fold)
    mx.eval(out)
    t = mx.get_peak_memory() - base
    del out
    return t

# forced budget → ~64 chunks at these shapes (legacy caps ~42 at the 2GiB slab)
SHAPES = [(4096, 262144), (4096, 131072), (2048, 262144)]
print(f"{'qL':>6} {'kL':>8} {'budget':>14} {'legacy_MB':>10} {'stream_MB':>10} {'saved_MB':>9} {'ratio':>6}")
for qL, kL in SHAPES:
    q, k, v = qkv(qL, kL)
    budget = max(1, (H_Q * qL * kL) // 64)
    leg = transient(q, k, v, budget, False)
    strm = transient(q, k, v, budget, True)
    print(f"{qL:>6} {kL:>8} {budget:>14} {leg/2**20:>10.1f} {strm/2**20:>10.1f} "
          f"{(leg-strm)/2**20:>9.1f} {leg/max(strm,1):>6.1f}x")
    del q, k, v; mx.clear_cache(); gc.collect()
