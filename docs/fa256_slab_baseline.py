#!/usr/bin/env python3
"""Phase 3.1 / Step 1 — FA256 partial-slab baseline (refutation gate O7).

Measures how the native qwen35_fa256_attention per-op transient scales with
n_chunks (forced via dispatch_budget), to confirm/refute that the
n-chunks-scaled partial slab (o_part/lse_part, capped at max_slab_bytes=2GiB)
is the dominant admission transient — BEFORE the multi-day streaming-fold work.

Run against the app-bundle native _ext (byte-identical csrc to the fork), from
a NEUTRAL cwd so the app package is not shadowed by the fork source:

  APP=/Applications/oMLX.app/Contents/Resources
  PYHOME=$APP/Python/cpython-3.11
  SITE=$APP/Python/framework-mlx-base/lib/python3.11/site-packages
  cd <scratchpad> && PYTHONHOME=$PYHOME PYTHONPATH=$APP:$SITE \
      $PYHOME/bin/python3 fa256_slab_baseline.py
"""
from __future__ import annotations

import gc
import json
import math

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

assert fast.has_symbol("qwen35_fa256_attention"), "native fa256 unavailable"
assert fast.fa256_supports_dispatch_budget(), "extension predates dispatch_budget"

H_Q, H_KV, D = 24, 4, 256          # Qwen3.8-27B GQA attention shape
DT = mx.bfloat16
DT_SIZE = 2
SCALE = 1.0 / math.sqrt(D)
SLAB_CAP = 2 << 30                  # max_slab_bytes in qwen35_prefill.cpp

# one chunk slot: o_part [B,H,qL,D] input-dtype + lse_part [B,H,qL] fp32
def slot_bytes(qL: int) -> int:
    return H_Q * qL * D * DT_SIZE + H_Q * qL * 4

def n_mem_cap(qL: int) -> int:
    return max(1, SLAB_CAP // slot_bytes(qL))

def make_qkv(qL: int, kL: int):
    mx.random.seed(3)
    q = mx.random.normal((1, H_Q, qL, D)).astype(DT)
    k = mx.random.normal((1, H_KV, kL, D)).astype(DT)
    v = mx.random.normal((1, H_KV, kL, D)).astype(DT)
    mx.eval(q, k, v)
    return q, k, v

def measure(q, k, v, budget: int) -> dict:
    """Peak-memory high-water for one fa256 call at the given dispatch_budget.

    clear_cache() empties the buffer pool so command-buffer temporaries
    (add_temporary slabs) must be freshly allocated and thus counted in the
    peak high-water mark rather than served silently from the pool.
    """
    mx.eval(q, k, v)
    mx.clear_cache()
    gc.collect()
    mx.clear_cache()
    base_active = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fast.qwen35_fa256_attention(q, k, v, SCALE, causal=True, dispatch_budget=budget)
    mx.eval(out)
    peak = mx.get_peak_memory()
    transient = peak - base_active
    del out
    return {"peak": peak, "base_active": base_active, "transient": transient}

def budget_for_chunks(qL: int, kL: int, n: int) -> int:
    # mirrors the existing chunked test: work ~ H_Q * qL * kL, budget = work / n
    if n <= 1:
        return 0
    return max(1, (H_Q * qL * kL) // n)

SHAPES = [
    (4096, 8192),
    (4096, 32768),
    (4096, 131072),
    (4096, 262144),
    (2048, 262144),
]

def main():
    results = []
    print(f"{'qL':>6} {'kL':>8} {'n_req':>6} {'budget':>14} "
          f"{'transient_MB':>13} {'slab_vs_single_MB':>18} {'slot_MB':>8} {'cap_chunks':>10}")
    for qL, kL in SHAPES:
        q, k, v = make_qkv(qL, kL)
        cap = n_mem_cap(qL)
        slot_mb = slot_bytes(qL) / 2**20
        single = None
        # n=1 (single dispatch, no slab) is the baseline for the slab delta.
        chunk_targets = [1, 2, 4, 8, 16, 32, 64, cap]
        seen = set()
        for n in chunk_targets:
            if n in seen:
                continue
            seen.add(n)
            budget = budget_for_chunks(qL, kL, n)
            try:
                m = measure(q, k, v, budget)
            except Exception as e:
                print(f"{qL:>6} {kL:>8} {n:>6} {budget:>14}  ERROR: {type(e).__name__}: {e}")
                continue
            if n == 1:
                single = m["transient"]
            slab_delta = (m["transient"] - single) / 2**20 if single is not None else float("nan")
            row = {
                "qL": qL, "kL": kL, "n_req": n, "budget": budget,
                "transient_bytes": m["transient"], "peak_bytes": m["peak"],
                "slab_delta_mb": slab_delta, "slot_mb": slot_mb, "cap_chunks": cap,
            }
            results.append(row)
            print(f"{qL:>6} {kL:>8} {n:>6} {budget:>14} "
                  f"{m['transient']/2**20:>13.1f} {slab_delta:>18.1f} {slot_mb:>8.2f} {cap:>10}")
        # real operating point: auto-calibrated budget
        del q, k, v
        mx.clear_cache(); gc.collect()

    with open("fa256_slab_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote fa256_slab_baseline_results.json")

if __name__ == "__main__":
    main()
