# Qwen3.5 (qwen35) hardening and optimization plan

Design doc + phased implementation checklist from the qwen35 kernel / KV-cache /
MTP / ANE review. Every file:line reference below was **re-verified against HEAD
`ce357792` on 2026-08-20**. Line numbers will drift as the tree moves — treat them
as anchors (the quoted identifiers are the stable handles), and re-locate rather
than trust a stale number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here).

---

## 1. Context

A bug-finding and optimization pass covered the qwen35 code path (the
`qwen3_5` model-family architecture, deployed here serving the Qwen3.8-27B
checkpoint): the custom kernels (`omlx/custom_kernels/qwen35_prefill/`), the KV/prefix/SSD cache stack
(`omlx/cache/`), the memory-admission guard (`omlx/scheduler.py`,
`omlx/memory_monitor.py`), and the MTP and ANE patch layers
(`omlx/patches/mlx_lm_mtp/`, `omlx/patches/qwen35_ane_prefill.py`). It was
triggered by two empirical observations on a 262k-context deployment:

- The prefill admission guard rejects at nearly the same `kv_len` for 8-bit and
  4-bit KV (4-bit only bought +16%, `kv_len` 100352 → 116736), implying the
  dominant transient near the ceiling is KV-bit-depth-insensitive.
- Disabling ANE prefill entirely produced **zero** change in the rejection point
  (still failed at `kv_len=100352`), contradicting the theory that ANE surface
  reservations dominate. This null result is now partially explained (§B2) but
  still needs instrumentation before any ANE-accounting fix is assumed to help.

> **Provenance correction (2026-08-24).** The `kv_len 100352 → 116736` /
> "+16%" numbers above were logged in sessions running the **`sdpa256` Python
> fallback**, NOT the native fa256 kernel: those `server.log` sessions (Aug
> 20/21/23) contain **zero** `Qwen FA-256 dispatch budget auto-calibrated`
> lines — that line only fires when native `_ext` loads, and the fork
> deployment ships no built `_ext` (see the deployment note below). So the
> +16% observation measured the **sdpa256 unfused/qsplit score-matrix
> transient** (~9.4 GiB at kv_len=100352 in the same logs — 4.5× the fa256
> slab, and equally KV-bit-insensitive), not the fa256 partial slab. The KV
> A/B itself was valid (same oQ4e weights, kv_bits flipped 8→4), but its
> attribution to §B1's slab was measuring a kernel that wasn't in the process.
> **The native-mode admission baseline** (packaged-app runs Aug 24, native
> fa256 + the shipped 3.2/3.4/3.5 fixes) rejects at **`kv_len` 222464 / 230144**
> with a `~55.52GB + min-chunk transient exceeds 57.00GB` shortfall of
> **~1.5 GB** — the correct order-of-magnitude anchor for §B1's success
> criterion, and consistent (not yet proven) with the ~2 GB capped slab being
> the binding term for the last ~30–40k tokens to 262k. Re-run the KV A/B in
> **native mode** before trusting any "beat the 16%" success number.
>
> **Deployment note (2026-08-24).** `run-fork-server.sh` sets
> `PYTHONPATH="$FORK:$SITE"`; the fork checkout contains no built
> `_ext*.so`/`*.metallib` (only `csrc/`), and the compiled extension lives only
> in the app bundle (`/Applications/oMLX.app/Contents/Resources/omlx/
> custom_kernels/qwen35_prefill/`), which is not on the path. So `from . import
> _ext` fails **silently** (the warn in `fast.py` only fires when a broken
> `_ext.so` is present), `fast.has_symbol("qwen35_fa256_attention")` is False,
> and every fork-run node serves on the pure-MLX `sdpa256`/qmm-fallback path —
> losing ALL native qwen35 kernels (fa256, qmm, ANE, moe_weighted_sum) and, per
> the native vs. fallback rejection points above, roughly **halving the
> effective context ceiling** (100–116k fork vs. 222–230k native). Every
> Python-level fix (Phases 0–2, 3.2/3.4/3.5) still runs under the fork; Phase
> 3.1 is the FIRST native-kernel change, and Step 2's build must produce a fork
> `_ext` (or the native serving path must switch to the app-copy-in deploy) or
> the change ships into a path the live fork nodes never execute. This is a
> live-deployment decision, flagged to the user 2026-08-24.

Two items from the original review are **already resolved** (§9): the MTP draft
depth admin plumbing (fixed, tested, verified live) and the GDN blocked_seq
kernel engine-attachment question (verified fine). Several other findings were
fixed or reshaped by commits that landed after the review (`dcb317fe`,
`8e50f125`/`e55ffd17`, the #2569/#2644 GDN sidecar rework, `267d5436`,
`7fb96d32`/`f4a923ab`); §9 lists them so nobody re-investigates. Everything else
below was confirmed still present at HEAD.

---

## 2. Theme A — Prefix-cache recurrent-state correctness

### A1. GDN/recurrent state desync on store (double-ingestion) — HIGH

**What's wrong.** In `_extract_block_tensor_slice`
(`omlx/cache/prefix_cache.py`), the three non-sliceable branches — rotating
(~2100-2113), CacheList non-sliceable (~2273-2288), ArraysCache/GDN
(~2314-2327) — compute `has_valid_state = is_last_block or (snapshot exists)`
and, when the last full block has no matching boundary snapshot, fall back to
storing the **live end-of-request state** as that block's boundary state.
`store_cache` (~799-830) skips trailing partial blocks, so `is_last_block`
points at the last *full* block boundary while the live recurrent state has
already ingested the trailing tokens beyond it.

**Failure scenario.** A later request hits this prefix, restores the stored
"boundary" state, then re-prefills the trailing tokens — which the state has
already seen. Silently wrong outputs across all 48 GDN layers on every such
prefix-cache hit. No error, no crash; just wrong tokens.

**Mitigation already landed.** The split-GDN sidecar layout (#2569/#2644) is
exempt: it stores a placeholder (~2298-2304) and commits the recurrent
checkpoint separately with rejection-on-failure (`store_cache` ~1200-1249).
Only the embedded (non-split) GDN/rotating/CacheList layouts remain exposed.

**Fix.** In all three embedded branches: store the last block's state **only
when a matching boundary snapshot exists**; otherwise emit a placeholder
exactly as middle blocks do. Losing reuse of the final block is strictly better
than corrupting it. Add a regression test that stores a cache whose trailing
partial block was skipped, restores it, and asserts the restored state matches
a from-scratch prefill of exactly the stored tokens.

### A2. Shared-prefix path still splices blocks without hash validation — HIGH

**What's wrong.** The prefix-index fetch path was fixed since the review (it
now uses hash-validated `acquire_cached_block`, `prefix_cache.py:662-672`,
`paged_cache.py:867-896`, and self-heals the index on mismatch). But the
**shared-prefix** path (`prefix_cache.py:629-635`) still calls membership-only
`increment_ref(block_id)` (`paged_cache.py:852-865`, no hash check) on IDs
returned by `find_shared_prefix` outside the finding lock — the exact TOCTOU
that `acquire_cached_block`'s docstring warns about.

**Failure scenario.** Concurrent eviction + reallocation between
`find_shared_prefix` and `increment_ref` splices a foreign block's KV into the
restored sequence — silent positional corruption.

**Fix.** Use `acquire_cached_block(block_id, expected_hash)` in the
shared-prefix path too, stopping the chain at the first mismatch (mirror the
662-672 pattern).

### A3. `token_count` never validated against stored arrays — MEDIUM-LOW

`prefix_cache.py:1051, 1183/1193`: `block.token_count = len(block_tokens)` is
passed straight to `save_block` without checking it against the actual KV/seq
length of the arrays being stored. A bug anywhere upstream silently persists a
mismatched block. Fix: assert (or reject-and-log) `token_count ==
_get_cache_seq_len(...)`-derived length at store time.

---

## 3. Theme B — Memory-guard accuracy for long-context admission

The guard gate is `current + _admission_transient_bound(chunk, kv_len) <= cap`
(`omlx/scheduler.py:4001`, re-checked at 4006-4007). Near the 262k ceiling the
bound is dominated by terms that don't shrink with KV bit-depth. Four
contributors, in current-code terms:

### B1. FA256 chunked-attention partial slab — the implicated dominant term

`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_prefill.cpp`: `max_slab_bytes =
2LL << 30` (line 325, now a chunk-count cap `n_mem_cap`, 327-330); all
`n_chunks` partial `o_part`/`lse_part` slabs are materialized simultaneously as
command-buffer temporaries (384-391). The kernel accepts only dense fp16/bf16
K/V (171-176), so quantized KV is fully dequantized per chunk — same transient
bytes at 4-bit and 8-bit. This is directly implicated by the empirical
"4-bit KV only helps 16%" result. Note the existing cross-chunk fold already
uses normalized partials + fp32 logsumexp weights, but it is **not** a
streaming accumulator — all partials coexist.

**Fix (large).** Replace the n-chunk partial slab with an online-softmax
accumulator: one fp32 `(B, H, qL, D)` accumulator + running LSE, folding each
KV chunk as it is computed. Removes up to ~2GiB of per-op transient and makes
the transient independent of `n_chunks`. This is the highest-leverage capacity
item in the doc.

#### 3.1 implementation plan — FA256 streaming chunk fold (online-softmax accumulator)

Scoped 2026-08-24 (design only, no code — deliberately deferred as the one [L]
item in Phase 3). Written for an implementer with no memory of the session that
produced it; every claim below was verified against the code that night, with
file:line references current as of branch `deploy/session-fixes-v2`.

##### Context — what exists and what exactly is wrong

`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_prefill.cpp`,
`Qwen35Fa256AttentionPrimitive::eval_gpu` (~205-497):

- **Chunking exists for preemptibility, not memory** (comment at 308-314): one
  dispatch scanning the whole key range grows linearly in wallclock with kL and
  past the macOS IOGPU interactivity threshold the OS demotes/kills the command
  buffer (issue #2225, mlx#3302). Keys are split into `n_chunks` dispatches of
  ≤ `chunk_keys` each (315-337).
- **The fold is already numerically correct** — each chunk kernel emits a
  *normalized* partial (input dtype) plus an fp32 per-row logsumexp
  (`steel_attention_block_token.h:536-553`, values in the kernel's scaled log2
  domain), and `attention_chunk_reduce` (same header, 584-627) combines them
  with logsumexp weights, skipping causally-dead chunks via `lse == -INF`.
  Nothing about the math changes in this item.
- **The problem is residency**: `o_part` `[n_chunks, B, H, qL, bd]` (input
  dtype) and `lse_part` `[n_chunks, B, H, qL]` (fp32) are allocated up front
  and held simultaneously as command-buffer temporaries (384-391), folded once
  at the end. The slab is capped at `max_slab_bytes = 2LL << 30` (325) via a
  chunk-count cap `n_mem_cap` (327-330).
- **The cap creates a second, latent bug**: at huge qL, `n_mem_cap` *overrides*
  the dispatch budget (330), so per-dispatch work can exceed
  `dispatch_budget_` — a quiet re-exposure of exactly the #2225 failure mode
  the chunking was built to prevent, at the extreme sizes where it matters
  most. Streaming removes the memory/preemptibility tension entirely.
- **KV quantization can't help**: the kernel only accepts dense fp16/bf16 K/V
  (`unsupported()`, 171-196), so TurboQuant KV is dequantized before entry and
  the slab is the same size at 4-bit and 8-bit — the implicated cause of the
  empirical "4-bit KV only helps 16%" result (§B1). Success for this item is
  measured against that number, not just peak-memory deltas.

Worked transient example (B=1, H=24, qL=4096, D=256, bf16 — the standard
chunked-prefill shape): one chunk slot = 1·24·4096·256·2 ≈ 50MB, so `n_mem_cap`
admits ~42 chunks ≈ 2.1GiB of coexisting slab. After this change:
fp32 accumulator ≈ 100MB + one bf16 chunk slot ≈ 50MB + two fp32 LSE rows
≈ 1MB → **~150MB, independent of `n_chunks`**. Honest crossover note: at
`n_chunks ≤ 3` the old slab is comparable or smaller (the fp32 accumulator
costs 2× a bf16 slot); accept that rather than adding a path split — the slab
is trivially small there anyway.

> **CORRECTION 2026-08-24 (Fable-reviewed) — the single `lse_run` buffer in
> point 2/3 below is racy; use a double buffer.** With the per-d4 grid copied
> from `attention_chunk_reduce`, the bd/4 threads of a row share `lse_run[row]`
> and span TWO SIMD-groups at bd=256 (the shipping config). The fold reads
> `lse_run[row]` (old) to weight `acc`, then writes it — a within-dispatch,
> cross-thread RMW with no ordering ⇒ a reader can see the post-write value and
> double-fold. `acc` is safe (each `acc[...,4d4..+3]` is single-owner). Fix
> (design X, implemented): keep the per-d4 grid + single in-place `acc`, but
> **ping-pong two `[B,H,qL]` fp32 buffers `lse_run_prev`/`lse_run_next`** (bind
> prev via `set_input_array`, next via `set_output_array`; host swaps each
> chunk), and **gate the lse write to `d4 == 0`**. MANDATORY: the dead-chunk
> (`lse_c == -INF`) branch must still write `lse_run_next[r] = lse_run_prev[r]`
> (acc passes through by not-writing, but the ping-pong lse buffer would go
> stale otherwise); `is_first` + dead writes `lse_run_next = -INF`, `acc = 0`.
> Make dead-chunk pass-through an explicit numerics test (the (4096,4096)
> square shape exercises causally-dead later chunks). Rejected alternatives:
> (Y) single buffer + `threadgroup_barrier(mem_device)` — correct only while the
> "all d4 of a row share one group" invariant + `dispatch_threads` exact-grid
> both hold, neither enforced; (Z) one-thread-per-row — loses acc coalescing;
> keep-full-lse-slab + recompute — reintroduces the n_chunks-scaled allocation
> §B1 exists to remove. The ~1MB double buffer is noise vs the 1.5GB target.

##### Approach

**1. Recommendation: host-loop restructure + a small "fold-one-chunk" kernel;
the attention kernel stays byte-identical.**

Two candidate shapes were considered:

- **(A) Read-modify-write epilogue in the attention kernel**: each chunk
  dispatch loads the running accumulator/LSE, rescales, accumulates, stores.
  Fewer dispatches, fewer memory round trips.
- **(B) Keep the attention kernel exactly as-is** (it already emits normalized
  partial + LSE per chunk); shrink the slab to **one** chunk slot; after each
  chunk dispatch, run a small fold kernel that merges that slot into a
  persistent fp32 accumulator + running LSE.

**Recommendation: (B), and it is not a close call.** The attention kernel body
lives in `omlx/custom_kernels/common/csrc/kernels/steel_attention_block_token.h`,
which is **shared with the glm_moe_dsa extension**
(`glm_moe_dsa/csrc/exact_block_attention.metal` includes it — verified). Option
A means editing the epilogue of a kernel a *second model family* compiles and
ships, gated by new function constants, revalidating both. Option B leaves
every existing compiled function's behavior untouched — the chunk dispatches
produce byte-identical partials to today, so the change reduces to host
orchestration plus one new elementwise kernel that is essentially a 2-term
specialization of the already-trusted `attention_chunk_reduce`. For an item
whose main risk is "native kernel change breaks generation" (see the 3.6
finding), minimizing the kernel-code delta dominates the modest bandwidth
saving of A.

**2. Buffers.** Replace the `[n_chunks, ...]` temporaries (384-391) with, all
via the same `compute_encoder.add_temporary` pattern (temporaries live until
command-buffer completion, which spans all of one primitive's dispatches —
exactly the lifetime needed; no cross-command-buffer persistence is involved):

- `o_slot`: `[B, H, qL, bd]`, input dtype — the single chunk-partial slot,
  written by every chunk dispatch at offset 0 (drop the `c * o_chunk_stride`
  offsets at 467-470).
- `acc`: `[B, H, qL, bd]`, fp32 — running normalized output.
- `lse_slot`: `[B, H, qL]`, fp32 — chunk LSE.
- `lse_run`: `[B, H, qL]`, fp32 — running LSE.

One slot, not two: double-buffering only pays if chunk dispatch c+1 can overlap
fold c, and (see point 4 below) MLX's per-buffer hazard tracking already
serializes these dispatches today, so there is no concurrency to preserve. Note
the two-slot variant as a contingent follow-up only if the implementer's
re-verification of the barrier semantics (O1 below) shows otherwise.

**3. Fold-one kernel spec.** New template kernel, e.g. `attention_chunk_fold`,
placed **additively** next to `attention_chunk_reduce` in
`steel_attention_block_token.h`, instantiated for fp16/bf16 in
`qwen35_attention.metal` alongside the existing reduce instantiations (28-35).
Additive-only: no existing function body in the shared header changes (but
glm_moe_dsa's metallib recompiles from this header — run its tests in
verification).

Grid/threading: copy `attention_chunk_reduce` exactly — one thread per 4
head-dim elements of one (b, h, row); grid `(bd/4, qL, B*H)`, group
`(bd/4, max(1, 256/(bd/4)), 1)` (494-495 in qwen35_prefill.cpp).

Params: reuse the `AttnChunkReduceParams` shape (H, qL, D, O_strides) minus the
chunk strides, plus two flags passed in the params bytes (host values, no
function constants → no pipeline-state churn per chunk):

- `is_first` (c == 0): **never read `acc`/`lse_run`** — they are uninitialized
  garbage and `0 × NaN = NaN` would silently poison everything. Write
  `acc = float(o_slot)`, `lse_run = lse_slot` directly (including
  `lse_slot == -INF` rows: write `acc = 0`, `lse_run = -INF`). This removes any
  need for a separate init/fill dispatch.
- `is_last` (c == n_chunks-1): after merging, cast the merged row to the output
  dtype and store to `O` through `O_strides` (the primitive's output is
  deliberately non-row-contiguous, 283-297 — `attention_chunk_reduce` already
  handles this exact addressing at 621-626; copy it). On non-last chunks, write
  the merged fp32 row back to `acc`/`lse_run` only.

Merge math, all in the kernel's scaled log2 domain (matching 536-553 and
602-618 — differences cancel, use `exp2`/`log2` throughout):

```
if (lse_c == -INF)   -> dead chunk: pass acc/lse_run through unchanged
                        (and on is_last, still cast-store acc to O)
else:
  m       = max(lse_run, lse_c)
  a       = exp2(lse_run - m)      // 0 when lse_run == -INF
  b       = exp2(lse_c   - m)
  s       = a + b
  acc'    = (a * acc + b * o_slot) / s
  lse_run'= m + log2(s)
```

This is the standard online-softmax rescale and is algebraically identical to
the batched fold at 602-618 — same weights, different association order. The
only numerical difference vs. today is fp32 associativity across n_chunks-1
merges instead of one weighted sum, bounded by a few fp32 ulps and invisible at
bf16/fp16 output granularity (verification quantifies this rather than assumes
it).

**4. Host-loop restructure and the preemptibility question.** In `eval_gpu`:

- `n_chunks <= 1` path (339-377): untouched.
- Chunked path: inside the existing `for (c...)` loop, after each chunk's
  `dispatch_threadgroups` (471), set up and dispatch the fold kernel against
  `o_slot`/`lse_slot`/`acc`/`lse_run` (+ `O` on last). Delete the trailing
  batched reduce (474-496) on this path. Pipeline-state switching per chunk
  (attention kernel ↔ fold kernel) is ordinary encoder usage; the fold's
  pipeline is fetched once outside the loop.
- **Streaming path drops `n_mem_cap`** (327-330): with an n_chunks-independent
  transient, `dispatch_budget_` is honored exactly at every qL — fixing the
  latent budget violation described in Context. Keep `min_chunk_keys = 4*bq`
  (319-321) and the bk alignment (332).

**Does serializing dispatches reintroduce #2225?** No — and this is the most
important paragraph in the plan, so the reasoning is spelled out:

1. The #2225 failure is a function of **per-dispatch wallclock**, not
   command-buffer total time: the fix that shipped was "each chunk its own
   preemptible dispatch" within the *same* command buffer, and it worked. The
   streaming design keeps every dispatch individually bounded — attention
   dispatches by `chunk_keys` exactly as today, fold dispatches a tiny
   bandwidth-bound elementwise pass over `B*H*qL*bd` (~sub-millisecond at the
   150MB working set above). Serial dependency between bounded dispatches does
   not recreate an unboundedly-long dispatch.
2. There is in fact **no inter-chunk concurrency to lose**. Read against MLX
   v0.32.0 (the pinned version) `mlx/backend/metal/device.cpp`:
   `CommandEncoder::set_output_array` calls `set_input_array`, which sets
   `needs_barrier_` whenever the buffer is in `prev_outputs_`, and
   `dispatch_threadgroups` calls `maybeInsertBarrier()` first — i.e., hazard
   tracking is **per-buffer, not per-range**. Today's chunk dispatches all bind
   the same `o_part`/`lse_part` buffers (and all bind `o` at buffer 3, line
   465), so MLX already inserts a memory barrier between every chunk dispatch.
   The current chunked path is already fully serial; streaming adds cheap fold
   dispatches to an already-serial pipeline. Separately — and this one is
   **correctness-load-bearing for the single-slot design**, not just perf —
   the write-after-read hazard (fold c *reads* `o_slot`, chunk c+1 *writes*
   it) is covered by the other check in the same source:
   `register_output_array` does
   `needs_barrier_ |= (prev_inputs_.find(buf) != prev_inputs_.end())`, so a
   dispatch writing a buffer a prior dispatch read also gets a barrier.
   Without that WAR barrier, chunk c+1 would overwrite the slot while fold c
   reads it — silent corruption. **Not independently re-verified in this
   session** (the between-dispatch `next_outputs_`/`prev_inputs_` → `prev_*`
   bookkeeping was not read verbatim) — see O1, this must be confirmed before
   implementing the single-slot loop.

**5. Rollout plumbing — mirror the `dispatch_budget` precedent exactly.** That
precedent already solved this item's exact deployment problem (new kwarg vs.
previously-built extensions): see `test_dispatch_budget_zeroed_on_old_extension`
and `fast.fa256_supports_dispatch_budget()`.

- Add `stream_fold: bool` to the binding/`fast.qwen35_fa256_attention` →
  primitive constructor → `is_equivalent`/`state` (501-509 — forgetting these
  breaks primitive caching/equivalence silently).
- Add `fast.fa256_supports_stream_fold()` capability probe; the patch layer
  (`omlx/patches/qwen35_fa256_attention.py`) passes the kwarg only when
  supported, gated by env `OMLX_FA256_STREAM_FOLD` following the existing
  `OMLX_FA256_*` pattern.
- **Default OFF at first build** (legacy slab path retained verbatim, including
  `n_mem_cap`); flip default ON only after the Step-3 benchmark below; delete
  the legacy path + `n_mem_cap` + `attention_chunk_reduce` usage in a separate
  cleanup PR one release later.
- The kwarg also gives in-process A/B for tests — required by Verification.

**6. Memory-guard interplay (where the capacity win actually lands).** The
admission bound (`scheduler.py:3785`, `_admission_transient_bound`) is
measurement-driven: `max(predicted, tracker.observed_max_bytes)`, and 3.4's
observed-max decay/reset is already shipped — so once real transients shrink,
the guard follows automatically. Two things to check, not assume:

- `memory_monitor.estimate_chunk_transient_bytes` (the static term feeding
  `_predicted_chunk_transient`, scheduler.py:3771-3777) must not hardcode the
  2GiB slab or an n_chunks-scaled FA256 term; if it does, update it to the new
  `~(6 bytes + ε) * B*H*qL*D` formula. Open question O2 below.
- After flipping the default, the first prefills still admit against the stale
  (pre-fix) observed max until decay catches up — expected, safe (conservative
  direction), worth a log line, not worth code.

##### Phased breakdown (independently shippable steps)

This decomposes cleanly; each step lands alone.

1. **[S] Baseline measurement, no native change.** Offline harness (scratch
   script or a Metal-gated benchmark-marked test): `mx.get_peak_memory()` /
   `mx.get_active_memory()` deltas around forced-budget
   `fast.qwen35_fa256_attention` calls across a shape ladder up to
   262k-context scale and an n_chunks sweep up to the cap. Record numbers into
   §B1. This also directly confirms (or refutes) "the slab is the dominant
   admission term" *before* the multi-day kernel work is spent — if the
   measured slab share is small, stop and re-scope.

   **DONE 2026-08-24 — gate PASSES, proceed to Step 2.** Harness:
   `docs/fa256_slab_baseline.py` (+ raw `docs/fa256_slab_baseline_results.json`),
   run against the **app-bundle** native
   `_ext` (fork csrc is byte-identical to the app's shipped csrc — verified
   `diff -q` on `qwen35_prefill.cpp` and `steel_attention_block_token.h`; the
   fork checkout ships no built `_ext`, see the deployment note in §1). Sweep
   forced `dispatch_budget` over n_chunks ∈ {1,2,4,8,16,32,64,cap} at qL ∈
   {2048,4096}, kL ∈ {8192…262144}, GQA H=24/4, D=256, bf16. Results:
   - Per-op transient scales **linearly with n_chunks and pins at the 2GiB
     cap**: at qL=4096, single-dispatch (n=1, no slab) transient = **48 MB**;
     it climbs 48→145→242→435→822→1596→**2031 MB** and saturates at
     n_mem_cap=42 slots × 48.4 MB/slot. Slab delta over single-dispatch =
     **~1.98–2.06 GB**.
   - **kL-independent**: identical 2031 MB at kL=8192 and kL=262144 (the slab
     depends only on qL and n_chunks, not kL) — this is the mechanism behind
     the KV-bit-depth insensitivity.
   - At the real auto-calibrated budget near the ceiling, n_chunks is forced
     far past the cap, so the slab sits pinned at the ~2GiB cap for any chunk
     size (qL=2048 caps at ~2056 MB / 84 slots; qL=4096 at ~2031 MB / 42).
   - The single-dispatch attention transient is tiny (48 MB @ qL=4096, 24 MB @
     qL=2048), so **~1.9 GB of the ~2.0 GB is pure partial slab** — the
     streaming fold's predicted ~150 MB would recover ~1.9 GB.
   Cross-checked against the native-mode rejection evidence (222464/230144
   rejected with a ~1.5 GB shortfall, see §1 provenance): a ~2 GB slab exceeds
   that shortfall, so eliminating it clears those admissions. O7 satisfied.
   O2 also resolved while here: the static estimator
   (`memory_monitor.estimate_prefill_transient_bytes`) is built from SDPA
   score-matrix terms only — **no hardcoded slab / n_chunks / fa256 term** — so
   the slab reaches the guard solely via the measurement-driven tracker
   (`observed_max_bytes`); with 3.4's decay shipped, no static-estimator change
   is needed (the "tracker path suffices" branch of §6).
2. **[M/L] Implement behind `stream_fold` (default OFF) + offline numerics.**
   All of Approach points 2-5. Build in an isolated worktree per the
   established native-change pattern (below). Numerical verification matrix
   (below) must fully pass before any benchmark numbers are trusted.

   **DONE 2026-08-24 — implemented, built, numerics + memory validated.**
   Worktree `/Users/alytaphoenix/repos/omlx-fa256` (branch
   `feat/fa256-stream-fold`, uncommitted). Files: `AttnChunkFoldParams`
   (`.../steel/attn/params.h`); `attention_chunk_fold` kernel (additive in
   `steel_attention_block_token.h`) + fp16/bf16 instantiations
   (`qwen35_attention.metal`); host-loop streaming path + `stream_fold_`
   member/ctor/`is_equivalent`/`state` + `n_mem_cap` lifted when streaming
   (`qwen35_prefill.cpp`); builder + `.h` decl; `stream_fold` binding +
   `FA256_HAS_STREAM_FOLD` probe (`bindings.cpp`); `fa256_supports_stream_fold`
   + wrapper pass-through (`fast.py`); `OMLX_FA256_STREAM_FOLD` env gate
   (`patches/qwen35_fa256_attention.py`). Design **X** per the correction above:
   single in-place fp32 `acc`, ping-pong `lse_run_a/b`, lse write gated to
   `d4==0`, dead-chunk writes `next=prev`. Clean build (all 4 exts, exit 0),
   ABI-loads under the app-bundle mlx runtime.
   - **Numerics matrix (`scratchpad/fa256_streamfold_numerics.py`) ALL PASS**:
     stream vs legacy fold ≤ **4.88e-4** (bf16) / **1.22e-4** (fp16) — near
     bit-identical; stream vs single-dispatch ≤ 1.95e-3; stream-vs-fp32-SDPA
     == legacy-vs-fp32-SDPA to printed precision (streaming adds ZERO error);
     no NaN including the (4096,4096) causally-dead-chunk shape. Covers
     n_chunks ∈ {2,8,48}, aligned/unaligned/square, fp16+bf16.
   - **Peak-memory A/B (`scratchpad/fa256_streamfold_memory.py`)**: at the
     forced operating budget, transient drops **2079.8 MB → 193.1 MB (10.8×)**
     at qL=4096 and 1572 → 96.6 MB (16.3×) at qL=2048 — **n_chunks-independent**
     (same 193.1 MB at kL=131072 and 262144), matching the ~150 MB prediction.
     The ~1.9 GB saved **exceeds** the ~1.5 GB native rejection shortfall, so it
     should clear the 222464/230144 admissions.
   - **Tests**: `tests/test_qwen35_fa256_attention.py` extended (stream-vs-legacy
     matrix, n_chunks==1 no-op, capability-probe on/off) — **27 passed**;
     glm_moe_dsa co-owner suite (shared-header recompile) — **51 passed**.
   Remaining before merge: Step 3 (tokens/s overhead + flip default), Step 4
   (ship + live 262k). Also pending: deploy the freshly-built fork `_ext` (the
   native-kernel regression fix) to the running nodes — user decision.
3. **[M] Benchmark and flip the default.** Rerun the Step-1 ladder A/B
   (`stream_fold` on/off): peak memory, prefill tokens/s (fold-dispatch
   overhead check at large n_chunks), and — the doc-level success criterion —
   **re-run the 4-bit-vs-8-bit KV comparison that produced the "4-bit only
   helps 16%" measurement** and record the new delta. Flip default ON.

   **DONE 2026-08-24 (benchmark + flip); KV A/B still pending.** Fold-overhead
   benchmark (`docs/fa256_streamfold_bench.py`) at MATCHED n_chunks (below the
   legacy cap, isolating the extra fold dispatches): **1.9%–7.7%** fa256 call
   overhead (~4% typical; 7.7% only at 32 chunks on the cheap 2048×16384
   shape). At the real long-context operating point legacy is *capped* at the
   2GiB slab and violates the ~10ms/dispatch budget (latent #2225), while
   streaming honors it — so the ~4% is bounded and offset by the 10.8×
   transient drop. **Default flipped ON** in the patch (commit 356e0ad0);
   `OMLX_FA256_STREAM_FOLD=0` forces legacy. **Still TODO**: re-run the native
   4-bit-vs-8-bit KV capacity A/B to record the new delta vs the (fallback-mode,
   now-corrected) "+16%" — needs the live long-context harness.

   **DONE 2026-08-25 — §B1 hypothesis CONFIRMED: 4-bit's benefit jumps +16% →
   +98%.** Ground-truth KV bytes/token measured from `TurboQuantKVCache` (16 KV
   layers, 4 heads, 256 dim; overhead ~256 B/tok so ~linear in bits): 4-bit
   16.25 KB, 6-bit 24.25 KB, 8-bit 32.25 KB → 8:4 ratio **1.985×**. Transient
   flatness verified empirically on the peer's native+streaming logs: implied
   transient = 12.51 GB @ kv_len 4881 and 12.70 GB @ kv_len 50726 (Δcurrent
   1.25 GB ≈ ΔKV_storage 1.06 GB) — i.e. `current = model + KV(kv_len) + flat
   transient`, KV is now the sole kv_len-dependent admission term. Capacity
   **empirically anchored 2026-08-25**: the peer's guard rejected an 8-bit
   prefill at **kv_len 175040** ("39.85 GB + min-chunk transient exceeds the
   41.8 GB safety cap = 95% of the 44 GB ceiling") → shared KV budget **5.38 GB**
   (the earlier 10.58 GB estimate was optimistic: it assumed the raw 44 GB and a
   12.6 GB transient, but the guard enforces a 41.8 GB safety cap and the real
   transient is ~13.6 GB + a next-min-chunk margin). At that budget:
   **8-bit ~175k, 6-bit ~233k, 4-bit ~347k** → 4-bit vs 8-bit still **+98%
   (1.985×)**, vs the historical **+16%** (100352→116736). Practical consequence
   on this 44 GB peer: **only 4-bit KV reaches the full 262k model max**; 8-bit
   caps ~175k, 6-bit ~233k (raise `iogpu.wired_limit_mb` to lift all three). The historical figure was small
   because the fallback `sdpa256` transient was O(kv_len) (~9.4 GB at 100k) and
   masked KV storage; streaming makes the transient flat so the full ~2× KV
   ratio shows. **Caveat (this deployment):** both rejection points exceed the
   262k model max, so for THIS hybrid cheap-KV model (16/64 layers) KV bit-depth
   no longer gates capacity in the servable range — it's a quality/decode-speed
   choice here; the +98% would bind on a dense all-attention model. Harness:
   the `TurboQuantKVCache` byte measurement + the two-point transient-flatness
   check above (both reproducible offline; no rejection sweep is possible on
   this model since it doesn't hit the memory wall below 262k in native mode).
4. **[S] Ship + live validation.** PR; packaged-app ABI-load check; one real
   262k-ceiling request end-to-end, patiently to completion.

   **DEPLOYED 2026-08-24 (native regression fix + streaming), live 262k gate
   pending.** Merged `feat/fa256-stream-fold` (ff) into `deploy/session-fixes-v2`
   (356e0ad0), pushed to fork. Rebuilt `_ext` in the main checkout (native
   kernels + streaming), ABI-verified under the app-bundle mlx runtime. Peer is
   same-arch (M4 Pro) but can't build (no `.venv`/cmake) → rsync'd the 13 built
   artifacts (all 4 exts' `_ext.so`/metallibs/dylibs) to the peer,
   ABI-verified there too (stream≈legacy 2.44e-4, no NaN). **Both nodes
   restarted** onto native+streaming (coordinator pid 49645, peer new pid),
   healthy, no tracebacks — this also closes the "fork nodes lost native
   kernels / halved ceiling" regression. Kernels load lazily, so the FA-256
   patch + `stream_fold=ON` engage on first model load. **Still TODO**: the one
   real 262k-ceiling request end-to-end (loads a 27B + auth) as the final gate.
5. **[S, next release] Cleanup.** Remove legacy slab path, `n_mem_cap`, and the
   batched reduce dispatch path; keep `attention_chunk_reduce` compiled only if
   glm_moe_dsa grows a dependency on it in the interim (it has none today).

##### Explicitly not doing

- **Option A (RMW epilogue in the shared attention kernel)** — rejected above;
  do not revisit unless the fold-dispatch overhead measured in Step 3 is
  somehow material (it should be noise).
- **Two-slot double buffering of the chunk partial** — contingent complexity;
  only if re-verification of MLX barrier semantics shows real inter-dispatch
  concurrency exists to recover (it does not, per v0.32.0, pending O1).
- **Quantized-KV ingestion in this kernel** (accepting TurboQuant blocks
  directly, dequantizing per-tile in-kernel) — a real future capacity item, but
  a separate, larger one; this item makes the kernel's own transient
  KV-bit-depth-*irrelevant* rather than KV-bit-depth-*aware*.
- **Retuning `chunk_keys` / dispatch-budget calibration** — the auto-budget
  machinery is untouched; streaming only makes the budget honored exactly
  where `n_mem_cap` used to override it.
- **Any change to the `n_chunks == 1` fast path** (339-377) — byte-identical.
- **Touching `steel_attention_block_token.h` existing function bodies** — the
  fold kernel is additive only; glm_moe_dsa co-owns this header.

##### Verification

1. **Numerical matrix (before any perf work).** Extend
   `tests/test_qwen35_fa256_attention.py`, following
   `test_native_fa256_chunked_matches_single_dispatch`'s exact pattern
   (Metal-gated, `has_symbol` + capability skips, forced `dispatch_budget` to
   pin n_chunks, fp32-cast max-abs diff + NaN assert):
   - Streaming vs. **legacy fold** (in-process A/B via the kwarg), identical
     inputs/budget: tolerance **1e-3** (same partials, same weights, fp32
     association order is the only difference — this should be near
     bit-identical; a looser result is a bug signal, not a tolerance problem).
   - Streaming vs. **single-dispatch** (`dispatch_budget=0`) and vs. **MLX
     reference SDPA** (`mx.fast.scaled_dot_product_attention` on fp32-cast
     inputs, as the existing small-shape reference test does): existing **5e-3**
     anchor.
   - Shapes: the three existing parametrized shapes — (2048, 8192) chunked,
     (4096, 4096) square/causally-dead-chunk rows, (2048, 8001) unaligned last
     chunk — plus a **large-n_chunks** case (e.g. qL=1024, kL=32768, budget
     forcing ≥ 32-64 chunks; slot is ~12MB at this qL so it's cheap) and
     explicit n_chunks ∈ {1 (must take the untouched fast path), 2, large}.
   - Both dtypes (fp16 and bf16).
   - `stream_fold=False` regression case + capability-probe test mirroring
     `test_dispatch_budget_zeroed_on_old_extension`.
2. **Shared-header co-owner**: run the glm_moe_dsa kernel test suite — its
   metallib recompiles from the modified (additively) common header.
3. **Build/deploy sequence — the established native-change pattern from the
   3.6/ANE session**: (a) isolated worktree; (b) build the *unmodified* source
   first and confirm a clean baseline compile+link before attributing any
   failure to the change; (c) verify ABI-load of the rebuilt extension in the
   **packaged app's actual runtime** (the app lags the repo by 50+ files —
   smoke-test imports; never assume repo-venv success transfers); (d) all
   numerics and benchmarks run **offline against the extension directly** —
   do not restart a live server to test anything benchmarkable offline.
4. **Peak-memory A/B** (Step 3): measured transient at the ladder shapes drops
   from n_chunks-scaled (≈2GiB at cap) to the flat ~6 bytes/elem figure;
   confirm no tokens/s regression from fold dispatches at large n_chunks.
5. **The §B1 headline number**: re-run the 4-bit-vs-8-bit KV capacity/pressure
   comparison; the expectation under the §B1 hypothesis is that 4-bit KV's
   benefit materially exceeds the historical 16% once this transient stops
   masking it. Either outcome is signal; record it in §B1.
6. **Guard follow-through**: confirm `estimate_chunk_transient_bytes` reflects
   the new transient (O2), and observe one real long-context admission
   accepting a request the pre-fix bound would have rejected.
7. **Full-ceiling run**: one 262144-token request end-to-end on the live
   deployment, to real completion, as the final gate.

##### Risks and open questions (resolve before or during Step 2, don't guess)

- **O1 — RESOLVED 2026-08-24, single-slot is safe (verified against pinned
  v0.32.0 `device.cpp` via GitHub raw + local `device.h`).** `register_output_array`
  does `needs_barrier_ |= (prev_inputs_.find(buf) != prev_inputs_.end())`;
  `maybeInsertBarrier` runs at the START of every `dispatch_threadgroups`/
  `dispatch_threads` and each dispatch moves `next_inputs_/next_outputs_` →
  `prev_*` (via `std::move` when a barrier fires, else `insert`). Trace: chunk
  c+1's `set_output_array(o_slot)` → `register_output_array` finds `o_slot` in
  `prev_inputs_` (put there by fold c's `set_input_array(o_slot)` read the prior
  dispatch) → `needs_barrier_=true` → barrier before c+1 runs. WAR covered; RAW
  (chunk→fold on `o_slot` via `prev_outputs_`) and the `acc` WAW covered
  symmetrically. The primitive uses no `ConcurrentContext` (concurrent_=false),
  so nothing suppresses the barrier. **Proceed with the single slot; the
  two-slot fallback is unnecessary.** Original precondition (kept for context):
  v0.32.0 `device.cpp` shows both checks (`set_input_array` →
  `needs_barrier_` on `prev_outputs_` for RAW/WAW; `register_output_array` →
  `needs_barrier_` on `prev_inputs_` for WAR; `maybeInsertBarrier()` before
  each dispatch), but the between-dispatch move of `next_*` into `prev_*`
  sets was not read verbatim in this session. **Re-confirm both in the pinned
  source before implementing the single-slot loop**; re-check whenever the
  MLX pin moves. If the WAR barrier turned out not to exist, fall back to two
  alternating slot buffers (distinct MTL buffers) — which then also buys back
  overlap.
- **O2 — static estimator**: does `memory_monitor.estimate_chunk_transient_bytes`
  encode the slab? If yes it must shrink with this change or the guard won't
  release the capacity; if no, the tracker path (with 3.4's decay) suffices.
- **O3 — fold-dispatch overhead**: ~n_chunks extra tiny dispatches per op, per
  layer. Expected noise vs. attention compute; Step 3 measures it. If it ever
  matters, fold every k chunks into k slots (hybrid) before considering
  Option A.
- **O4 — `is_equivalent`/`state`**: the new constructor flag must join both
  (501-509) or primitive caching will conflate stream/legacy instances.
- **O5 — accumulator dtype**: fp32 chosen deliberately (matches the existing
  reduce's accumulation dtype); do not be tempted into a bf16 accumulator to
  halve the 100MB — that changes numerics this plan promises are unchanged.
- **O6 — old-extension tolerance**: the patch layer must degrade gracefully
  when the loaded extension predates the kwarg (probe pattern exists; copy it,
  test it).
- **O7 — Step-1 refutation path**: if baseline measurement shows the slab is
  *not* a dominant admission term at real shapes, stop after Step 1 and
  re-scope §B1 rather than spending the [L] effort on principle.

### B2. ANE transient reservation — reshaped since review; instrument BEFORE fixing

The review-era model ("fixed surfaces double-charged after first dirty") no
longer matches the code. Commit `dcb317fe` replaced dirty-tracking with a
deliberate reservation: `memory_monitor.py:805` adds
`self._ane_prefill_transient_bytes` to **every** request's peak estimate, and
the value is computed at `set_model_info` time by
`ane_prefill_transient_bytes(model)`
(`omlx/patches/qwen35_ane_prefill.py:3321-3354`), which walks the **actually
attached** compiled slice states (`_omlx_ane_prefill_state`,
`_omlx_ane_gdn_state`, `_omlx_ane_fused_down_state`, `down_ane`) and sums
`(input_dim + output_dim) * sequence_length * 2` from live native dims. It does
**not** read `qwen35_ane_prefill_enabled`.

Consequences:

- The term is zero only if no slice state is attached at `set_model_info` time.
  Disabling ANE at load ⇒ term already 0 ⇒ the empirical "ANE-off changed
  nothing" result is *consistent* with this code — but it is equally consistent
  with the ANE term simply not being dominant. **These must be distinguished
  before investing in any ANE-accounting fix.**
- A runtime disable, or a failure latch (`_omlx_ane_prefill_failed = True`,
  see C2), does **not** detach state, so the reservation persists for
  procedures that will never run again.
- Once the fixed surfaces are resident (inside measured `current`), adding the
  full reservation to every request double-counts them.

**Step 0 (do first, small).** Instrument the admission decision: log
`current`, predicted chunk transient, `tracker.observed_max_bytes`, and
`_ane_prefill_transient_bytes` at every rejection. Re-run the 262k rejection
with ANE on/off-at-load/off-at-runtime and record which term moves. Only then
decide whether B2 fixes (charge-once semantics; detach state + refresh
`set_model_info` on disable/latch) are worth doing.

### B3. TurboQuant's bounded transient invisible to the guard — RESOLVED (2026-08-24)

`omlx/patches/turboquant_attention.py` tiles long prefill at 256×16384
(`_LONG_PREFILL_QUERY_BLOCK_SIZE`/`_LONG_PREFILL_KEY_CHUNK_SIZE`, lines 26-27,
applied 635-638) but never registers this bound with the memory monitor — the
`register_tiled_prefill_head_dim` pattern used by
`sdpa256_attention.py` has no turboquant caller. The guard therefore prices
turboquant prefill with a generic (over-)estimate. Fix (small): mirror the
sdpa256 registration.

**Design consult finding, before implementing "just mirror sdpa256"**:
`register_tiled_prefill_head_dim`'s registry (`_SDPA_TILED_PREFILL_HEAD_DIMS`,
`memory_monitor.py`) is a bare module-level global, safe for sdpa256 only
because that kernel is installed by monkeypatching `scaled_dot_product_attention`
itself — process-wide, true for every model sharing that head_dim. TurboQuant's
tiled route is gated on a per-scheduler config (`_turboquant_kv_bits`), and
`engine_pool.py` keeps two models resident during a swap; a naive global
registration would leak one model's turboquant config into a concurrently
-resident non-turboquant model sharing the same head_dim, under-charging that
other model's admission guard — an actual OOM risk, not just a missed
optimization (the opposite failure direction from the bug this item
describes, which is safely over-conservative).

**Fix shipped**: a per-`MemoryMonitor`-instance override
(`register_tiled_prefill_tile`/`clear_tiled_prefill_tile`,
`_tiled_prefill_override` field), set from `Scheduler._set_model_info_for_monitor`
gated on the same eligibility check already used for the KV dtype-width
override (`_turboquant_kv_bits is not None` and MLA/attention-sink/cache
-layer-type eligibility). `_estimate_sdpa_activation_bytes` checks this
instance override before the module-global head_dim registry, and caps the
charged query length at `query_block` (256) since TurboQuant's kernel tiles
the query axis too — charging the raw `query_tokens` would silently
re-introduce an O(query_tokens) over-charge on large chunks.

**Deliberately deferred, not fixed in this diff** (flagged by the design
consult, tracked here rather than silently dropped):
- `_ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE` (`memory_monitor.py:85-96`, used by
  the inkling banded-mask bias term) has the **identical** latent
  module-global-during-swap bug — during a model-swap window it can under- or
  over-charge whichever model didn't set it last. Not touched here (blast
  radius vs. payoff for an unrelated model type); worth its own small fix
  mirroring the per-instance pattern above if/when inkling + swap-window
  overlap is a live concern.
- The dequantize+SDPA exception fallback in `turboquant_attention.py`
  (when `quantized_attention` itself raises) remains unpriced by the tile
  registration — rare failure path, accepted as-is.
- Unrelated, found while testing: `tests/test_sdpa256_attention.py::test_should_route_gate`
  is flaky when run alongside other memory-monitor/turboquant test files in
  the same session (passes in isolation; fails intermittently in combined
  `-k` runs with a tiled/unfused routing-decision mismatch) — looks like
  cross-file leakage of a registered headroom provider that isn't cleaned up
  by whichever test sets it. Not caused by this fix (reproduces identically
  pre-fix); not investigated further here.

### B4. Session-wide non-decaying observed-max ratchet — RESOLVED (2026-08-24)

`scheduler.py:3775-3799`: `bound = max(prediction,
tracker.observed_max_bytes)`. Since the review, two mitigations landed: only
`floor_sample` readings feed the max, and samples above
`_OBSERVED_MAX_CLAMP_BYTES = 4GiB` are rejected
(`prefill_transient_tracker.py:114-125`). But `observed_max_bytes` still only
ever increases and `reset()` is never called on the scheduler's tracker — one
pathological floor-chunk sample below 4GiB still permanently raises the
admission floor for the session. Fix (small): add decay (e.g. exponential
toward the prediction) or a windowed max, and/or reset on model swap.

**"Reset on model swap" was already true by construction** — verified
directly, not assumed: `engine_pool.py` always sets `entry.engine = None`
on eviction and constructs a brand-new `BatchedEngine`/`VLMBatchedEngine`/
`DistributedBatchedEngine` on the next load (`_load_engine`,
`engine_pool.py:2438-2600`), each of which builds a fresh `Scheduler` ->
fresh `PrefillTransientTracker` with `observed_max_bytes` starting at 0.
This is true both for swapping to a different model and for reloading the
*same* model after eviction. No code change needed for that half of the
fix. The real gap is a single continuously-loaded model (this is a
personal server that plausibly runs one model for hours/days without a
swap) where one early floor-chunk spike pinned the admission floor for
the rest of that uptime even if it never recurred.

**Fix shipped**: exponential decay (`_OBSERVED_MAX_DECAY = 0.98`) in
`PrefillTransientTracker.update`'s `floor_sample` branch — a new floor
reading above the current max still raises it instantly (a fresh spike is
real evidence the moment it's seen), but a reading below the current max
now pulls it down by 2% of the gap per sample instead of leaving it
frozen. Deliberately slow: this bound is a safety floor, not a rolling
average, so a single lucky low reading right after a real spike must not
immediately erase the protection that spike earned; recovery is gradual
over many floor samples. Kept the existing scalar `_observed_max_bytes`
representation (rejected a windowed-deque alternative) since four
existing tests white-box-poke that attribute directly
(`test_prefill_oom_graceful.py`, `test_scheduler_prefill_memory_guard.py`)
to seed guard-abort test scenarios — decay-in-place preserves that
contract with no test rewrites needed for those files.

---

## 4. Theme C — ANE lifecycle robustness

### C1. Unprotected `begin()`/`end()` on single and dual dispatch paths — HIGH (conditional)

`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`: on the single path,
`producer_buffer->retain()` + `model_->begin(...)` (2009-2011) sit outside any
try; the enclosing try starts at 2098. An exception between `begin()` and the
detached-thread spawn (2085) — e.g. `get_library`/`get_kernel` at 2029-2045 —
leaks the retained command buffer and orphans the `submitted_` counter; the
next `begin()` then throws "overlapping evaluations" (856-859), wedging the
model permanently. The dual path is identical (2317-2320). The new fused path
(from `d2110d1c`) **is** protected (retain 2644, begin 2648 inside try 2647) —
use it as the template. Fix (medium): RAII/try-catch the retain+begin+spawn
window on single and dual paths; on failure, release the buffer and call the
matching `end()`/state rollback.

### C2. Failure latches never release ANE state/weights — MEDIUM

`omlx/patches/qwen35_ane_prefill.py:2610-2650`: warmup failure only sets
`module._omlx_ane_prefill_failed = True` (and the GDN twin); the compiled state
attached at 2581 — native models, IOSurfaces, dequantized fp16/fp32 weights —
stays resident forever, and (per B2) the memory guard keeps reserving for it.
The recent leak fixes (`7fb96d32`, `f4a923ab`) addressed the compile-retry
ladder, not this. Fix: on latch, detach the state attributes, drop the dense
weight copies, and re-invoke `set_model_info` so the reservation re-prices.

### C3. Permanent weight duplication — MEDIUM (memory floor)

Same file: CPU rows eagerly dequantized to dense fp16 held in state (437-445,
516-524; `dense_rows` goes through fp32 at 509); GPU-suffix rows of routed
gate/up/qkv duplicated via `mx.contiguous`/`concatenate` (646-652, 772-778,
850-856). Constant, KV-bits-independent memory floor (~47% of routed projection
bytes at fraction 0.53). Also `_prepare_fused_down_for_bank` (2818-2824)
dequantizes the **entire** down matrix to fp32 but consumes only
`[:gpu_start]` columns (2827/2830/2844) — the GPU part reads the quantized
weight directly (2858), so the suffix dequant (~0.5GB/layer transient) is pure
waste. Fixes: slice before dequantizing in `_prepare_fused_down_for_bank`
(small, isolated); ~~free GPU-suffix duplicates once banks are
compiled/latched (medium)~~ — **investigated 2026-08-24, don't implement as
scoped, see 3.6.**

**3.6 investigation finding**: the "free GPU-suffix duplicates post-compile"
half of this fix is unsafe as written. Design-consulted (Fable) given this
touches the same ANE compiled-state subsystem that already caused two real
incidents earlier tonight (the procedure-bank compile-retry jetsam kill, and
the C1 dispatch exception-safety fix, PR #3105). Verified directly against
the code, not taken on faith:
- `_backend` (qwen35_ane_prefill.py:1865-1932) falls through to the
  ORIGINAL, un-split `gate_proj`/`up_proj`/`down_proj` `__call__` — which
  reads the full original `.weight`/`.scales`/`.biases` directly — for any
  input shorter than `config.sequence_length` that doesn't clear the
  tail-padding-profitable bar (line 1883-1894, with the comment "this
  wrapper runs on every MLP call of every layer of every decode") and for
  unprofitable tiling tails (line 1925-1931, `_tail_qmm_or_linear` against
  `mlp.gate_proj` etc. directly). **Every decode token of every request
  needs the originals intact.** Freeing them post-compile would break
  generation on the first decode step, not just some rare edge case.
- `_prepare_pair_runtime_state` (line ~814) backs the admin hardware tuner
  (`omlx/admin/ane_tuning.py`), which sweeps `fraction`/`cpu_fraction`/
  `cpu_down_fraction` candidates mid-session — its own docstring says "CPU
  and GPU boundaries can then move... without... recompiling the ANE
  prefix." Each swept boundary re-slices the ORIGINAL weight at a new
  split point, so even a scheme that only serves the *first* compiled
  variant from cached derived slices doesn't cover the tuner's later
  sweeps.
- What's actually already reclaimed: the fp32 compile slabs
  (`compile_weight0`/`compile_weight1`) are nulled after bank compile
  (~2571-2572, 2588-2589). The remaining "duplicates" — the concatenated
  gate+up GPU-suffix `weight`/`scales`/`biases` and the dense fp16 CPU rows
  — are live runtime operands of the fused native kernels on every call
  that doesn't hit the two fallback paths above, not dead post-compile
  scratch.
- The only real remaining reduction would be a native kernel signature
  change (separate gate/up suffix pointers so the GPU portion can be a
  zero-copy view of the originals instead of a concatenated copy) — real
  [L]-effort work in the same `.mm` file PR #3105 touched tonight, and not
  a good fit for a session that's already hit this subsystem twice.
  Deferred to a calmer session with the admin tuner sweep exercised in
  tests first, so a future attempt doesn't repeat this investigation from
  scratch.

### C4. Per-procedure I/O surfaces and per-op thread churn — optimization

Still per-`AneLinearModel`-instance fp16 input/output IOSurfaces sized by
compiled `sequence_length` (`make_surface` 425-428; allocations 592-595,
718-721, 767-770), and per-op detached `std::thread` + host
`waitUntilCompleted` per layer per chunk (2085/2097, 2380-2397, 2727-2746).
Since evaluation is strictly serialized by `begin()`'s overlap guard (856-859),
one shared surface per (shape, ANE instance) and a persistent serial submit
queue are both safe. Optional: fuse the SwiGLU suffix into the down-qmm
epilogue (2682-2691). Do C1 first — it touches the same dispatch sites.

> **RESCOPED 2026-08-25 (Fable-reviewed against current HEAD, C1 already
> shipped) — this item's own safety premise does not survive contact with the
> code. Effort corrected [M/L] → [L]. Implementation deferred; do not attempt
> the naive version described above.**
>
> **The load-bearing claim above is false as stated.** "Evaluation is strictly
> serialized by `begin()`'s overlap guard" is true only **per `AneLinearModel`
> instance** — the guard's `state_mutex_`/`submitted_`/`completed_` are
> members of `Impl` (now ~line 1045), not process-global. The dual-dispatch
> sites **deliberately run two ANE instances concurrently**: the dual-linear
> path spawns two detached-thread evaluations for `model0`/`model1` (now
> ~2603/2612), and the fused path does the same for `model_`/`model1_` (now
> ~2963/2974), splitting columns across ANE instance hints
> (`ane_execution_options`, ~452-463, `kANEFAneInstanceHint` 1-4) — an
> intentional overlap-for-throughput design, not an oversight. A single global
> persistent serial queue would serialize work the code currently overlaps —
> a behavior *regression*, not a behavior-preserving refactor. Any real
> implementation needs a worker **per (ANE instance / shared-program group)**,
> which the one-line C4 note above does not say. (Cross-instance serialization
> only exists via `shared_program_->evaluation_mutex_`, ~933-935, for models
> sharing one compiled program; non-shared models have no cross-model lock at
> all today, ~940-944.)
>
> **Also note the doc's own line numbers here are stale** — written before the
> C1 fix (`AneDispatchGuard`, now ~1145-1176; `cancel_ticket`, ~966-980) and
> the timeout-latch machinery (`ane_wait_timeout`, ~497+) landed; re-locate,
> don't trust the numbers above.
>
> **Sobering measurement to make before designing anything:** every dispatch
> site host-waits before returning (~2318, ~2632-2633, ~3001-3003) and
> `begin()` blocks on pending work — so queue depth is **already ≤1 per
> instance**. The "persistent submit queue" would be a single-slot mailbox;
> its entire win is thread-spawn overhead, which the existing
> `kAne0LaunchNs`/`kAne1LaunchNs` profiling counters already measure. **Profile
> that overhead first ([S], before any of the below) — if it's noise, the
> worker-queue half of this item should be dropped, not built.**
>
> **This is three severable changes bundled as one item, with different risk
> profiles — land independently if at all, don't do all three in one PR:**
>
> 1. **Persistent submit queue** (the risky half). Hardest part isn't queue
>    mechanics, it's **wedge blast radius**: today a wedged
>    `evaluateWithQoS` parks one *disposable* detached thread; `wait()` times
>    out and poisons only that one `Impl` (~986-999), and the per-module
>    fallback takes over cleanly. A persistent worker parked inside the
>    private ANE framework is lost **for every program routed through it**
>    once wedged. Preserving today's failure isolation means the worker must
>    still spawn-and-abandon on a timeout — detached threads can't actually
>    be eliminated from the failure path, which shrinks this item's payoff
>    further on top of the ≤1-depth finding above. Must also pin down:
>    cancellation ordering (`AneDispatchGuard`/`cancel_ticket` mapped onto
>    *enqueue* instead of *spawn*; what happens if enqueue succeeds but the
>    host throws before `disarm()`), and worker lifecycle vs `~Impl`
>    (~843-872) and `SharedAneProgram` teardown (~475-489).
> 2. **Shared I/O surface per (shape, instance)** (also risky, independently
>    of #1). The guard does **not** actually protect surface sharing —
>    it's emergent from today's per-op-thread-then-host-wait pattern, not
>    enforced. Three concrete holes: (a) the dual-dispatch overlap in #1
>    above means two instances are legitimately live at once by design; (b)
>    the exception-unwind path (~2319-2328) explicitly lets the host unwind
>    *while the detached thread is still evaluating* ("the detached ANE
>    thread holds its own model reference and signals the ticket on its
>    own") — so a next op could pack input into a shared surface while a
>    prior evaluation still reads it; (c) `request_`/`input_object_`/
>    `output_object_` are built from the surfaces at construction
>    (~616-637, ~742-763, ~791-810), so sharing means rebuilding the
>    binding model, and `warmup()` assumes exclusive surface ownership
>    (~1022-1024). Needs either a per-surface lock or a proof the failure
>    path fully drains before reuse — or drop this half explicitly rather
>    than ship it unverified.
> 3. **SwiGLU-into-down-epilogue fusion** (~2682-2691) — already marked
>    optional above; genuinely separable kernel-fusion work, lowest risk of
>    the three, could land alone without touching dispatch/surface
>    lifecycle at all.
>
> **Verification a real implementation would need:** dual-instance overlap
> stays intact (profile `kAne0LaunchNs`/`kAne1LaunchNs`/region-timing
> counters before/after — throughput must not regress); abort-path tests
> exercising `cancel_ticket`; wedge injection via `OMLX_ANE_WAIT_TIMEOUT_S`
> (~497-510) confirming a wedge still poisons only the affected instance, not
> the whole worker; the usual native-change sequence (isolated worktree,
> baseline-then-modified build, packaged-app ABI-load check, live 262k-class
> request) already established elsewhere in this doc.
>
> **Recommendation:** don't implement this in the same pass that produced the
> rest of Phase 4 — this subsystem has already caused two real incidents this
> effort (the procedure-bank compile-retry jetsam kill, and the C1
> exception-safety bug, both discussed above), the doc's own safety argument
> for it doesn't hold, and half its claimed value (the queue) may not exist
> once measured. Do the [S] `kAne0LaunchNs` profiling step first — that alone
> tells you whether items 1-2 are worth attempting at all — then take on #3
> (fusion) or #1/#2 as their own dedicated, isolated-worktree passes with the
> same rigor as the §B1 FA256 streaming-fold implementation.

---

## 5. Theme D — MTP reconcile and sampling correctness

### D1. Reconcile failure silently ignored + unchunked re-prefill — do together

`omlx/patches/mlx_lm_mtp/batch_generator.py`:

- Lines 194 and 221 call `_reconcile_mtp_to_standard(...)` and **discard the
  bool**, then unconditionally `_drop_mtp_state` — standard decode resumes from
  stale `_next_tokens` against an already-advanced MTP cache: duplicated or
  garbled continuation, no error. (Line 1068 shows the correct pattern — it
  checks the return.)
- The reconcile fallback re-prefills the **entire** streamed history in one
  unchunked `_call_backbone` call (line 1180; `_call_backbone` 1381-1410 has no
  chunking and bypasses the prefill guard). The code's own comment (1195-1197)
  concedes the risk. On a long context this is a likely OOM trigger — inside
  the failure path, where it hurts most.

**Fix (one change, medium).** Chunk the reconcile re-prefill through the normal
prefill machinery (guard-priced chunks), and make both call sites at 194/221
honor the return value: on `False`, drop the sequence with an explicit error
rather than continuing corrupted.

### D2. XTC breaks MTP rejection-sampling exactness — MEDIUM

`_accept_lp_for` (1259-1293) reconstructs temp/top_p/min_p/top_k only; XTC
(`omlx/utils/sampling.py`, `apply_xtc` engaged when `xtc_probability > 0`) is
applied at emit time but absent from the acceptance math, so the emitted
distribution diverges from the target. There is no `xtc` reference anywhere in
`omlx/patches/mlx_lm_mtp/`. Fix (small): gate MTP off for requests with
`xtc_probability > 0` at eligibility time (exactness first; modeling XTC in
acceptance math can come later if ever needed).

### D3. Micro-wins (bundle with any D-area PR)

- Bonus row: line 2842 filters all `k+1` rows through the full-vocab acceptance
  transform but only `[:k]` are consumed (2845, 2854); the bonus token is drawn
  separately (2859). Skip the bonus row.
- Line 2945 queues MTP-head (`draft`) logprobs for accepted tokens instead of
  the target/combined distribution — reported logprobs are wrong-source. Queue
  the target-model logprobs.

---

## 6. Theme E — Kernel routing and numerical safety

### E1. One q2 qmm call permanently disables NAX process-wide — HIGH (latent), trivial fix

Chain fully intact: `omlx/custom_kernels/qwen35_prefill/fast.py`
`_qmm_nax_kwargs()` (933-936) has no bits parameter and the q2 wrapper
forwards it (`qwen35_q2_affine_qmm_t`, 1013-1034);
`qwen35_prefill.cpp` applies no bits guard (997) and the q2 kernel-lookup catch
at 675 stores `nax_qmm_runtime_ok=false` (atomic at line 85) — while
`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_qmm_nax.metal` only defines
bits 4/5/6/8 (108-124). One q2 layer ⇒
every q4/q6/q8 layer demoted to classic kernels for the process lifetime.
**Fix (small): gate `use_nax` on `bits != 2` in `_qmm_nax_kwargs`/the q2
wrapper.** Highest fix-value-per-line in this doc.

### E2. GDN chunked kernels: OOB tail reads, fp16 narrowing, over-permissive gate

`omlx/custom_kernels/qwen35_prefill/gdn.py` (chunked kernels A/B are the
**non-default** `OMLX_GDN_IMPL=chunked` path; default blocked_seq kernel S is
properly masked):

- Tail-chunk OOB: kernel A staging (95-99) and kernel B staging (288-290) read
  `k_base[(t0+i)*row + ...]` up to C=64 with no `tt` bound (masking happens
  only in compute). UB; can fault at page boundaries. Fix: bound the staging
  loops (predicate the load, zero-fill the tail).
- fp16 narrowing: `threadgroup half U_s` (226) and the delta-rule update
  narrowed through `(half)` (275) contradict the module's fp32-state contract
  (docstring line 27) — Inf risk on activation spikes. Fix: keep U in fp32
  threadgroup memory (or at minimum saturate/clamp), accept the occupancy cost
  on this non-default path.
- Route gate: `omlx/patches/qwen35_gdn_chunked.py:84-85` admits any
  `Dk % 16 == 0 && Dv % 32 == 0`, but kernels A and S hard-assume 128
  (gdn.py 155-156; 441-461). The review asserted `qwen3_5`-family configs with
  Dk=192/Dv≠128 exist (unverified this pass — confirm against actual family
  configs before landing, though the tightened gate is safe regardless).
  Fix (small): require `Dk == 128 and Dv == 128` in the gate.

Cheap alternative if the chunked path has no users: delete/hard-disable
`OMLX_GDN_IMPL=chunked` and keep only blocked_seq.

### E3. GDN prework fallback runs on clobbered state — MEDIUM, small fix

`omlx/patches/qwen35_gdn_prework.py`: `cache[0] = new_conv_state` at 243; any
exception after it (delta update 249, norm 279, out_proj 280) reaches the
except branch (283-291) which calls `orig_call` on the clobbered conv state.
Also, exceptions after `cache.advance(S)` (267-270) double-advance under the
fallback. Fix: snapshot `cache[0]` (and offset) before the mutation window,
restore both in the except branch.

### E4. fa256 treats `mask=None` as causal — MEDIUM

`omlx/patches/qwen35_fa256_attention.py:157`: `mask=None` passes `_should_route`
and both lm/vlm paths then invoke the kernel with `causal=True` (259/263,
311/313). Any caller relying on `None` = "no mask" gets silently causal
attention. Fix: route only on explicit `mask == "causal"`; audit call sites for
who actually passes `None` before changing behavior.

### E5. sdpa256 ratchet: per-layer state viewed through a global logger — LOW (observability)

Post-`e55ffd17` the ratchet keys on q_sub size, but all three review findings
stand (`omlx/patches/sdpa256_attention.py`): the ratchet lives on the cache
object (427-435) whose "stable across layers" comment (424-426) is wrong — the
model passes per-layer caches, so 16 full-attention layers ratchet
independently; `_LAST_ROUTE_DECISION` is one module-global (110, 119-122), so
interleaved layers/requests flap the transition log; and the `ceiling == 0`
tiled latch (205, 431-432) is never cleared — benign only because caches die
with the request. Cost is log noise, not compute (no recompilation either
route). Fix (small): key the ratchet per-request (cache-list identity or
request id), scope/rate-limit `_note_route`, and either document or clear the
0-latch.

---

## 7. Theme F — On-disk durability and store hot paths

### F1. No fsync before rename, no payload checksum — HIGH (crash window)

All three writers confirmed: main SSD blocks (`paged_ssd_cache.py`
`_write_block_file` 2940-2958 → `_write_safetensors_no_mx` 854-905, `os.rename`
2958), GDN sidecars (staged via boundary store writer, promoted with
`os.replace` at 2383), boundary snapshots (`boundary_snapshot_store.py` 979,
1133). No `os.fsync` on file or directory, no checksum in the format. Crash or
power loss ⇒ a rename can land pointing at zero-length/zero-filled payload,
read back silently as garbage KV. Fix (medium): `flush()` + `os.fsync(fd)`
before rename (+ directory fsync where cheap), and add a payload checksum
(xxhash64 of the tensor bytes in the header) verified on load; treat mismatch
as cache-miss, not error.

### F2. Preload runs multithreaded `mx.load` — HIGH (conditional deadlock)

`paged_ssd_cache.py:4152-4180`: `ThreadPoolExecutor(max_workers≤8)` running
`mx.load` + eval — directly contradicting the same file's ban comment
(3773-3778) citing Metal-contention deadlocks (MLX issues #978 #1040 #1106
#1437 #1558). Fix (small): serialize the `mx.load`/eval portion (I/O may stay
parallel; do the array materialization on one thread), matching the
`load_block` discipline.

### F3. Store-path efficiency (bundle, low risk)

- `save_block` evals tensors one-by-one (`_extract_tensor_bytes` → `mx.eval`
  per array, `paged_ssd_cache.py:819, 3379-3381`); boundary store already
  batch-evals (`mx.eval(*arrays.values())`). Measured 19ms/48-layer state.
- Sidecar commit file I/O runs under the global `self._lock`
  (`commit_gdn_checkpoint_file`, 2351-2412). Move mkdir/stat/`os.replace` out.
- Prefix hashing: `str(tuple(token_ids))` serialization
  (`paged_cache.py:113`); hash computed twice per block (store_cache 1056 then
  `register_block_hash` → recompute at `paged_cache.py:1145`); loop-invariant
  `_get_cache_seq_len` inside the per-block loop (`prefix_cache.py:1078`).
  One-pass packed-int hashing + pass the precomputed hash + hoist.
- Paged-cache pressure valves are decorative: `reset_prefix_cache` compares
  against `max_blocks` not currently-allocated (`paged_cache.py:1409`);
  `evict_lru_blocks`/`handle_memory_pressure` (1317-1352) only shuffle
  already-free blocks. Either make them evict ref-0 cached blocks for real or
  delete them so callers stop believing they work.

### F4. turboquant fallback + finalize

- `turboquant_attention.py:657`: batch-view mismatch fallback calls
  `real_cache.dequantize()` with no `keys_state`/`values_state` — materializes
  the entire resident cache densely in one shot (instant OOM at long context)
  instead of the passed views. Fix (small): dequantize only the passed states,
  or chunk.
- `omlx/turboquant_kv.py:362-372` `finalize()`: dequantize → `dynamic_roll` →
  requantize round trip. Rolling the per-token `norms`/`indices` tensors
  directly is exact (no fp16 materialization, no re-rounding). Medium effort,
  pure win.

---

## 8. Phased implementation checklist

Ordering: correctness before optimization; low-risk/high-confidence first;
items sharing code are bundled. Effort tags: S(<~1h) / M(half-day) /
L(multi-day). Risk is conveyed by phase grouping (Phase 1 = low-risk
high-confidence; Phase 2 = needs design judgment; Phases 3-4 = capacity/perf),
not per-item.
All line refs verified 2026-08-20 @ `ce357792`; anything executed later than a
few weeks from then should re-grep the quoted identifiers first.

### Phase 0 — Instrumentation (unblocks Phase 3 decisions)

- [x] **0.1** [S] Log all admission-bound terms at rejection: `current`,
  predicted transient, `observed_max_bytes`, `_ane_prefill_transient_bytes`
  (`scheduler.py:4001,3775-3799`; `memory_monitor.py:805`). Re-run the 262k
  rejection with ANE on / off-at-load / off-at-runtime; record which term
  moves. **Prerequisite for 3.3 — do not assume the ANE reservation fix moves
  the needle until this says so** (the empirical ANE-off null result is
  currently explainable both as "term was already 0 at load-time disable" and
  as "term isn't dominant").

### Phase 1 — Correctness, low-risk / high-confidence

- [ ] **1.1** [S] Gate `use_nax` on `bits != 2`
  (`custom_kernels/qwen35_prefill/fast.py:933-936,1013-1034`; latch at
  `csrc/qwen35_prefill.cpp:85,675`). One-line guard kills a process-wide perf
  cliff. (§E1)
- [ ] **1.2** [M] GDN prefix-store desync: placeholder instead of live-state
  fallback in all three embedded branches of `_extract_block_tensor_slice`
  (`prefix_cache.py:~2100-2113, ~2273-2288, ~2314-2327`) + regression test.
  (§A1)
- [ ] **1.3** [S] Shared-prefix path: switch `increment_ref` →
  `acquire_cached_block` with expected hash (`prefix_cache.py:629-635`,
  mirror 662-672). (§A2)
- [ ] **1.4** [M] MTP reconcile bundle — one PR: honor the bool at
  `batch_generator.py:194,221` (drop sequence with explicit error on False)
  AND chunk the re-prefill at 1180 through guard-priced prefill. These touch
  the same failure path; do together. (§D1)
- [ ] **1.5** [S] Gate MTP off when `xtc_probability > 0` at eligibility time
  (`batch_generator.py:1259-1293`; `utils/sampling.py`). (§D2)
- [ ] **1.6** [S] GDN prework: snapshot/restore `cache[0]` + advance-offset in
  the except branch (`qwen35_gdn_prework.py:243,267-270,283-291`). (§E3)
- [ ] **1.7** [S] turboquant dense fallback: dequantize only the passed
  `keys_state`/`values_state` (`turboquant_attention.py:657`). (§F4)
- [ ] **1.8** [S] Serialize `mx.load`/eval in the preload path
  (`paged_ssd_cache.py:4152-4180`; discipline documented at 3773-3778). (§F2)
- [ ] **1.9** [S] GDN route gate: require `Dk == 128 and Dv == 128`
  (`qwen35_gdn_chunked.py:84-85`). (§E2)

### Phase 2 — Correctness, needs more care / design judgment

- [ ] **2.1** [M] fsync-before-rename + payload checksum for SSD blocks, GDN
  sidecars, boundary snapshots (`paged_ssd_cache.py:854-905,2940-2958,2383`;
  `boundary_snapshot_store.py:979,1133`). Checksum mismatch ⇒ cache miss.
  Benchmark the fsync cost; consider batching directory fsyncs. (§F1)
- [ ] **2.2** [M] ANE begin()/end() exception safety on single + dual paths
  (`qwen35_ane.mm:2009-2011..2098, 2317-2320`), using the fused path
  (2644-2648) as the template. (§C1)
- [ ] **2.3** [M] ANE failure latches release state: detach
  `_omlx_ane_*_state`, drop dense weight copies, re-invoke `set_model_info`
  (`qwen35_ane_prefill.py:2610-2650,2581`). Division of labor: **2.3 owns the
  detach + re-price on latch/disable; 3.3 is only the charge-once accounting
  change** — don't do the detach twice, and 2.3 does not need to wait on 0.1.
  (§C2)
- [ ] **2.4** [M] GDN chunked kernels: bound tail-chunk staging loads
  (`gdn.py:95-99,288-290`) and keep U in fp32 (226,275) — or hard-disable
  `OMLX_GDN_IMPL=chunked` if unused (decide first; blocked_seq is default and
  clean). (§E2)
- [ ] **2.5** [S] fa256 `mask=None`: audit callers, then route only explicit
  `"causal"` (`qwen35_fa256_attention.py:157,259-263,311-313`). Behavior
  change — needs the audit before the one-liner. (§E4)
- [ ] **2.6** [S] Validate `token_count` against actual stored KV length at
  store time (`prefix_cache.py:1051,1183,1193`). (§A3)
- [ ] **2.7** [S/M] sdpa256 observability: per-request ratchet keying +
  scoped/rate-limited `_note_route`
  (`sdpa256_attention.py:110,119-122,424-435`). (§E5)
- [ ] **2.8** [M] Fix or remove the paged-cache pressure valves
  (`paged_cache.py:1317-1352,1409`). (§F3)

### Phase 3 — Memory-guard accuracy & long-context capacity (only 3.3 is gated on 0.1; 3.1/3.2/3.4-3.6 can proceed independently)

- [x] **3.1** [L] FA256 online-softmax streaming accumulator replacing the
  n-chunk partial slab (`qwen35_prefill.cpp:325-391`). Highest-leverage
  capacity item; directly implicated by the 4-bit-only-helps-16%
  measurement. (§B1) — **SHIPPED 2026-08-25**: implemented behind
  `stream_fold` (Fable-reviewed design, fixed a within-dispatch race the
  original plan spec missed), numerics validated (≤4.9e-4 vs legacy, zero
  added error vs fp32 SDPA), transient 2080→193 MB (10.8×) measured, default
  flipped ON, deployed + live-validated on both cluster nodes (50k-token
  request end to end). KV A/B re-run confirms the §B1 hypothesis: 4-bit's
  capacity benefit over 8-bit is now +98% (was +16% under the fallback path
  that masked it). Also surfaced and fixed an unrelated but load-bearing
  blocker found during rollout: native ANE prefill failed at large-context
  occupancy (§ Theme C addendum, headroom-gated in `qwen35_ane_prefill.py`).
  Full implementation record in §B1; harnesses in `docs/fa256_*`.
- [x] **3.2** [S] Register turboquant's 256×16384 prefill tile with the memory
  monitor, mirroring sdpa256's `register_tiled_prefill_head_dim`
  (`turboquant_attention.py:26-27,635-638`). (§B3) — shipped per-instance
  (not the naive global mirror; see §B3 writeup for why).
- [ ] **3.3** [M, gate not met] ANE reservation charge-once semantics — ONLY
  IF 0.1 shows the term matters: subtract the reservation once the surfaces
  are resident in measured `current` (`memory_monitor.py:805,476,1377-1388`;
  `qwen35_ane_prefill.py:3321-3354`). The detach-on-latch/disable half lives in
  2.3. (§B2) — **checked 2026-08-24**: `~/.omlx/logs/server.log` has only 2
  admission-rejection log lines from the 0.1 instrumentation so far
  (2026-08-24 03:39 and 07:49), and both show `ane_prefill_transient_bytes=
  0.00GB` — the ANE reservation term isn't contributing to either observed
  rejection (both were dominated by `current` near the ceiling). Sample size
  is small (n=2), but the explicit gate condition ("only if 0.1 shows the
  term matters") isn't met by the evidence that exists — don't implement
  yet. Revisit once more rejection samples accumulate, especially any with
  ANE prefill actually active (both samples show `ane_prefill_transient_
  bytes=0.00GB`, meaning ANE may not even have been enabled/loaded for
  either of these two requests — worth confirming before concluding
  anything stronger than "no evidence yet").
- [x] **3.4** [S] Admission observed-max: decay/windowed max or reset on model
  swap (`scheduler.py:3775-3799`; `prefill_transient_tracker.py:114-125`, plus
  the never-called `reset()` at ~194). (§B4)
- [x] **3.5** [S] `_prepare_fused_down_for_bank`: dequantize only
  `[:gpu_start]` columns (`qwen35_ane_prefill.py:2818-2844`). (§C3)
- [x] **3.6** [M→L, deferred] ~~Free duplicated GPU-suffix / dense-fp16 ANE
  weight copies post-compile~~ (`qwen35_ane_prefill.py:437-445,516-524,
  646-652,772-778,850-856`). Investigated 2026-08-24, not implemented as
  scoped — unsafe (breaks every decode step + the admin hardware tuner's
  boundary sweep). Real fix is an [L]-effort native kernel signature
  change, deferred. See §C3 writeup for the full finding. (§C3)

### Phase 4 — Pure optimizations

- [x] **4.1** [S] MTP micro-wins: skip bonus-row acceptance filter
  (`batch_generator.py:2842-2859`); queue target-model logprobs for accepted
  drafts (2945). Bundle with any Phase-1.4 follow-up. (§D3) — **SHIPPED
  2026-08-25** (commit `1e3ebe4f`): sliced `combined_lp[:k]` before
  `_accept_lp_for` instead of after (the bonus row was never read from its
  output); queued `combined_lp[j]` (verified backbone logprobs) instead of
  `state.draft_lps[j]` (the MTP-head's own draft-time distribution) for
  accepted tokens. 190 MTP tests pass. **⚠️ Upstream overlap found**: draft
  PR jundot/omlx#3115 (`michaelasper`, WIP) touches the same
  `state.queue.append(...)` line in `_run_verify_cycle_chain` for a different
  reason (deferred/batched normalization of greedy draft logits) — it does
  NOT fix the wrong-source-logprobs bug (still reads from `state.draft_lps`
  in that PR as of this check) and doesn't touch the bonus-row slice, so it's
  not a duplicate, but an eventual upstream PR for this item will need manual
  reconciliation against #3115 if/when it lands first.
- [x] **4.2** [S] Batch-eval tensors in `save_block`
  (`paged_ssd_cache.py:819,3379-3381`; pattern at
  `boundary_snapshot_store.py:~1305`). (§F3) — **SHIPPED 2026-08-25** (commit
  `d7dee9c0`): `mx.eval(*arrays.values())` once before the extraction loop,
  mirroring boundary_snapshot_store exactly. 170 tests pass. No upstream
  overlap (origin/main unchanged in this region).
- [x] **4.3** [S→ scope grew to a real concurrency analysis] Move sidecar
  commit file I/O out of the global lock (`paged_ssd_cache.py:2351-2412`).
  (§F3) — **SHIPPED 2026-08-25** (commit `6840f921`). The terse doc note
  undersold the coupling (Fable-reviewed): `enforce_size_limit_for_new_block`
  + the index remove/add bracket `os.replace` to protect the in-flight file
  from `forget_gdn_checkpoint`'s own remove+unlink composite, not
  incidentally. Verified independently: `self._lock` is an RLock,
  `GDNCheckpointIndex` has its own internal RLock (index integrity never
  depended on `self._lock`), and `commit_gdn_checkpoint_file`'s only caller
  path runs through a single-worker executor (commit-vs-commit races
  structurally impossible). Shipped the two-phase structure this implies:
  directory-safety checks hoisted out entirely; lock held only to
  remove-old/add-new the index entry, not across `os.replace`/fsync/the
  inline eviction unlinks inside `enforce_size_limit`. 306 tests pass. No
  upstream overlap.
- [x] **4.4** [M] Prefix hashing: one-pass packed-int hashing; pass precomputed
  hash into `register_block_hash`; hoist `_get_cache_seq_len`
  (`paged_cache.py:113,1145`; `prefix_cache.py:1056,1078`). (§F3) —
  **SHIPPED 2026-08-25** (commit `f91c8ea5`): `struct.pack` instead of
  `str(tuple(token_ids))`; `register_block_hash(..., precomputed_hash=...)`
  optional passthrough (default `None`, existing direct callers unaffected);
  `_get_cache_seq_len` memoized on first actual need inside the per-block
  loop, not eagerly hoisted before it (many blocks dedup via
  `continue`/`break` before reaching it — an eager hoist would regress the
  all-cache-hit path from zero cost to always-pay). 599 tests pass, all
  behavior-relative (no golden hash values broke). **Deploy note**: the hash
  algorithm change means on-disk paged-SSD cache blocks from before this
  ships go cold (unreachable under the new keys) — not a leak, existing LRU
  eviction reclaims them like any other cold entry. No upstream overlap
  (both files unchanged vs origin/main).
- [x] **4.5** [M] turboquant `finalize()`: roll `norms`/`indices` directly, no
  dequant round trip (`omlx/turboquant_kv.py:362-372`). Exactness win too.
  (§F4) — **SHIPPED 2026-08-25** (commit `a5884411`): added `_roll_state`,
  a recursive dispatcher over every TurboQuant state variant (mirrors the
  file's existing `_slice_state`/`_concat_state`/`_filter_state` pattern),
  applying `dynamic_roll` directly to each field sharing the token axis.
  Verified structurally against the `mlx_vlm.turboquant` source that
  dequantize/quantize are pure per-token operations (no cross-token state),
  so the reindex commutes exactly with dequantize — confirmed empirically
  with new parametrized (bits=1.0/4.0/8.0) tests asserting
  `dequantize(finalize(state)) == dynamic_roll(dequantize(state))` bitwise.
  56 tests pass. No upstream overlap.
- [ ] **4.6** [M/L → corrected to L] ANE dispatch: persistent serial submit
  queue replacing per-op detached threads; shared I/O surface per (shape,
  instance); optional SwiGLU-into-down-epilogue fusion
  (`qwen35_ane.mm:2085-2097,2380-2397,2727-2746,425-428,592-595,2682-2691`).
  **RESCOPED 2026-08-25, NOT IMPLEMENTED — see the design-correction block
  inline in §C4 above.** The item's own safety premise (serialization via
  `begin()`'s overlap guard) is false as stated — the dual-dispatch paths
  deliberately run two ANE instances concurrently by design — and queue
  depth is already ≤1 per instance (every dispatch site host-waits), so the
  "persistent queue" may have no real win once `kAne0LaunchNs` overhead is
  measured. Deferred to a dedicated pass per Fable's review; not a gap in
  this session, a considered decision. No upstream PR exists for this
  (checked jundot/omlx — nothing touches `qwen35_ane.mm`).
  Do after 2.2 — same dispatch sites. (§C4)

---

## 9. Already resolved / verified non-issues — do not re-investigate

- **MTP draft depth (DONE, tested, verified live).** The real bug was the admin
  API: `ModelSettingsRequest` lacked `mtp_num_draft_tokens` (Pydantic silently
  dropped it) and `_engine_runtime_signature` didn't cover it. Both fixed
  (`admin/routes.py:178,2671-2672`; `engine_pool.py:527-530`), with tests
  (`tests/test_admin_model_settings.py::test_mtp_num_draft_tokens_is_persisted`,
  `::test_mtp_num_draft_tokens_change_unloads_a_loaded_engine`,
  `tests/test_engine_pool.py::...runtime_signature...`). Live measurement:
  depth 3→8 gave only 1.05× decode (acceptance decays sharply with depth:
  d4=100%, d5=62%, d6=80%, d7=50%, d8=25% conditional) — the adaptive
  `_DepthController` staying shallow is *correct*; the fix just permits depth
  when content warrants.
- **GDN blocked_seq kernel IS active on the serving engine** — confirmed via
  startup log ("GDN prefill kernel patch applied (Metal, impl=blocked_seq,
  min_t=64)") and engagement during real generation. The `mlx_vlm`-only-rebind
  worry was unfounded.
- **Fixed by post-review commits** (verified at HEAD):
  - Prefix-index fetch path now hash-validates via `acquire_cached_block` and
    self-heals stale index entries (`prefix_cache.py:662-672`) — the original
    "skip missing block, corrupt splice" hole is gone. Only the shared-prefix
    path remains (item 1.3).
  - Ref-0 cold-registered SSD blocks are now lazily registered inline during
    lookup (`paged_cache.py:1059-1074`) — visible immediately, no sweep wait.
  - GDN sidecars are budget-tracked (`_tracked_ssd_size` includes
    `_gdn_sidecar_index.total_size`, `paged_ssd_cache.py:2179-2190`).
  - Boundary-store load no longer does ~48 GPU syncs under `_pending_lock`
    (lazy byte-rebuild; slow-path `mx.load` outside the lock).
  - The fused ANE dispatch path has protected begin()/end() (template for 2.2).
  - Split-GDN prefix storage uses placeholders + separately committed
    checkpoints (narrows A1 to embedded layouts).
  - Observed-max ratchet partially mitigated (floor-only samples, 4GiB clamp) —
    residual is item 3.4.
- **Reshaped, not fixed:** ANE transient accounting was rewritten by `dcb317fe`
  (reservation from live attached state; dirty-tracking removed). The review's
  "double-charge after first dirty" framing is obsolete — see §B2 for the
  current model and why 0.1 must run first.
- **Dropped:** the "negative `_hot_cache_total_bytes` in SSD-only mode" finding
  is no longer reconstructible (add/remove are symmetric under one lock; getter
  clamps at `paged_ssd_cache.py:1796`). Re-open only if observed. Hot-cache
  raw-bytes storage is now documented-intentional (Metal memory release).
