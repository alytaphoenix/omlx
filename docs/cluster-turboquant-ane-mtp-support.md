# Cluster support for TurboQuant KV, Qwen ANE-prefill, and MTP

Design doc + phased implementation checklist for enabling the three
currently-gated (or silently-ignored) single-node optimizations under the
2-Mac distributed cluster. Every file:line reference below was **verified
against HEAD `cb281726` (branch `deploy/session-fixes-v2`) on 2026-08-25**.
Line numbers will drift as the tree moves — treat them as anchors (the quoted
identifiers are the stable handles), and re-locate rather than trust a stale
number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here). References into the
**pinned runtime packages** are written `site-packages/mlx_lm/...` and
`site-packages/mlx/...` (under `.venv/lib/python3.11/site-packages/`) — those
are not repo files, and a pin bump invalidates them wholesale.

Findings are labeled **CONFIRMED** (the full causal chain was traced in code)
or **PLAUSIBLE** (mechanism traced, final runtime behavior needs a repro or
real hardware). Where a claim genuinely cannot be settled from code, the
required empirical test is named instead of guessed at.

Companion doc: `docs/cluster-hardening-and-optimization.md` (the cluster
subsystem review). Its Theme A — lockstep execution and partial failure — is
the failure class that dominates the MTP analysis here; references to "A1"
below point into that doc.

---

## 1. Context

### The single architectural fact that drives all three analyses

The distributed path does not run oMLX's engine. `DistributedBatchedEngine`
(`engine/distributed.py`) keeps only tokenizer/config metadata on the
coordinator and proxies inference over HTTP to a private mlx-lm server; the
per-rank process is `cluster/inference_worker.py::run_worker`
(inference_worker.py:936), which loads its shard and then serves with the
**pinned `mlx_lm.server`** (`ModelProvider` / `ResponseGenerator`,
inference_worker.py:1093). Consequences:

- **`model_settings` never reach a rank.** Zero references in
  `inference_worker.py` or `cluster/deployment.py`; the worker contract
  (`_server_arguments`, inference_worker.py:389) carries only execution knobs
  (prefill_step_size, concurrency, cache sizes). Any setting-driven feature is
  structurally inert on the cluster path unless explicitly plumbed.
- **All three optimizations activate only in code the cluster never runs**:
  `BatchedEngine.start()` (`engine/batched.py:328` TurboQuant, `:397`
  ANE-prefill) and `utils/model_loading.py:635` (MTP head attachment) on the
  single-node path, plus `engine/vlm.py` equivalents.
- **KV cache is rank-local by construction** (`kv_cache_scope="rank_local"`,
  ready-marker at inference_worker.py:1163). No collective ever touches KV
  state; the TP collectives live in `o_proj` / `down_proj`
  (`sharded-to-all` linears, `site-packages/mlx/nn/layers/distributed.py:333`
  and `:585`).
- **All ranks run the generation loop in lockstep** with MLX-LM's
  synchronized sampler under pure TP
  (`cluster/runtime_optimizations.py:217`: "pure tensor parallelism keeps
  MLX-LM's synchronized sampler"). Every rank must execute the identical
  sequence of collectives or the deployment desyncs (companion doc, A1).

### The gate, and what its history says

`_validate_model_settings` (`engine/distributed.py:258-277`) raises
`ValueError` at activation for: `dflash_enabled`, `specprefill_enabled`,
`mtp_enabled`, `vlm_mtp_enabled`, `turboquant_kv_enabled`. The list was born
wholesale in the initial cluster commit (`08b19a89`, 2026-07-30, "one-click
activation API") with no per-flag rationale. `thinking_budget_enabled` was on
the original list and was **removed** when the feature was implemented on the
distributed path (#2731, `de10ae75`, "align thinking budget behavior across
engines"). The gate is therefore a **"not yet implemented on this path"
list, not an incompatibility list** — entries come off it by porting the
feature to the worker, and there is precedent for exactly that.

`qwen35_ane_prefill_enabled` is *not* on the list: a cluster activation with
an ANE-prefill profile succeeds with no error and no ANE (finding A3 below).

### Deployment facts constraining validation

The peer node (`M-FJX1D769D0.local`) is **off the network** until further
notice — every empirical test in this doc is **coordinator-local**: a
2-process `mlx.launch` ring over localhost on the coordinator (TP=2, both
ranks on one Mac). That harness settles logit-parity and lockstep questions;
it does **not** exercise Thunderbolt/jaccl transport, cross-chip ANE
heterogeneity, or dual-machine memory behavior — those need the peer back.
The most recent real deployment was Nemotron-3.5-Lightning-30B-A3B (oQ4,
`nemotron_h`) at TP=2 — which is itself an MTP-capable model_type
(depth-1 head, `utils/model_loading.py:657`), so the MTP theme is directly
relevant to the hardware this cluster actually serves.

---

## 2. Theme A — Shared plumbing gaps (all three features)

### A1. Settings do not ride the launch contract — MEDIUM (CONFIRMED)

`build_mlx_launch_argv` (`cluster/launch.py:321`) builds one argv shipped
verbatim to every rank; per-node values ride `PipelineAssignment` inside the
signed plan. All settings needed here (TQ bits/skip_last, ANE fractions/
sequence length, MTP depth) are **uniform across ranks**, so plain argv flags
are the correct channel per the launcher's own docstring — no plan-schema
change needed. The worker must parse them and thread them to the respective
apply/enable calls. This is the common prerequisite for every phase below.

### A2. The worker already has the pre-load patch seam — LOW (CONFIRMED)

`run_worker` already calls `maybe_apply_pre_load_patches(args.model)`
(inference_worker.py:1061) — today without `model_settings`, so the MTP
branch inside it (`utils/model_loading.py:635`) computes `mtp_enabled=False`
and skips head attachment while still applying the sanitize-correctness
patch. Passing settings through this existing call is the entire model-side
MTP hook; TurboQuant's SDPA patch and ANE's enable call need new (small)
call sites in `run_worker`.

### A3. ANE-prefill silently no-ops under cluster today — MEDIUM (CONFIRMED)

Asymmetric with the gated flags: TurboQuant/MTP fail loud at activation;
ANE-prefill is simply ignored (not gated, settings never forwarded,
`server.py:2568 _ane_prefill_status` reports inactive). A user comparing
cluster vs single-node throughput for an ANE profile gets a silently
different configuration. Until Phase 2 lands, add
`qwen35_ane_prefill_enabled` to the gate list (or surface "ignored under
cluster" in activation status). Five lines plus a test.

---

## 3. Theme B — TurboQuant KV under cluster

### B1. The math is TP-clean; there is no sharding hazard — no-action (CONFIRMED)

- Qwen3.5's native `shard()` (`site-packages/mlx_lm/models/qwen3_5.py:445`)
  and both explicit TP adapters (`cluster/tensor_strategies.py:332,475`)
  split attention **by heads**; `head_dim` stays intact on every rank.
- TurboQuant quantizes per head **along `head_dim`** (kernels require
  `dim % 32 == 0`; extraction is per-head-dim-lane,
  `patches/turboquant_attention.py:47-78`). Quantization groups therefore
  never cross a rank boundary.
- KV state is rank-local and collective-free (Context); the `all_sum` fires
  in `o_proj` *after* attention, so quantization error feeds the collective
  identically on all ranks and the synchronized sampler stays coherent.
- Under pipeline parallelism each rank owns whole layers — trivially clean.

The gate entry for `turboquant_kv_enabled` is defensive, not architectural.
"Lifting" it means porting the activation to the worker, not deleting a line.

### B2. Port surface: SDPA patch + cache construction — MEDIUM effort (CONFIRMED mechanism)

Two integration points, both bounded:

1. `apply_turboquant_attention_patch()`
   (`patches/turboquant_attention.py:516`) is process-global and already
   rebinds `scaled_dot_product_attention` in every already-imported
   `mlx_lm.models.*` / `mlx_vlm.models.*` module (`:677-687`), so calling it
   once in `run_worker` (before or after load — ordering is handled) covers
   the sharded model's attention.
2. The pinned server creates per-request caches via `make_prompt_cache(model)`
   (`site-packages/mlx_lm/server.py:971`). A worker-side wrapper must convert
   eligible `KVCache` layers to `TurboQuantKVCache`, reusing the eligibility
   logic of `Scheduler._turboquant_eligible` (`scheduler.py:3146`: no MLA, no
   attention sinks, pass-through for GDN `ArraysCache` / rotating caches) —
   extracted into a shared helper rather than duplicated.

### B3. Batch-merge protocol compatibility is the one empirical unknown — LOW-MED (PLAUSIBLE; bounded downside)

The pinned server computes `is_batchable = all(hasattr(c, "merge") ...)` at
load (`site-packages/mlx_lm/server.py:372`). `TurboQuantKVCache` has
`is_trimmable`/`trim` but no `merge`; the batch protocol lives on oMLX's
`BatchTurboQuantKVCache` (`turboquant_kv.py:276`, merge/extract/extend/
filter), built against oMLX's scheduler, and whether its signatures match the
pinned server's continuous-batching machinery has not been verified. The
downside is bounded and already handled gracefully:
`cluster/runtime_optimizations.py:255` reports "this model's KV cache cannot
be merged, so MLX-LM serves [sequentially]" — a throughput cost, never a
correctness cost. Resolve empirically in the Phase 0 harness before deciding
whether protocol glue is worth writing.

### B4. `turboquant_skip_last` must key on the global layer index — LOW (CONFIRMED)

Under pipeline parallelism, a naive per-rank "skip the last KVCache layer"
skips each rank's *local* last layer — one extra unquantized layer per
non-final rank. Over-conservative (extra memory), not corrupt, but the worker
plumbing must translate the flag against `assignment.start_layer/end_layer`.
Under pure TP (every rank holds every layer) the flag behaves identically to
single-node.

---

## 4. Theme C — Qwen ANE-prefill under cluster

### C1. Naive enablement drops the down-projection `all_sum` — silent output corruption — HIGH (CONFIRMED)

The patch wraps the **whole MLP class** `__call__`
(`patches/qwen35_ane_prefill.py:2000 _wrap_class`; dispatch install at
`:2016`). Under TP, `down_proj` is a `QuantizedShardedToAllLinear` whose
`__call__` carries the `mx.distributed.all_sum`
(`site-packages/mlx/nn/layers/distributed.py:585`). Every ANE-engaged path
computes the down projection from **raw weight/scales tensors**, bypassing
the module and its collective:

- fused-down ANE kernels (`_backend_exact`, `:1699` on:
  `fast.qwen35_ane_dual*_swiglu_down_t(...)` from stashed weight blobs),
- `_post_ane_down` (`:953` → `_post_ane_linear` / raw dual-qmm),
- the long-tail branch of `_tail_qmm_or_linear` (`:323` → `_linear_qmm`).

Only the short-tail branch (`linear(x)`, `:322`) goes through the module.
Result: whenever the ANE path engages on a sharded model, the MLP returns
**rank-partial sums** — plausible-looking garbage, not a crash. This is the
one finding in this doc in the data-corruption class.

### C2. And it would engage: eligibility survives sharding — MEDIUM (CONFIRMED shape identities; attribute path PLAUSIBLE-high)

`shard_linear` produces `QuantizedAllToShardedLinear` /
`QuantizedShardedToAllLinear`, which expose `weight/scales/bits/group_size`
(`site-packages/mlx/nn/layers/distributed.py:394-395,413,529-530,548`) — the
attributes `_eligible_pair` (`qwen35_ane_prefill.py:367`) checks. Its shape
identities hold on halved shards (gate/up out-cols and down in-rows halve
together). So C1 is *reachable*, not hypothetical: the wrapper would not
fall through. (Residual uncertainty: `_affine_spec`'s full attribute walk
was not exhaustively traced; the Phase 0 harness settles engagement in
minutes.)

### C3. The fix is one collective, applied knowingly — MEDIUM effort (CONFIRMED mechanism)

Make `_backend` distributed-aware: when `mlp.down_proj` is a sharded-to-all
variant, apply `mx.distributed.all_sum(out, group=down_proj.group)` before
returning — exactly what the module's own `__call__` does. One collective
per MLP call, identical count on every rank (the wrapper is class-level, so
it engages symmetrically given identical shapes). Rank capability asymmetry
(one Mac has ANE headroom, the other doesn't — probes via `has_symbol`,
`_ane_headroom_ok`, per-module failure latching are all rank-local) is then
only a *timing* skew at the collective, not a correctness issue: each rank's
MLP partial is correct regardless of which backend computed it. GDN in-proj
acceleration is collective-free by construction (in_proj is all-to-sharded;
the collective lives in `out_proj`) and needs no equivalent fix.

### C4. Chunk-width interlock: the cluster's memory-constrained profile would compile-but-never-execute — LOW (CONFIRMED)

The ANE tiles chunks *wider* than the compiled `sequence_length` and never
executes on narrower ones (`configure_qwen35_ane_prefill_scheduler`,
`qwen35_ane_prefill.py:326`: "Chunks narrower than the compiled shape cannot
tile onto it"). Cluster `prefill_step_size` defaults: 1024 / 2048 / 4096 by
profile (`cluster/performance.py:260,268,276`) vs the ANE default
`sequence_length=2048`. Worker enablement must validate
`prefill_step_size >= sequence_length` and refuse (or clamp the ANE shape)
otherwise, with a log line — the single-node path already warns on the
analogous mismatch.

### C5. Bank compilation on sharded weights needs real hardware — MEDIUM (empirical; cannot be settled from code)

Compiling procedure banks from half-size shards should *help* the ~4 GiB
per-ANE device-address window (`_ANE_BANK_RETRY_MAX_BYTES` commentary,
`qwen35_ane_prefill.py:37-58`), but the private ANE runtime's acceptance of
the halved shapes, driver memory behavior under two ranks, and `dual_ane`
bank pinning are not derivable from code. Note the localhost-harness caveat:
two ranks on one Mac **contend for the same physical ANEs**, so the
coordinator-local harness validates *correctness* (logit parity with the C3
fix) but not the two-machine performance or per-chip bank behavior — that
part waits for the peer.

---

## 5. Theme D — MTP under cluster

MTP is the newly-investigated theme. Verdict up front: **not fundamentally
incompatible with TP, but unlike TurboQuant (clean) and ANE (one collective
fix), MTP's control plane is rank-local-nondeterministic *by design* and
must be re-architected for lockstep before it can run distributed.** It is
the highest-effort and highest-risk of the three.

**Prior art:** jundot/omlx#2970 (`feat/mtp-tensor-parallel-clustering`, reopened)
already implements the "rank 0 decides, everyone follows" fix shape described
in D1 below — a rank-0-broadcast-with-checksum design for the per-cycle
draft-depth decision — and it was hardware-validated on the two-node
cluster. It predates this doc; when acting on Theme D, start from that PR's
diff rather than re-deriving the broadcast design from scratch.

### D1. The adaptive depth controller is wall-clock-driven per machine — designed rank divergence — HIGH (CONFIRMED)

`_DepthController` (`patches/mlx_lm_mtp/batch_generator.py:1825`) selects
draft depth per cycle from an EMA of **measured cycle cost on this machine**
— its own docstring: "Everything the decision uses is measured on this
machine, on this model, under the current load." Wall-clock probe cadence
(`PROBE_PERIOD_MS`), spike damping, staleness-directed probes, depth-0
parking, and the `EXIT_STREAK` hand-off to the standard decoder
(`_park_mtp_to_standard`) are all driven by `time.perf_counter()` /
`time.monotonic()` (`:208,1793,2927-3292`). Under lockstep TP, each rank
running its own controller **will** choose different depths / park at
different times — different numbers of forward passes, different collective
counts, immediate desync (companion doc A1's failure class, but here as
steady-state behavior, not an edge case). Even a fixed
`mtp_num_draft_tokens=1` does not neutralize this: depth-0 parking, the
loop-tax probe (`_omlx_mtp_loop_tax`, `:1793`), and re-entry probes remain
wall-clock-driven.

The fix shape: **rank 0 decides, everyone follows.** All ranks run the
backbone/verify forwards (they must — the model is sharded); rank 0 runs the
controller, the sampler, and the accept/reject comparison, then broadcasts
`(depth, accepted_count, emitted token IDs, park/exit decisions)` per cycle
— token IDs included, because the next forward's inputs depend on them;
leaving followers to re-derive tokens from their own logits would reinstate
the bitwise-identical-`all_sum` dependency this design removes. Cost context: a
TP=2 decode forward already fires two `all_sum`s per layer (dozens per
step), so one additional tiny per-cycle broadcast is noise; the pinned
server already broadcasts request objects over the collective
(`_share_object`, `site-packages/mlx_lm/server.py:485-502`), giving a
precedent mechanism. This also removes MTP's dependence on bitwise-identical
`all_sum` outputs across ranks for *control flow* (companion doc A3 — a
stated, only partially-evidenced assumption), leaving it only where the
non-MTP path already carries it.

### D2. Rank-divergent fallback latching — HIGH (CONFIRMED raise sites; desync endgame per companion A1)

The MTP loop is deliberately fallback-rich: `_MtpStepFallback` is raised
from twelve sites (`batch_generator.py:1098-3396`), including at least one
purely rank-local capability failure ("embedded DSpark host is
unavailable", `:2246`), and per-module failure latching elsewhere in the
speculative stack degrades gracefully by falling back to standard decode.
Single-node, graceful degradation is a feature. Under lockstep TP, *any*
fallback that fires on one rank and not the other changes that rank's
forward count and wedges the collective. The port must convert every
fallback into either (a) a synchronized decision (rank 0 decides, broadcast,
all ranks fall back together) or (b) a pre-launch capability check that
disables MTP uniformly before the first cycle. An inventory pass over all
twelve raise sites classifying each as deterministic-given-identical-state
vs. rank-local is a required design task, not optional hardening.

### D3. The MTP head would be an unsharded per-rank replica — workable at TP=2 — MEDIUM (CONFIRMED absence; parity PLAUSIBLE pending harness)

No TP adapter touches the head: zero references to `mtp` in
`cluster/tensor_strategies.py`, and native `shard()` methods iterate
`self.layers` only. Under TP the head stays fully replicated per rank. That
is *correct by construction* for TP: inter-layer activations are full-width
and identical on every rank (each sharded layer ends in `all_sum`), so a
replicated head computing on replicated hidden states produces identical
draft logits everywhere — at the cost of redundant compute and one head's
worth of memory per rank (~one layer; acceptable at TP=2). The patched
`Model.__call__` returning hidden states alongside logits needs no change
for TP. Under **pipeline parallelism** none of this holds (hidden states
exist only on the last stage; head placement, cross-rank hidden-state
plumbing, and rollback coordination all become real work) — PP-MTP is
explicitly out of scope (§ Explicitly not doing).

### D4. Cache rollback is rank-local and lockstep-compatible — LOW (CONFIRMED mechanism)

Rollback (`patches/mlx_lm_mtp/cache_rollback.py`) is per-cache-object trim /
undo-log replay — no cross-rank state. Given D1/D2's synchronized
`accepted_count`, every rank trims its rank-local caches by the same amount
and stays aligned. The rotating-cache undo log and `ArraysCache` GDN
snapshot are cache-class-level and indifferent to sharding. No additional
distributed work beyond the control-plane sync.

### D5. The activation seam already exists in the worker — LOW (CONFIRMED)

Because `run_worker` already calls `maybe_apply_pre_load_patches`
(A2), passing `mtp_enabled` + depth through it attaches the head and
installs the `BatchGenerator` patches **inside the exact generation loop the
worker runs** (the patch was designed to fold into mlx-lm's
`GenerationBatch.next()`, `patches/mlx_lm_mtp/__init__.py` module
docstring). The mechanical port is small; D1/D2 are the actual work. Note
also that the deployed cluster model (`nemotron_h`) is depth-1 MTP-capable —
the smallest possible speculative configuration, and the right first target.

---

## 6. Phased implementation checklist

Ordering rationale: TurboQuant first (clean math, bounded port, graceful
fallback), ANE second (one known corruption fix plus hardware validation),
MTP last (control-plane redesign, gated on determinism evidence and on the
companion doc's Theme A hardening — a desync bug shipped here wedges the
cluster in exactly the way A1 describes). All Phase 0 items are
coordinator-local; **do not target the peer node until it is back on the
network.**

### Phase 0 — Gate hygiene + local determinism harness (unblocks everything)

- [ ] 0.1 Add `qwen35_ane_prefill_enabled` to `_validate_model_settings`
      (`engine/distributed.py:262`) so ANE profiles fail loud like the other
      flags, with a test alongside the existing gate tests. (A3)
- [ ] 0.2 Build the coordinator-local harness: 2-process `mlx.launch` ring
      over localhost, TP=2, small `qwen3_5` (dense) and `nemotron_h`
      checkpoints; assert token-level parity vs single-node greedy decode.
      This is the acceptance gate for every later phase.
- [ ] 0.3 In the harness, measure whether `all_sum` outputs are bitwise
      identical across ranks over long decodes (companion doc A3). Result
      determines whether Phase 3 may rely on identical-logits control flow
      anywhere or must broadcast every decision.
- [ ] 0.4 Plumb a generic `--model-settings-subset` channel (argv flags per
      A1's uniform-value rule) from `ClusterDeployment` →
      `build_mlx_launch_argv` → worker argparse, empty by default. No
      behavior change; later phases populate it.

### Phase 1 — TurboQuant KV under cluster (gate on 0.2, 0.4)

- [ ] 1.1 Extract `_turboquant_eligible`'s cache-layout logic
      (`scheduler.py:3146`) into a shared helper usable without a Scheduler.
- [ ] 1.2 Worker activation: parse TQ flags; call
      `apply_turboquant_attention_patch()` in `run_worker`; wrap the pinned
      server's `make_prompt_cache` call sites to convert eligible layers
      (bits + `skip_last` keyed on **global** layer index via
      `assignment.start_layer/end_layer`, B4).
- [ ] 1.3 Harness: logit parity sharded-vs-single-node with TQ on, at
      matching bits; long-context decode to exercise the quantized decode
      kernels per rank.
- [ ] 1.4 Empirically resolve B3 in the harness: does batched serving
      survive cache conversion, or does `is_batchable` drop to sequential?
      If sequential: measure the throughput cost at TP=2 and decide whether
      merge-protocol glue for `BatchTurboQuantKVCache` is worth writing
      (separate item, do not block the port on it).
- [ ] 1.5 Only after 1.3 passes: remove `turboquant_kv_enabled` from the
      gate, replacing it with the runtime eligibility downgrade (MLA/sinks
      models keep failing loud or downgrade with a logged reason — match
      single-node behavior).
- [ ] 1.6 Peer-online validation (blocked on peer): repeat 1.3 on the real
      2-Mac ring/jaccl transports before enabling in any saved profile.

### Phase 2 — ANE-prefill under TP (gate on 0.2; 2.5 blocked on peer)

- [ ] 2.1 Fix C1 before any enablement plumbing: teach `_backend` /
      `_backend_exact` / `_post_ane_down` / `_tail_qmm_or_linear` to detect a
      sharded `down_proj` and `all_sum` the result over `down_proj.group`
      (single exit-point collective; audit that *every* return path is
      covered, including the fused-down kernels and the long-tail qmm).
- [ ] 2.2 Confirm C2 in the harness: post-shard `_eligible_pair` /
      `_affine_spec` engagement on a sharded quantized checkpoint (minutes;
      settles the one PLAUSIBLE link in the corruption chain).
- [ ] 2.3 Worker activation: plumb ANE settings; call
      `enable_qwen35_ane_prefill` after shard+load; enforce the C4 interlock
      (`prefill_step_size >= sequence_length`, refuse with a logged reason).
- [ ] 2.4 Harness correctness run: logit parity with ANE engaged on both
      localhost ranks (accepting ANE contention on one Mac — correctness
      only, not performance).
- [ ] 2.5 Peer-online validation (blocked on peer): per-chip bank compile on
      halved shards, `dual_ane` behavior on each Mac's chip, asymmetric-
      capability run (one rank ANE, one GPU-only) confirming C3's
      timing-only claim, and real prefill throughput vs GPU-only TP.

### Phase 3 — MTP under TP (gate on 0.2, 0.3, and companion doc Phase 1/2 lockstep hardening)

- [ ] 3.1 Design pass: inventory all twelve `_MtpStepFallback` raise sites
      plus every wall-clock decision point (D1's list) and classify each as
      deterministic-given-identical-state vs rank-local. Output: the
      synchronized-decision protocol spec (what rank 0 broadcasts per cycle,
      and the uniform pre-launch capability checks that replace rank-local
      latching).
- [ ] 3.2 Implement rank-0-decides: controller, sampler, accept/reject on
      rank 0; per-cycle broadcast of `(depth, accepted_count, park/exit)`
      piggybacked on the existing collective cadence — the payload includes
      the emitted token IDs, not just counts (D1); all other ranks follow
      decisions verbatim. Loop-tax and re-entry probes run on rank 0 only.
- [ ] 3.3 Worker activation: pass `mtp_enabled`/depth through the existing
      `maybe_apply_pre_load_patches` call (D5); start with `nemotron_h`
      fixed depth-1 (smallest surface, matches the deployed model).
- [ ] 3.4 Harness: long greedy decodes with fault injection — force a
      fallback on rank 1 only and assert the deployment fails loud (A1-style
      detection) instead of wedging; parity vs single-node MTP output.
- [ ] 3.5 Only after 3.4: remove `mtp_enabled` from the gate for TP
      deployments (keep it gated for PP — see below); adaptive depth
      (controller on rank 0) as a follow-up once fixed depth-1 has soaked.
- [ ] 3.6 Peer-online validation (blocked on peer): real-transport soak;
      measure whether the per-cycle broadcast is visible in tok/s on ring
      vs jaccl.

### Explicitly not doing (this design)

- **MTP under pipeline parallelism** — head placement on the last stage,
  cross-rank hidden-state plumbing, and PP rollback coordination are a
  separate design (D3). The gate keeps `mtp_enabled` blocked for PP plans.
- **`dflash_enabled`, `specprefill_enabled`, `vlm_mtp_enabled`** — the other
  gate entries; separate draft-model/VLM architectures, not investigated
  here, stay gated.
- **TurboQuant + MTP combined under cluster** — the multi-row verify
  kernels interact with TQ caches on the single-node path; validate each
  feature under TP independently before stacking them.
- **VLM cluster deployments** for any of the three (`engine/vlm.py` paths).
- **Merge-protocol glue for `BatchTurboQuantKVCache`** unless 1.4 shows the
  sequential fallback costs enough at TP=2 to justify it.
- **ANE-prefill under PP** — untested interaction between stage-local layer
  ownership and the class-level MLP wrapper; TP-only until someone needs it.

---

## 7. Verified non-issues — do not re-investigate

- **TurboQuant quantization-group / TP-shard interaction**: groups run along
  `head_dim`, shards split heads; no group ever crosses a rank (B1).
- **TurboQuant seeds across ranks**: each rank quantizes only its own KV;
  cross-rank seed agreement is not required.
- **ANE capability asymmetry between unlike Macs**: rank-local probes
  degrade independently; with C1 fixed this is timing skew only (C3).
- **MTP cache rollback distribution**: rank-local trim with a synchronized
  count needs no distributed machinery (D4).
- **Patch import-order for the worker's SDPA rebinding**: the TurboQuant
  patch rebinds already-imported model modules by scanning `sys.modules`
  (B2), so `run_worker` call placement is not delicate.
