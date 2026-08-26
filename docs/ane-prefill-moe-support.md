# ANE prefill on Qwen3.5-MoE models: corruption post-mortem and support plan

Design doc + phased implementation checklist for the 2026-08-26 incident:
enabling `qwen35_ane_prefill_enabled` on `Qwen3.6-35B-A3B-oQ4e-mtp`
(`config_model_type: qwen3_5_moe`) silently corrupted all prefills longer
than the compiled 2048-token shape, poisoned the SSD prefix cache, and was
blessed by the ANE tuner with an "18.4% speedup". Every file:line reference
below was **verified against HEAD `6107372a` (branch
`deploy/session-fixes-v3`) on 2026-08-26**. Line numbers drift; the quoted
identifiers are the stable handles.

Paths are relative to the repo root (`omlx/...` = the package). References
into the pinned runtime packages are written `site-packages/mlx_vlm/...`
(the serving tree runs against the framework-mlx-base site-packages).

Findings are labeled **CONFIRMED** (causal chain traced in code *and*, where
marked, reproduced empirically on this machine's ANE) or **PLAUSIBLE**
(mechanism traced, final behavior not fully isolated). The parity harness
used for the empirical results is described in Theme 2; it ran against the
real native extension (`custom_kernels/qwen35_prefill/_ext`) on 2026-08-26.

---

## 1. Context

`omlx/patches/qwen35_ane_prefill.py` (docstring, line 1) targets "dense
Qwen3.5/3.6/3.8 MLPs": it splits each MLP's gate/up projections across two
fixed-shape ANE programs plus a GPU suffix, and optionally does the same for
the GDN in-projections. It is enabled per-model from
`engine/vlm.py:1875-1949` and `engine/batched.py:397-410`, with
`gdn=True` by default (vlm.py:1915).

On the incident model the patch found *something* to grab — and what it
grabbed, how it went wrong, and what a correct MoE story looks like are the
four themes below. Headline conclusions:

- **The corruption is not an MoE math bug.** Module selection on the MoE is
  actually correct: only the dense per-layer *shared experts* were patched,
  and the compiled function is the right function. What breaks is a
  **synchronization race in the hybrid ANE/GPU native kernels** whose loss
  probability is a function of workload geometry. The MoE's tiny shared
  expert (512-row intermediate vs. many thousands on the dense models) makes
  the race lose on most evaluations; at dense-model geometry it almost never
  loses — which is why the same patch "works" on `Qwen3.8-27B-oQ4e-mtp`.
- Reproduced in isolation: at shared-expert geometry the GPU-computed half
  of the fused SwiGLU output is nondeterministic garbage (stale or
  uninitialized buffer contents, occasionally NaN) on most calls; the
  ANE-computed half is always correct.
- **The race is latent on dense models too** (observed one small
  nondeterministic deviation at dense-like geometry in ~6 repeats). Combined
  with SSD prefix-cache persistence this is a standing silent-corruption
  hazard independent of the MoE question.
- The FLOPs math says MoE *MLP* offload is ~2% of prefill compute — not
  worth implementing even correctly. The GDN offload (~15% of compute, and
  the likely source of most of the measured 18.4%) is structurally
  dims-generic and was numerically clean in isolation; it is the only part
  of ANE prefill worth re-enabling for MoE, and only after the race is
  fixed.

Incident-report correction for the record: the model has **256 routed
experts with 8 active per token** (`num_experts: 256`,
`num_experts_per_tok: 8` in the model's config.json), not 512/~10.

---

## 2. Themes

### Theme 1 — What the patch actually grabs on `qwen3_5_moe` (CONFIRMED, severity: context)

The eligibility walk (`enable_qwen35_ane_prefill`,
qwen35_ane_prefill.py:3196-3209) scans `model.modules()` for anything with
`gate_proj`/`up_proj`/`down_proj` attributes and passes it through
`_eligible_pair` (:359-381). On the MoE model three module kinds carry those
names:

1. **`switch_mlp` (the 256 routed experts, `SwitchGLU`)** — firmly rejected,
   *by design, not luck*: `_eligible_pair` → `_affine_spec` →
   `_eligible_affine_linear` (:193-217) requires
   `isinstance(linear, nn.QuantizedLinear)` (:196) and `weight.ndim == 2`
   (:201). The SwitchGLU projections are `QuantizedSwitchLinear` with 3-D
   `[256, out, packed_in]` weights and fail both tests.
2. **The per-layer `shared_expert`** — accepted, and *correctly* so from the
   walk's point of view: `site-packages/mlx_vlm/models/qwen3_5_moe/language.py:10`
   imports `Qwen3_5MLP as Qwen3_5MoeMLP`, so the shared expert **is the very
   class** (`Qwen3_5MLP`) that the dispatch hook wraps at class level
   (`_wrap_class`, :2048-2061; installed on `mlx_vlm...Qwen3_5MLP` and
   `mlx_lm...MLP` in `_install_dispatch`, :2064-2112). It is a plain dense
   SwiGLU MLP (hidden 2048 → intermediate 512), 8-bit/group-128 affine in
   this checkpoint, and passes every `_eligible_pair` shape identity. 40
   layer shared experts + 1 MTP-layer shared expert get compiled and
   dispatched.
3. **The GDN in-projections** (30 of 40 layers are `linear_attn`) — the MoE
   reuses the dense model's `Qwen3_5GatedDeltaNet` class unchanged
   (qwen3_5_moe/language.py:8), so `_eligible_gdn` (:1012-1025) accepts them
   exactly as on the dense models. Dims here: qkv 8192 rows, z 4096 rows,
   b/a 32 rows each, hidden 2048.

The `shared_expert_gate`, router `gate`, and attention projections are never
touched. The dispatched computation for a patched shared expert —
`down(swiglu(gate(x), up(x)))` over a 2048-token tile — is algebraically
identical to `Qwen3_5MLP.__call__`
(site-packages/mlx_vlm/models/qwen3_5/language.py:1520-1531), and the MoE
block then scales and sums it correctly (qwen3_5_moe/language.py:64-73).

**So the "wrong assumption" is not in module selection.** The wrong
assumption is in the native kernels the selected modules were fed to —
Theme 2.

### Theme 2 — The corruption mechanism: a workload-size-dependent race in the hybrid ANE/GPU kernels (CONFIRMED empirically, severity: critical)

Reproduced on this machine with a synthetic parity harness (scratchpad
scripts `ane_moe_parity*.py`, 2026-08-26): a bare `nn.Module` with three
quantized linears at exact shared-expert geometry (hidden 2048,
intermediate 512, seq 2048), driven through the production entry points
(`_backend_exact`/`_compile_pair`, and the raw
`fast.qwen35_ane_dual_affine_swiglu_t` call), compared against the stock
quantized GPU path. Findings, in causal order:

- **The ANE-computed output channels are always correct** (error ≈ 0.35%
  mean relative — the expected per-row-int8 requantization noise; see
  Theme 5). Verified across seeds, bit-widths (4/gs64 and 8/gs128), dual
  and single ANE, and program sizes from 128 to 4224 rows. A separate
  probe of bare `qwen35_ane_compile_linear` programs at 128–4096 rows was
  also clean, so small ANE programs per se are fine.
- **The GPU-suffix channels are nondeterministically wrong.** Repeating the
  identical call (same weights, same input, same compiled state) flips
  between correct (mean err ≈ 0.06% — quantization noise) and garbage
  (mean err ≈ comparable to the signal itself; in extended runs the
  region degraded to NaN). Three repeats at seed 5/7/13 each produced a
  different mix of correct and corrupt evaluations.
- **Loss probability scales with how small the workload is.** Sweep at
  4-bit/gs64, hidden 2048, dual ANE, through the full production wrapper:
  intermediate 512 → 145% mean relative error; 1024 → 106%; 2048 → 43%;
  4096 → 3.6% (clean); 8192 → 3.9% (clean). Dense-model MLPs live at the
  clean end; the MoE shared expert lives at the catastrophic end.
- **The window does not fully close at dense geometry**: one of six
  repeats at intermediate 4096 differed from its siblings by up to 1.3
  absolute in a handful of elements. The dense path is exposed to the same
  race at low probability (see Theme 6).

Where the race lives (code-level, PLAUSIBLE as to the exact missing
barrier): `DualAneHybridPrimitive::eval_gpu`
(`custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm:2610-2887`; the
single-ANE `AneHybridQ4Primitive`, :2268+, has the same structure and the
same empirical failure). The sequence is: pack kernel → commit + host wait
(:2693-2698); GPU qmm encoded and committed in **its own command buffer**
(:2723-2753); detached ANE evaluation threads; then the host waits on the
**ANE tickets only** — `model0_->wait(ticket0); model1_->wait(ticket1)`
(:2790-2792) — and immediately encodes the merge kernel, which reads
`gpu_output`, into a *later* command buffer (:2817-2860). In the
no-CPU-share path **nothing ever waits on the qmm command buffer** (the
only `[qmm_buffer waitUntilCompleted]` sits in the unwind path :2797-2801
and the CPU-share path :2806-2809). MLX buffers are hazard-untracked, so
cross-command-buffer ordering of the `gpu_output` write→read is not
guaranteed by Metal; when the merge runs early it reads whatever the
recycled allocation last held — stale activations or uninitialized memory.
That is exactly the observed signature: plausible-magnitude finite garbage
(sometimes NaN), confined to GPU channels, nondeterministic, no crash, no
speed anomaly. The profiling counters even record which side finishes last
(`kAneLast`/`kGpuLast`, :2810-2816) without acting on it.

Why the incident presented as it did: with 41 shared experts each ~50%
corrupted per 2048-token tile, the hidden state is destroyed in one
forward pass → degenerate logits → token-0 spam (verified against the
model's tokenizer: `"!"` is token id 0). Prompts under the 2048-token tile
never enter `_backend_exact` (`_backend`, :1994-2007), so short prompts
stayed clean.

Harness caveat: the repeat-loop scripts call the fused kernel back-to-back,
which likely hits the race *more* often than production's interleaved
per-layer workload; the conclusion does not rest on that, because the
corruption also reproduced through the unmodified production entry point
(`_backend_exact` via `_omlx_ane_prefill_config`, i.e. exactly what the
dispatch hook executes) in the first-round matrix. Throughput is unaffected — the work all
happens, just unsynchronized — so the tuner measured a genuine-looking
speedup (mostly from the GDN offload, per Theme 4's FLOPs split).

Why the dense 27B looks fine: at dense geometry the GPU qmm all but always
completes during the multi-millisecond ANE waits, so the merge reads
completed data. "Validated" there means "the race almost never loses",
not "the code is race-free".

### Theme 3 — The gate that let it through, and where the real gate must live (CONFIRMED, severity: high)

`QWEN35_ANE_CONFIG_PREFIXES = ['qwen3_5', 'qwen3_6', 'qwen3_8']` with
`startsWith` matching exists **only in the dashboard UI**
(`admin/static/js/dashboard.js:39`, used at :7952-7958 to decide whether to
*show* the toggle) — `qwen3_5_moe` prefix-matches `qwen3_5` and the toggle
appeared. The backend has **no model-type check at all**:
`enable_qwen35_ane_prefill` (qwen35_ane_prefill.py:3099) is purely
structural, and both call sites (`engine/vlm.py:1891`,
`engine/batched.py:410`) plus the tuner's settings writer
(`admin/ane_tuning.py:420`) forward whatever the settings say. Any fix that
only edits the JS prefix list leaves three unguarded backend entry points.

Correct gating, in order of robustness:

1. **Structural, at patch time, inside `enable_qwen35_ane_prefill`** so all
   callers inherit it: before accepting MLP candidates, collect
   `id(block.shared_expert)` for every module that has both `switch_mlp`
   and `shared_expert` attributes, and skip those ids in the candidate
   walk. Class identity cannot be used — `Qwen3_5MoeMLP` *is* `Qwen3_5MLP`
   (Theme 1) — and name-prefix matching on config strings is exactly what
   failed. Note the class-level dispatch hook is harmless once no shared
   expert carries `_omlx_ane_prefill_config` (`_backend`, :1989-1991
   returns None immediately).
2. **Declarative, at the call sites and dashboard**: replace prefix match
   with an equality allowlist of validated dense types
   (`{'qwen3_5', 'qwen3_6', 'qwen3_8'}` against the exact
   `config_model_type`), so a future `qwen3_9_moe`/`qwen3_5_vl_moe` string
   does not sail through either. Belt and suspenders with (1).

Until the Theme 2 race is fixed, the gate must exclude MoE models entirely
(including the GDN half, which shares the racy primitive even though its
geometry rarely loses).

### Theme 4 — Is a correct MoE offload worth building? FLOPs say: MLP no, GDN yes (CONFIRMED arithmetic, severity: decision)

Per token, per layer, in MACs, from the model's config.json (hidden 2048,
moe_inter 512, shared_inter 512, 256 experts top-8, GDN on 30/40 layers,
full attention on 10/40, head_dim 256, 16 q-heads, 2 kv-heads):

| Component | MACs/token (layer-avg) | Share |
|---|---|---|
| Routed experts (8 × 3 × 2048×512) | 25.2M | ~39% |
| GDN in-projections (qkv 16.8M + z 8.4M + b/a 0.13M, ×0.75) | 19.0M | ~29% |
| GDN out-proj + conv/recurrence (×0.75) | ~8.5M | ~13% |
| Attention proj + SDPA@2k (×0.25) | ~8.9M | ~14% |
| Shared expert (3 × 2048×512) | 3.15M | ~4.8% |
| Router + shared gate | 0.53M | ~0.8% |
| **Total** | **~65M** | |

- **What the incident config actually offloaded to ANE**: 53% of the shared
  expert's gate/up = **~1.7% of prefill compute**. Even a perfect
  implementation of shared-expert offload moves ≤5% of the work.
  **Verdict: correctly gateable, not worth implementing.**
- **GDN in-projection offload at fraction 0.50 ≈ 15% of compute**, running
  on an otherwise-idle engine concurrently with the GPU — this is where
  the tuner's 18.4% wall-clock gain plausibly came from, and it is already
  dims-generic and (in isolation, at these dims) numerically clean. This is
  the only MoE ANE surface worth shipping.
- **Routed experts are closed-form infeasible for fixed-shape ANE**: over a
  2048-token tile, the expected number of *unused* experts is
  256·(1−8/256)^2048 = 256·e^(−65) ≈ 10⁻²⁶ — the per-tile activated-expert
  union is all 256 experts, so a "union tile" ANE formulation degenerates
  to running the full 35B dense equivalent (~32× the useful expert work).
  The gather_qmm sparsity is intrinsically per-token and shape-dynamic; no
  fixed-shape reformulation recovers it.
- Attention projections (~3.6% at a 0.5 fraction) are a marginal future
  candidate at best, and are currently unpatched on dense models too.

### Theme 5 — The ANE tuner validates throughput, never correctness (CONFIRMED, severity: high, systemic)

`admin/ane_tuning.py::_measure_candidate` (:541+) generates real tokens per
candidate and computes `speedup_percent` purely from throughput ratios
(`_refresh_speedups`, :287-298). It is *carefully* defensive about
mismeasuring **speed** — it verifies ANE programs actually executed so a
silently-disabled path can't report GPU throughput as an ANE win
(`_ane_execution_observed` gate, :640-670) — but it never once compares
outputs against the unpatched path. During the incident run the tuner was
generating from ANE-corrupted prefills and blessing them (writes at :420).
A single greedy-decode comparison (or prefill logit-parity check) between
the baseline slot and each ANE slot, on the very tokens it already
generates, would have failed the candidate instantly. The
`quantize_rows` per-row-int8 conversion (qwen35_ane.mm:663-686) means exact
equality is not the bar; token-level agreement over a short greedy
continuation, or a logit MSE threshold, is.

### Theme 6 — Corrupted prefill persists: the SSD prefix cache has no integrity or provenance guard (CONFIRMED behavior, severity: high, systemic)

KV blocks produced by a corrupted prefill were stored by the paged SSD
prefix cache (`cache/paged_ssd_cache.py::PagedSSDCacheManager`, wired in
`scheduler.py:356` and `:1501-1517`, prefix matching in
`cache/prefix_cache.py::BlockAwarePrefixCache`) and replayed after ANE was
disabled — converting a transient numeric bug into a persistent one that
survives the fix. Two structural gaps:

1. **No provenance in the cache signature.** The cache signature
   distinguishes models (`cache_signature_for`, scheduler.py:1610-1620) but
   not the numeric configuration that produced the KV — an ANE-prefilled
   block is indistinguishable from a GPU-prefilled one. Folding the
   ANE-prefill settings (or any "experimental numeric path" flag) into the
   signature would have auto-invalidated the poisoned entries the moment
   ANE was toggled off.
2. **No integrity spot-check.** Nothing samples cached KV for sanity
   (NaN/Inf scan at write time is nearly free and would not have caught
   this specific finite-garbage case, but catches the NaN variants observed
   in the repro). Full correctness checking of cached KV is not practical;
   provenance-keying is the effective control.

Given Theme 2's residual race probability at *dense* geometry, this is not
just an MoE cleanup item: today a rare race loss on the validated dense
config could silently poison the cache too.

---

## 3. Explicitly not doing

- **Routed-expert ANE offload** in any form (union tiles, per-expert
  programs, top-k re-batching). Closed-form dead end per Theme 4.
- **Shared-expert ANE offload**, even after the race fix. ≤5% of compute;
  41 extra resident ANE programs and compile time for noise-level gains.
- **Fixing the Theme 2 race in this doc's scope beyond specification.** The
  fix itself (ordering the merge behind the qmm — an
  `MTLSharedEvent`/encodeWait, or a host `waitUntilCompleted` on the qmm
  buffer before encoding the merge) belongs to the native-kernel owner;
  compiled-kernel wrapper files are also explicitly off-limits for
  hot-deploy on this cluster.
- **Re-quantizing or re-fetching the incident model.** The checkpoint is
  fine; nothing here implicates the oQ4e weights.
- **Dashboard-only gating.** Editing `QWEN35_ANE_CONFIG_PREFIXES` alone is
  cosmetic (Theme 3).

## 4. Verified non-issues

- **`switch_mlp` was never patched** — rejected structurally
  (`_eligible_affine_linear`:196,:201), not by shape-arithmetic luck.
- **`shared_expert_gate`, the MoE router, and attention projections are
  untouched** by the patch on this model.
- **The GDN b/a projections (32 rows)** stay on the exact GPU path unless
  the 6-bit suffix packing engages (`_pack_affine_gdn_suffix`, :1035
  requires qkv bits == 6, which this checkpoint's layers mostly are not).
- **Decode and short-prompt paths never enter the ANE backend**
  (`_backend`, :1994-2007) — consistent with the observed "short prompts
  fine" symptom.
- **The bare ANE linear programs are correct at every size probed**
  (128–4224 output rows), including the per-row-int8 noise floor of
  ~0.35% mean relative error. Small ANE programs are not the problem.
- **`_target_verify` handling**: the MoE calls
  `shared_expert(x, target_verify)` positionally; the wrapper detects it
  (:187-190) and falls through to the original path during target-verify,
  so MTP verification was not additionally corrupted by dispatch-signature
  confusion.
- **The ANE compile cache is content-keyed** (qwen35_ane.mm:139) — no
  cross-model program collision was involved.

## 5. Phased implementation checklist

**Go/no-go criterion for Phases 2-3:** proceed only if (a) the Phase 1 race
fix lands and (b) the Phase 0 parity harness then shows the GDN path clean
at MoE dims across ≥100 repeated evaluations (it is clean today at its
favorable timing, but "clean" must be re-established once ordering is
explicit). If the native race cannot be fixed, stop at Phase 0: no ANE
prefill on MoE, ever, and consider narrowing dense enablement too.

### Phase 0 — Safety gate + tuner correctness check (cheap, immediate, ships regardless)

- [ ] Structural MoE refusal inside `enable_qwen35_ane_prefill`: collect
      `id(m.shared_expert)` for modules having both `switch_mlp` and
      `shared_expert`; exclude from MLP candidates. While the Theme 2 race
      is unfixed, also refuse GDN enablement when any such module exists
      (log one clear warning naming this doc).
- [ ] Exact-match allowlist (`config_model_type` equality, not prefix) at
      `engine/vlm.py:1875+`, `engine/batched.py:397+`, and in
      `dashboard.js` `isQwen35AnePrefillModel`.
- [ ] Tuner correctness slot: in `_measure_candidate`, run a short greedy
      continuation on the baseline (ANE-off) engine once, and compare each
      ANE candidate's greedy tokens over the same prompt; fail the
      candidate on divergence beyond a small edit threshold. Wire the same
      check as a startup self-test option (`_warm_ane_models` already
      exercises the path; it just never compares).
- [ ] Recreate the parity harness from Appendix A as a module under
      `benchmarks/` (the session scratchpad it was developed in is
      ephemeral) so the repeat-nondeterminism check is one command.
- [ ] Prefix-cache provenance: fold ANE-prefill-affecting settings into the
      SSD cache signature (`cache_signature_for` path, scheduler.py:1610),
      so toggling the numeric path invalidates stale KV automatically.

### Phase 1 — Fix the hybrid-kernel race (native, prerequisite for everything else)

- [ ] Order the merge kernel behind the GPU qmm in
      `DualAneHybridPrimitive::eval_gpu` and `AneHybridQ4Primitive`
      (qwen35_ane.mm): encodeWait on an event signaled by the qmm buffer,
      or host-wait on `qmm_buffer` before encoding the merge (the CPU-share
      path already does the latter, :2806-2809).
- [ ] Regression-test with the Phase 0 harness at *both* geometries: 100×
      repeats at intermediate 512 (must be bit-stable across repeats) and
      at 4096/8192 (closes the dense-model residual window from Theme 2).
- [ ] Re-validate the dense 27B deployment after the fix (it has been
      running with a low-probability version of this race).

### Phase 2 — GDN-only ANE prefill for MoE models (conditional on go/no-go)

- [ ] Add a mode where MLP candidates are skipped but GDN offload proceeds
      (today `gdn=True` still requires the MLP walk to run; the enable path
      already tolerates zero MLP candidates, :3211-3217 — verify the fused
      and bank paths do too).
- [ ] MoE-aware tuner grid: candidates with `mlp_fraction=None`,
      GDN-fraction sweep only; correctness slot from Phase 0 mandatory.
- [ ] Validate on `Qwen3.6-35B-A3B-oQ4e-mtp` with long-prompt greedy parity
      vs. ANE-off before blessing any config.

### Phase 3 — Optional/means-test (only if Phase 2 lands and shows real wins)

- [ ] Measure whether attention-projection offload (~3.6%) is worth a
      kernel path at all — likely no; write the number down and close.
- [ ] Revisit shared-expert offload only if a future MoE checkpoint has a
      large shared expert (≥25% of per-layer compute); for this family it
      stays out.

---

## Appendix A — Minimal race reproducer

Run with the repo venv (`.venv/bin/python`); requires the private ANE
runtime, so it must run on Apple silicon with the native extension built.
Corrupted repeats show `gpu_err` jumping from ~6e-4 (correct) to ~0.17+
or NaN, while `ane_err` stays constant; repeat-to-repeat max deltas are
nonzero on identical inputs. At `INTER = 4096` the same loop is almost
always stable — run more repeats to see the residual dense-geometry window.

```python
import sys
sys.path.insert(0, "/path/to/omlx-repo")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from omlx.patches.qwen35_ane_prefill import _AnePrefillConfig, _compile_pair
from omlx.custom_kernels.qwen35_prefill import fast

SEQ, HIDDEN, INTER, BITS, GS = 2048, 2048, 512, 8, 128
mx.random.seed(5)

def qlin(out_d, in_d):
    lin = nn.Linear(in_d, out_d, bias=False)
    lin.weight = (mx.random.normal((out_d, in_d)) * 0.02).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, group_size=GS, bits=BITS)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (
            qlin(INTER, HIDDEN), qlin(INTER, HIDDEN), qlin(HIDDEN, INTER))

mlp = MLP(); mx.eval(mlp.parameters())
cfg = _AnePrefillConfig(sequence_length=SEQ, fraction=0.53, variant=8,
                        dual_ane=True)
state = _compile_pair(mlp, cfg)
x = mx.random.normal((1, SEQ, HIDDEN)).astype(mx.bfloat16); mx.eval(x)
ref = swiglu(mlp.gate_proj(x), mlp.up_proj(x)).astype(mx.float32)
mx.eval(ref)
a = state.ane_outputs
outs = []
for rep in range(6):
    act = fast.qwen35_ane_dual_affine_swiglu_t(
        x, state.weight, state.scales, state.biases,
        state.model, state.model1, state.bits, cfg.variant, state.group_size)
    mx.eval(act)
    act = act.astype(mx.float32); outs.append(act)
    diff = mx.abs(act - ref)
    print(f"rep{rep}: ane_err={float(diff[..., :a].mean()):.4g} "
          f"gpu_err={float(diff[..., a:].mean()):.4g}")
for i in range(1, len(outs)):
    print(f"delta rep0->rep{i}: {float(mx.abs(outs[0] - outs[i]).max()):.4g}")
```

Full matrices used for the Theme 2 numbers (intermediate sweep, dual vs
single, bare-program probe, stale-buffer proof) were variations of this
harness: swap `INTER`/`BITS`/`GS`, set `dual_ane=False`, or route through
`_backend_exact` after setting `mlp._omlx_ane_prefill_config = cfg` to
exercise the exact production dispatch path.

