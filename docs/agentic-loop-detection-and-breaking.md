# Agentic loop detection and breaking

Design doc + phased implementation checklist for a loop-detection/loop-breaking
subsystem targeting sentence-level and tool-call-level repetition in long
agentic generations. Every file:line reference below was **verified against
HEAD `cb281726` (branch `deploy/session-fixes-v2`) on 2026-08-26**. Line
numbers will drift as the tree moves — treat them as anchors (the quoted
identifiers are the stable handles), and re-locate rather than trust a stale
number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here). References into the
**pinned runtime packages** are written `site-packages/mlx_lm/...` (under
`.venv/lib/python3.11/site-packages/`) — those are not repo files, and a pin
bump invalidates them wholesale.

Findings are labeled **CONFIRMED** (the full causal chain was traced in code)
or hedged (**PLAUSIBLE** / **empirical**) where a claim is a judgment call or
needs measurement. Where a claim genuinely cannot be settled from code, the
required empirical test is named instead of guessed at.

Status: **scoping ahead of need.** No measurement yet shows that
period->20-token loops occur at meaningful rates on the current stack
(Qwen3.8-27B, TurboQuant KV 6-bit, ANE prefill, Lightning MTP,
reasoning_effort=medium). Phase 0/1 below exists precisely to settle that
before any enforcement is built.

---

## 1. Context

### The gap: presence_penalty is a 20-token stutter guard, not a loop breaker

The sampling default `presence_penalty=0.4` was set on the 27B agentic
profile to counter repetition/looping in long tool-call chains. Its actual
semantics under oMLX are much narrower than the OpenAI option of the same
name: penalties are built via `mlx_lm.sample_utils.make_logits_processors`
(imported at `scheduler.py:48`, applied in `_create_batch_generator`
`scheduler.py:2916` and `_build_sampler_and_processors` `scheduler.py:5701`),
and `make_presence_penalty` (`site-packages/mlx_lm/sample_utils.py:315`)
subtracts the penalty only from tokens present in the **last
`presence_context_size` tokens — default 20** (`sample_utils.py:77`, window
slice at `:334`). oMLX forwards only `repetition_context_size`
(`scheduler.py:2946-2950`); the presence window stays at the hardcoded 20.

Consequences, both directions:

- Repeated tool-call JSON scaffolding across a long generation is **never**
  penalized (the window can't see the previous call) — the feared
  corruption of legitimate repeated structure is a non-issue.
- Any loop with period &gt; ~20 tokens — a repeated sentence of thinking, a
  repeated whole tool call — is **invisible** to the penalty. The configured
  mitigation is a placebo for exactly the failure mode it was set for.

That gap is what this subsystem fills: detection (and, conditionally,
breaking) of sentence-level and tool-call-level repetition *within one
generation*.

### A free co-signal: MTP acceptance rate

Degenerate repetition is maximally predictable, so the MTP head drafts loops
near-perfectly and acceptance spikes — meaning speculative decoding makes a
spiraling model burn tokens *faster*, and simultaneously hands us a
zero-added-cost loop smell. Acceptance accounting already exists
(`acceptance_rate = total_accepted / (rounds * max_per_round)`,
`scheduler.py:8477`, logged via the MTP stats path). A sustained abnormal
acceptance run over a window is a cheap corroborating signal for any
detector, and independently worth surfacing as telemetry.

### Design posture

Telemetry first, enforcement later, heavy machinery never (see § Explicitly
not doing). The phased checklist gates every intervention phase on data from
the phase before it. Config follows the established opt-in structured-config
pattern (per-model settings in `model_settings.py`, off by default).

---

## 2. Theme A — Detection

### A1. The scheduler emission loop is the natural detection site, and precedent already lives there — HIGH (CONFIRMED)

The per-committed-token processing in `Scheduler.step()`
(`scheduler.py:~10490-10545`) already does Python-level per-token work for
every request: protocol-parser models run
`parser_session.process_token(response.token)` (`:10491`), others run the
streaming detokenizer (`detokenizer.add_token`, `:10517`) **plus a
text-level stop-string tail scan** (`:10524-10545`) that exists precisely
because token-id matching misses BPE-boundary cases ("Text-level stop-string
fallback. Catches BPE edge cases where the tokenized stop sequence does not
match the model's actual output tokens"). That scan is the existence proof
for this design: text-level scanning at O(tail) per token in this loop is
already paid for and accepted. A detector added beside it inherits the same
per-request lifecycle pattern as `_request_detokenizers` /
`_output_parser_sessions` (`scheduler.py:2130-2139`, cleanup at
`:2863/:2898`).

### A2. Structural tool-call-repeat detection is near-free but only covers the rare in-generation case — MEDIUM (CONFIRMED mechanism; scope judgment)

`OutputParserSession` already segments output channel-aware — thinking vs
visible vs tool-call — token by token (`process_token` returning
`stream_text` / `visible_text` / stop semantics, consumed at
`scheduler.py:10491-10508`). Recording (tool_name, args-hash) at each parsed
call boundary and flagging N identical/near-identical calls in a row is a
dict lookup per call. Highest precision, lowest cost of any option.

Scope caveat (judgment, not code): most tool-call repetition in agentic
practice happens *across* turns — the harness re-issues a call after seeing
the same result — and the server never sees that as one generation.
In-generation structural detection only catches a model emitting the same
call repeatedly within a single response: real, but the rarer shape.
Cross-turn detection is the harness's job (§ Explicitly not doing).

### A3. Rolling-hash text n-grams are the right general-purpose layer — MEDIUM (CONFIRMED cost model; thresholds empirical)

Maintain a Rabin-Karp-style rolling hash over the detokenized stream, record
span hashes at sentence/newline boundaries in a small deque, and flag when
the same span hash recurs k times within a window. O(1) amortized per token
in Python — noise next to the detokenizer and parser work already in the
loop (A1). Hash **text, not token ids**, for the same reason the stop-string
fallback exists: token-id n-grams miss BPE-variant repeats of identical
text. Thresholds (span granularity, k, window) are empirical — Phase 0/1's
log-only mode is how they get tuned without user-visible risk.

### A4. Embedding-similarity detection is out — no-action (judgment; cost side CONFIRMED)

Fuzzy/paraphrase loop detection via chunk embeddings needs an extra encoder
forward per chunk in a pipeline already contending for one Metal stream
across TurboQuant dequant, MTP verify batches, and (during prefill) ANE
dispatch. Verbatim loops dominate the observed failure class; the marginal
recall on paraphrase-loops does not justify a second model in memory and
per-chunk latency. Research-grade for this codebase. (§ Explicitly not
doing.)

### A5. MTP acceptance-rate spike as corroborating telemetry — LOW (CONFIRMED signal exists; correlation empirical)

Per Context: the accounting already exists at `scheduler.py:8477`. Wiring a
sustained-high-acceptance window flag into the detector's telemetry is
near-zero cost. Whether the correlation is strong enough to gate
interventions on is an empirical question Phase 0/1 answers as a side
effect; do not build enforcement on this signal alone.

---

## 3. Theme B — Intervention

### B1. Abort with a distinct finish_reason is trivially buildable and composes with harness retries — HIGH (CONFIRMED)

The emission loop already flips `is_finished` / `finish_reason` inline (the
stop-string path sets `finish_reason = "stop"` mid-scan,
`scheduler.py:10537-10540`), and request-abort machinery exists
(`abort_request`, `scheduler.py:8812`; `_do_abort_request` `:8935`).
Emitting `finish_reason: "loop_detected"` with the partial result is a small
delta on existing paths. Crucially, the client's retry **is** the
resample-with-a-different-seed intervention, implemented at the request
boundary where streaming semantics stay clean and no server-side state is
needed. This is the only enforcement Phase 2 commits to.

### B2. A stateless logits-processor penalty spike is buildable, second priority — MEDIUM (CONFIRMED mechanism; channel-blindness is a real limit)

A processor added in `_build_sampler_and_processors` can implement targeted
suppression (e.g. penalize continuing an n-gram that has already repeated k
times), and processors may pre-scale logits arbitrarily before the static
per-request sampler — so temperature-spike-equivalents are expressible even
though samplers themselves are per-request-static. Two constraints:

1. Keep it **stateless-recomputable** from the token history the processor
   is handed each call (the pattern of `make_presence_penalty`,
   `sample_utils.py:332-336`), or it collides with the speculative rewind
   contract (D2).
2. Processors see `(token_ids, logits)` only — no channel information. They
   cannot distinguish tool-call JSON from thinking. Gating from the
   scheduler (which does know the channel via the parser session)
   reintroduces cross-component mutable state and, if stateful, the D2
   contract. Buildable, but only worth it if Phase 2 data shows aborts alone
   are too blunt.

### B3. Steering injection = abort + re-enqueue with prefix reuse; thinking-channel-only — MEDIUM (CONFIRMED mechanism; product-shape judgment)

`BatchGenerator` has no token-injection API (constructor surface:
`site-packages/mlx_lm/generate.py:1497-1521`); the honest implementation is:
abort the sequence, build a continuation prompt = original prompt + kept
output + steering text, re-enqueue, and lean on prefix/paged-cache reuse (the
boundary-snapshot machinery, `_extract_boundary_snapshot`
`scheduler.py:6504`, exists to make exactly such re-prefills cheap). Only
defensible **inside the thinking channel**: injected text in the visible
channel is either shown to the client or misrepresents what was generated.
Mid-size project; conditional Phase 3, gated on Phase 2 data (§ checklist).

### B4. Rollback-and-resample mid-generation is out — no-action (CONFIRMED blockers)

Two hard blockers, one soft one:

1. **Streaming**: tokens already sent cannot be unsent. Rollback works only
   for non-streaming or fully buffered responses; agentic clients stream.
2. **Hybrid-cache state**: `patches/mlx_lm_mtp/cache_rollback.py` exists
   because GDN/SSM layer caches are not arbitrarily trimmable — it keeps a
   **one-step** undo log (`rollback_state` slot on `ArraysCache`,
   cache_rollback.py:2-21; trim wrapping `:85-140`) scoped to the MTP
   draft/verify cycle. Arbitrary N-token rollback on a Qwen3.5-family
   hybrid requires restoring a full block-aligned boundary snapshot — which
   the snapshot path itself sometimes *skips* under speculative skew
   (`_extract_boundary_snapshot` docstring, `scheduler.py:6511-6521`).
3. Even where possible, it duplicates what B1 + harness retry achieves at a
   fraction of the complexity. (§ Explicitly not doing.)

---

## 4. Theme C — Architecture: a two-layer split, not one component

### C1. Text/structural detection lives in the scheduler emission loop — HIGH (CONFIRMED)

It needs detokenized text and channel/tool-call boundaries, which logits
processors fundamentally do not have (B2.2). Placement: beside the
stop-string fallback and parser-session consumption (A1), with per-request
detector state managed like `_request_detokenizers` /
`_output_parser_sessions`. Actions available at this layer: metrics/log
(Phase 0/1), finish-reason abort (Phase 2), abort-and-re-enqueue steering
(Phase 3).

### C2. Token-level suppression, if ever built, is a logits processor — MEDIUM (CONFIRMED seam)

`_build_sampler_and_processors` (`scheduler.py:5701`) is the per-request
seam; the processor must be stateless-recomputable (B2.1, D2). This layer is
optional and gated — the split exists so that neither layer is forced to do
the other's job badly.

### C3. Config shape — LOW (convention)

Opt-in, per-model, structured: detector granularities (sentence /
tool-call), thresholds (k, window), and `action: log | abort` in
`model_settings.py`, default absent/off, following the same
structured-opt-in pattern as the eviction rework. Telemetry lands in the
existing stats path so `rtk`-side and admin-UI consumers get it for free.

---

## 5. Theme D — Speculative decoding interactions

### D1. Lightning MTP (the 27B's path) applies logits processors correctly in lock-step — no-action for stateless processors (CONFIRMED)

`patches/mlx_lm_mtp/batch_generator.py` maintains
`GenerationBatch._token_context[0]` (a `TokenBuffer`) "in lock-step with
each forward-input position so that `logits_processors` see the same token
sequence the standard step would see", shrinking it by one on reject
(module docstring, batch_generator.py:62-67). Stateless processors — the
stock penalties, and any B2-style suppressor built to the same pattern —
ride the MTP fast path unchanged. This corrects an earlier assumption that
penalties might force the text-model path off speculation: they do not.

### D2. Stateful processors must satisfy the snapshot/restore rewind contract or lose speculation on the VLM path — MEDIUM (CONFIRMED)

The vlm_mtp bypass path is stricter: `_route_to_vlm_mtp`
(`scheduler.py:8198`) threads only processors implementing
`snapshot_state()` / `restore_state()` (`supports_vlm_mtp_processing`,
`speculative/processing_sampler.py:66`; position-keyed checkpoints
`:201-205`) into the positioned `sample_target` hook. Any request carrying a
processor without that contract **silently falls back to BatchGenerator**
(`scheduler.py:8233-8254`, logged at info) — "same convention as Lightning
MTP: fall back … so every processor stays enforced" — costing speculation,
not correctness. Design rule that follows: any new intervention processor is
either stateless-recomputable (preferred) or implements snapshot/restore;
never silently stateful.

### D3. Detection lag under MTP is bounded by one draft block and is acceptable — LOW (CONFIRMED)

Speculative decode "advances the cache in bursts and emits from a queue"
(`_extract_boundary_snapshot` docstring, `scheduler.py:6511-6517`), so
scheduler-side detection sees committed tokens up to a block late, and any
intervention lands a few tokens after the detection point. Irrelevant at
sentence/tool-call granularity.

### D4. Loops accelerate under MTP — raises the value of early abort — LOW (CONFIRMED mechanism)

Per Context: loop text drafts perfectly, acceptance spikes, wasted tokens
per wall-clock second go *up* relative to non-speculative decode. This is
both the argument for Phase 2's abort (stop paying for garbage sooner) and
the source of the A5 telemetry signal.

---

## 6. Phased implementation checklist

Ordering rationale: detection-as-telemetry first because no measurement yet
shows the failure mode is real at meaningful rates on this stack — the
detector is also the one component every later phase needs. Enforcement
phases are each gated on data from the phase before. **Phase 3 is
conditional, not committed**: it exists only if Phase 2 data shows aborts
wasting substantial completed work.

### Phase 0/1 — Detector + telemetry, log-only (no behavior change)

- [ ] 1.1 Rolling-hash sentence-level repeat detector in the `step()`
      emission loop beside the stop-string fallback (A1, A3): rolling hash
      over detokenized text, span hashes at sentence/newline boundaries,
      flag on k repeats within a window. Per-request state managed like
      `_request_detokenizers`.
- [ ] 1.2 Tool-call-repeat counter at parser-session call boundaries (A2):
      (tool_name, args-hash) history per request, flag N consecutive
      identical/near-identical calls.
- [ ] 1.3 Opt-in per-model config (C3): granularities, thresholds,
      `action: log`, default off. Structured settings in
      `model_settings.py` following the eviction-rework opt-in pattern.
- [ ] 1.4 Telemetry: loop-event counters (granularity, span length, repeat
      count, request id, channel) into the stats path; include the MTP
      acceptance-rate window flag (A5) as a corroborating field.
- [ ] 1.5 Soak on the 27B agentic profile; tune k/window from logged events.
      **Exit criterion: data showing whether period->20 loops occur at
      meaningful rates.** If they do not, stop here — the doc's premise is
      then disproven and Phases 2/3 are unnecessary.

### Phase 2 — Abort enforcement (gate on 1.5 showing real loop events)

- [ ] 2.1 `action: abort` config value: on detector fire, finish the request
      inline (the stop-string pattern, `scheduler.py:10537`) with
      `finish_reason: "loop_detected"` and the partial result (B1).
- [ ] 2.2 Verify clean teardown parity with the stop path: detokenizer /
      parser-session / detector cleanup, MTP-active abort (detection lag D3
      means the cache may be a burst ahead — reuse the existing abort
      machinery, `scheduler.py:8812`, which already handles in-flight
      speculative state).
- [ ] 2.3 Document the harness contract: clients treat `loop_detected` as
      retryable; the retry is the resample (B1). No server-side retry.
- [ ] 2.4 Measure: fraction of aborts where substantial completed work
      (long prefix before the loop) is discarded. **This is the Phase 3
      gate.**

### Phase 3 — Thinking-channel steering injection (CONDITIONAL; gate on 2.4 showing wasted-work cost)

- [ ] 3.1 Only if 2.4 shows aborts discard substantial work at meaningful
      rates: design pass for abort-and-re-enqueue (B3) — continuation
      prompt = original + kept output + steering text, prefix-cache reuse,
      thinking-channel-only injection, streaming semantics for the spliced
      continuation.
- [ ] 3.2 Implementation gated on that design pass; not scoped further here
      on purpose — Phase 2 data shapes it or kills it.

### Explicitly not doing (this design)

- **Embedding-similarity loop detection** (A4) — second model in memory +
  per-chunk encoder forwards on a contended Metal stream, for marginal
  recall on the rare paraphrase-loop shape.
- **Rollback-and-resample mid-generation** (B4) — blocked by streaming
  semantics and hybrid-cache (GDN) non-trimmability; superseded by
  abort + harness retry.
- **Cross-turn / cross-request loop detection** — the harness sees full
  request/response pairs and can inject corrective user-turn text; the
  server sees one generation at a time. Server-side scope is
  within-generation only (A2).
- **Dynamic sampler mutation mid-generation** — samplers are per-request
  static by design; anything of that shape is expressed as a logits
  processor under C2's constraints, or not at all.
- **Enforcement defaults** — every action above `log` stays opt-in
  per-model indefinitely; this subsystem must never fire on a workload
  whose owner didn't turn it on.

---

## 7. Verified non-issues — do not re-investigate

- **presence_penalty corrupting repeated tool-call scaffolding**: the
  20-token window (`sample_utils.py:334`) cannot see the previous call;
  within-window structural JSON tokens lose a flat 0.4 logit against
  margins that dwarf it (Context).
- **Penalties knocking the 27B off the MTP fast path**: Lightning MTP
  applies logits processors in TokenBuffer lock-step (D1,
  batch_generator.py:62-67). The silent fallback exists only on the
  vlm_mtp path for non-rewindable processors (D2) — irrelevant to text
  models, binding for any new stateful processor.
- **Detection cost in the emission loop**: per-token Python work
  (detokenizer, parser session, stop-string tail scan) is already the
  accepted cost floor there (A1); a rolling hash is O(1) amortized on top.
- **MTP cache rollback interfering with abort**: the undo log is scoped to
  the draft/verify cycle and abort machinery already handles in-flight
  speculative state (2.2); no new distributed/rollback work needed for
  Phase 2.
