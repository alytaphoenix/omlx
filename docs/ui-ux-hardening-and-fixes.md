# UI/UX hardening and fixes plan

Design doc + phased implementation checklist from the UI/UX audit of oMLX's two
user-facing surfaces: the admin web dashboard (`omlx/admin/`) and the native
macOS menu-bar app (`apps/omlx-mac/`). Every file:line reference below was
**verified against HEAD `15c10d85` on 2026-08-24**. Line numbers will drift as
the tree moves — treat them as anchors (the quoted identifiers are the stable
handles), and re-locate rather than trust a stale number if a reference doesn't
land on the described code.

This is a **pure UI/UX audit** — backend kernel/memory/cache correctness is
covered separately by `docs/qwen35-hardening-and-optimization.md`; nothing here
overlaps that doc. Paths are relative to the repo root; the Python package is
nested one level (`omlx/omlx/...` on disk, written `omlx/...` here).

---

## 1. Context

Surface 1, the **admin dashboard**, is one monolithic Alpine.js component
(`function dashboard()` in `omlx/admin/static/js/dashboard.js`, ~12.3k lines)
bound in `templates/dashboard.html` and driving one Jinja partial per tab.
State freshness is timer-based per tab: stats every 500ms on Status, cluster
runtime every 2s (Bonjour discovery + full setup re-init every 5th tick = 10s),
download/quantize/upload queues every 2s while active, logs on a user
interval; timers start/stop in `handleMainTabChange` and on `visibilitychange`.
Benchmarks stream over SSE with replay-on-subscribe + client dedupe. There is
**no toast system**: feedback is a mix of inline `x-show` banners, raw
`alert()`/`confirm()`, and — in the cluster pairing flow — calls to a
`showNotification()` that does not exist (§A1). i18n: server-side Jinja `t()`
plus `window._t` (a **flat dotted-key** dict injected in
`templates/base.html:57`); `window.t` (base.html:58) **returns the key itself
on a miss** — never English, never `undefined`. The server merges the English
locale under the selected one at request time (`routes.py:1142-1156`
`_load_locale`). The chat playground (`templates/chat.html`, ~7.8k lines) is a
second, separate Alpine app with its own streaming/tool/markdown machinery.

Surface 2, the **macOS app**: `ServerProcess`
(`apps/omlx-mac/Sources/Server/ServerProcess.swift`) owns the `omlx serve`
child (state machine `stopped/starting/running/stopping/unresponsive/failed`,
5s `/health` poll, auto-restart) and posts NotificationCenter events consumed
by `MenubarController` (NSMenu + status icon + the only NSAlert surface) and
`AppServices` (an `@Observable` bridge republishing state to SwiftUI screens).
Each screen has its own VM polling on-appear via an unstructured `Task` loop
(1-5s) against a shared mutable `OMLXClient`; errors render as small red
footer text (`lastError`). Config truth is `<basePath>/settings.json`, owned
by the Python server while it runs.

Severity scale used throughout: **HIGH** = user sees wrong state, loses work,
or a critical action is dead; **MEDIUM** = confusing/erroneous but
recoverable; **LOW** = real but minor (a11y, i18n leaks, cosmetic-with-impact).

---

## 2. Theme A — Dashboard: dead code and broken markup

### A1. `showNotification` is never defined — every SSH-pairing action throws, three buttons wedge — HIGH

**What's wrong.** `dashboard.js` calls `this.showNotification(...)` 12 times
(lines 2328, 2336, 2339, 2361, 2364, 2385, 2393, 2396, 2411, 2413, 2417, 2420
— e.g. `this.showNotification('SSH keys exchanged successfully', 'success')`),
but no `showNotification` is defined anywhere under `omlx/admin/` (verified by
grep). Scenario: on the Cluster pairing panel the user clicks "Generate SSH
key" — the key **is** generated server-side, then the success call throws
`TypeError`; the catch block calls `showNotification` again and rethrows out
of the catch, and `this.clusterSshKeyGenerating = false` (line 2341) never
runs (no `finally`) — the button stays disabled until page reload. Same
missing-`finally` pattern wedges `generateKeyExchangeToken` (flag reset at
2366) and `storeClusterKeyInKeychain` (2422); `exchangeKeysWithPeer` has a
`finally`, so only its feedback is lost.

**Fix.** Implement a small toast/banner `showNotification(message, level)` in
the dashboard component (or route these 12 sites through the existing cluster
error banner), and move the busy-flag resets into `finally` blocks. Also note
the 12 messages are hardcoded English (fold into §D3).

### A2. Unclosed `<button>` tag: the Settings > Models copy-name button renders blank — HIGH

**What's wrong.** `templates/dashboard/_settings.html:1424-1429`: the opening
`<button ...>` ends at `:title="window.t('settings.models.table.copy_model_name_tooltip')"`
with **no closing `>`** before the `<svg x-show="!copied" ...>` on the next
line. The parser folds the first SVG into the button's attribute list, so the
copy icon never renders — the button is an invisible click target (only the
`x-cloak`-hidden checkmark ends up inside it). The correct sibling is
`_models.html:147-151`.

**Fix.** Add the missing `>` after the `:title` attribute. One character.

### A3. MTPLX import button keyed off an English substring of a backend message — LOW

`_modal_model_settings.html:686` shows the import button only when
`mtp_compatibility_reason.includes('MTPLX side-car')`; `routes.py:713-719`
documents the marker, but any rewording/localization of that backend string
silently kills the button. Fix: return a structured flag (e.g.
`mtplx_sidecar_available: true`) next to the human-readable reason.

---

## 3. Theme B — Dashboard: state desync and data loss

### B1. Chat "Delete message" silently destroys every later turn, persisted immediately — HIGH

**What's wrong.** `chat.html` `deleteMessage` (6741-6755) calls
`sliceBeforeMessage` (3068-3074): deleting a user message does
`messages.slice(0, index)`; deleting an assistant message truncates back to
the preceding user turn. Everything **after** the target — all later
user/assistant turns — is removed and `saveCurrentChat()` overwrites
localStorage on the spot. The menu item is labeled `t('chat.delete_message')`
("Delete message"), and unlike `deleteChat` (7031, `confirm(...)`) there is no
confirmation and no undo.

**Failure scenario.** User deletes the first reply of a 50-turn chat to clean
it up → 49 turns of work are gone, permanently.

**Fix.** Either delete only the targeted turn/variant, or relabel to "Delete
from here…" and gate it behind the same confirm `deleteChat` uses. (If the
truncation semantics are intentional for regeneration reasons, the label and
the confirm are the fix.)

### B2. Cluster worker roles silently revert to Headless within 10 seconds — HIGH

**What's wrong.** `dashboard.js:2027`: in `syncClusterNodesFromPeers()` worker
node objects are rebuilt with a hardcoded `role: 'headless'` (while the local
node preserves its role: `role: local.role || 'workstation'`, line 1966; and
`reserve_gib` **does** consult `previous`, 2024-2026 — so the omission is
role-specific). The role buttons in `_cluster.html:2410-2419` write
`node.role = role.key` onto `clusterPlanNodes`, and `refreshClusterExperience`
(1510-1521) re-runs `initializeClusterSetup({preview:false})` every 10s, which
calls `syncClusterNodesFromPeers()` whenever workers exist (2144-2146).

**Failure scenario.** User sets a worker to Workstation → within 10s the role
resets to Headless, its reserve disappears from the plan, and the
`nodesChanged` JSON comparison (2071-2082) invalidates the built plan. Related
history: the 0.6.2 role-reset/flapping fixes addressed the plan-invalidation
half (comment at ~2063); this residual is the role field itself. If worker
roles are *intended* to be headless-only, the per-node role buttons should be
disabled for workers instead.

**Fix.** Preserve `previous.role` for workers exactly as the local node does
(`role: previous.role || 'headless'`).

### B3. Cluster tab wipes the model inventory + fit catalogue every 10s, flapping the status pill to a false "Ready" — HIGH

**What's wrong.** `dashboard.js:1922-1923`: `syncClusterNodesFromPeers()`
unconditionally sets `this.clusterModelInventory = null;
this.clusterCatalogue = null;` even when nothing changed (the plan-invalidation
half of this was fixed — comment at ~2063 — but the cache wipe stayed). Each
10s cycle then re-runs `loadClusterModelInventory()` (2159-2161 — a POST that
SSHes into every worker; "Reading models from every worker…" flashes,
`_cluster.html:646-650`) and `loadClusterCatalogue()` (2206-2210 — the server
re-plans every model). While `clusterCatalogue` is null,
`clusterCatalogueFit()` returns nothing, so a selected model that does **not**
fit loses its amber "Does not fit" state and `clusterQuickStatus()` falls
through to `{key:'ready', label:'Ready', ...}` (3745-3768).

**Failure scenario.** User selects a model that doesn't fit → every 10s the
status pill flips green "Ready" for the duration of the refetch, then back to
amber; meanwhile workers get SSHed continuously.

**Fix.** Null the caches only when `nodesChanged`, mirroring the existing
plan-invalidation guard.

### B4. Re-running an accuracy benchmark silently discards the new result — HIGH

**What's wrong.** `dashboard.js:10164-10171`: SSE `result` events are deduped
with `this.accAllResults.some(r => r.model_id === ... && r.benchmark === ...)`
and **dropped** on match. `accAllResults` accumulates across runs by design
and the server appends every result
(`admin/accuracy_benchmark.py:648` `_accumulated_results.append(result_data)`).
So re-running the same model+benchmark — the normal "changed a setting, run
again" flow — drops the fresh result event; the card keeps showing the old
run's accuracy, and the follow-up `upload` event (`findIndex` at 10179)
attaches the new run's upload URL to the stale card. A page reload then shows
both entries.

**Fix.** Key the replay-dedupe by run (include `bench_id`/timestamp in the
result payload) or replace-in-place instead of dropping.

### B5. Opening Settings silently rewrites the persisted SSD cache size (and destroys `auto`) — HIGH

**What's wrong.** `loadGlobalSettings` (dashboard.js:6634-6640) converts the
server value to an integer percent (`parseCacheToPercent`, 10591-10608,
`Math.round`) and immediately calls `updateCacheFromSlider()` (10838-10841),
overwriting `globalSettings.cache.ssd_cache_max_size` with a percent-derived
`${gb}GB` string (`percentToCacheString`, 10611-10616). `saveGlobalSettings`
always sends the field (6723).

**Failure scenario.** Cache configured as `auto` (or `100GB` on a 4TB disk);
user opens Settings, toggles an unrelated switch, saves — the stored size
silently becomes e.g. `81GB` (whole-percent rounding ≈ 0.5% of disk = tens of
GB), and `auto` (resolved to a concrete size by the GET at
`routes.py:3503-3508`) is persisted as a fixed value forever. Rated HIGH
because the corruption is silent, persists in on-disk config, and the `auto`
semantics cannot be recovered from the UI without knowing they were lost.

**Fix.** Keep the server string as source of truth; rewrite
`ssd_cache_max_size` only when the user actually moves the slider or edits the
GB input.

### B6. Idle-timeout dropdown commits (or reverts) every other unsaved Global setting — MEDIUM

`_settings.html:576-577`: `@change="...; saveGlobalSettings()"` posts the
**entire** settings form. Scenario: user edits Host/Port/sampling without
saving, changes Idle timeout — everything is committed as a side effect; if
the save fails validation, `loadGlobalSettings()` (dashboard.js:6766/6772)
reloads server values and silently discards all in-progress edits. Fix: post a
narrow `{ idle_timeout_seconds }` patch (the `saveCtxBenchPriority` pattern,
dashboard.js:9701-9718), or drop the auto-save.

### B7. HF search sorts "Trending / Recently created / Recently updated" are silently overridden — MEDIUM

`syncTableSortToDropdown` (dashboard.js:11820-11836) maps `trending`,
`created`, `updated` to `{col:'downloads', dir:'desc'}`, and `getPagedModels`
(11839-11848) always re-sorts through `sortModels`. The backend returns
correctly ordered results; the table re-sorts them by download count, so the
dropdown (`_models.html:344-352`) visibly does nothing for those three
options. Fix: preserve backend order (attach `rank` as
`loadRecommendedModels` does at 11738-11745 and map these sorts to
`{col:'rank', dir:'asc'}`).

### B8. Model Settings modal: stray backdrop click discards the whole form; no Escape handling — MEDIUM

`_modal_model_settings.html:2-11`: no `@keydown.escape.window` (every other
modal has one — `dashboard.html:48`, `_models.html:1936`, `_models.html:2027`)
and the backdrop `@click="showModelSettingsModal = false"` (line 11) closes
instantly. A mis-click on the dimmed area throws away minutes of
sampling/DFlash/ANE edits with no confirmation. Fix: add the escape handler,
and gate backdrop-close + escape behind a dirty-check confirm (or drop
backdrop-close on this form-heavy modal).

### B9. `openModelSettings` has no request-generation guard — rapid clicks mix two models' state — MEDIUM

`dashboard.js:8076-8122` awaits `loadProfilesForModel(model.id)` +
`loadTemplates()` **before** assigning `this.selectedModel`/showing the modal.
Clicking gear on model A then quickly on model B lets whichever profile fetch
resolves last write `this.profiles` — modal can show B's name with A's profile
pills. No spinner during the await either, so the click feels dead on slow
responses. Fix: monotonic sequence counter (the `_applySeq` pattern already
used by `applyProfileToForm`, 7570-7577) + an immediate loading state.

### B10. Profile "drift" indicator misses newly-set fields — MEDIUM

`computeDrift` (dashboard.js:7048-7060) iterates only
`Object.entries(active.settings)`. Active profile `{temperature: 0.7}` + user
sets `top_k = 40` in the form → no drift dot (`_modal_model_settings.html:185-187`)
although saving diverges — and the backend actually **merges** new fields into
the profile (`routes.py:2758-2795`), which the UI never communicates. Fix:
also compare form keys absent from the profile; surface the merge behavior.

### B11. Unguarded overlapping fetches can clobber newer data (Status + Logs) — MEDIUM

`loadStats` (dashboard.js:8950-8983) has no in-flight token; the 500ms
interval (9034) plus the model-filter `@change="loadStats()"`
(`_status.html:46`) can interleave so a slow earlier *unfiltered* response
lands after the filtered one. Same in `loadLogs` (10521-10563): manual button
+ interval + file-change race, and a 10,000-line tail can exceed a 1s refresh
interval. Self-corrects next tick. Fix: monotonically-increasing request id
checked before assignment.

### B12. Chat double-polls `/admin/api/stats` on two overlapping timers — MEDIUM

While a chat streams, `startStatsPolling` fetches every **200ms**
(chat.html:3859, 4994) *and* `_pollPrefill` independently fetches the same
endpoint every **500ms per streaming chat** (4226, 4244) — `5 + 2N` req/s
against an endpoint that rebuilds full engine-pool/scheduler snapshots per
call (`routes.py:5131`), during exactly the moments the machine is busiest.
Fix: one shared poll loop; `_pollPrefill`'s status text can be derived from
the same response `fetchStats` already claims per-stream.

### B13. Log auto-refresh re-enables itself on every Logs-tab visit — LOW

`startLogRefresh` (dashboard.js:10565-10573) sets `logAutoRefresh = true`
whenever `logRefreshInterval > 0` (default 5), and `handleMainTabChange`
(914-918) calls it unconditionally on tab entry — a user who paused
auto-refresh gets it silently re-enabled after any tab switch. Fix: persist
the on/off choice separately from the interval.

### B14. Zero values silently coerced to "default" in model settings — LOW

`saveModelSettings` (dashboard.js:8281) sends
`ttl_seconds: this.modelSettings.ttl_seconds || null` while the input allows
`min="0"` (`_modal_model_settings.html:400-405`) — entering 0 ("never expire /
no per-model TTL") becomes `null` = "fall back to global idle timeout"
(`ttlPlaceholder`, 8604-8611). Same `|| null` zero-swallow on
`max_context_window`/`max_tokens` (8271-8272). Fix: explicit
`Number.isFinite` checks, as the adjacent temperature/top_p lines already do.

---

## 4. Theme C — Dashboard: silent failures and misleading progress states

### C1. Throughput/context bench SSE treats any transport blip as fatal while the run continues — MEDIUM

`connectBenchSSE` `es.onerror` (dashboard.js:9521-9529) and
`connectContextBenchSSE` (9624-9632): the **first** error event sets
`benchError = t('js.error.benchmark_connection_lost')`, flips
`benchRunning = false`, and closes the EventSource — but EventSource fires
`onerror` on routine auto-reconnects (`readyState === CONNECTING`), and
closing forfeits the server's replay-on-subscribe stream (`routes.py:7161`)
that the dedupe code at 9449-9453 was built for. A blip mid-benchmark shows a
dead "connection lost" run while the server keeps benchmarking; Start
meanwhile yields a raw 409; recovery only via tab-switch (`loadBenchState`).
The accuracy tab does it right — `onerror` falls back to polling + re-attach
(10210-10215). Fix: adopt the accuracy pattern (or at minimum let
`CONNECTING` retries proceed). Related nicety: a deliberate **cancel**
surfaces through the SSE `error` event and renders in the red error box
(`_bench.html:285-290`) — special-case it.

### C2. Claude Code / Integrations settings fail silently, leaving UI state the server rejected — MEDIUM

`saveClaudeCodeSettings` (dashboard.js:8779-8797) and
`saveIntegrationSettings` (8845-8878): on `!response.ok` or throw, only
`console.error(...)` runs. The mode toggle (`_status.html:675-681`) and tier
selects (716-758) keep the new value with no banner and no revert — the
displayed launch command (`claudeCodeCommand`) reflects settings that were
never persisted and revert on reload. Fix: surface the error and restore the
previous value (`saveCtxBenchPriority` pattern, 9701-9717).

### C3. Applying a global template can fail with zero feedback — MEDIUM

`applyTemplateToForm` (dashboard.js:7608-7651): in the create-new-profile
branch only `r.ok` is handled (7638-7645); non-OK falls through with no
`profileError`, catch only logs (7647-7649). Concrete trigger: template name
collides with an exposed profile ID → backend 400
(`_raise_if_profile_id_conflicts_model_id`, routes.py:2906-2920) → the
template pill click does visibly nothing. Fix: set
`this.profileError = data.detail || ...` on both paths, matching
`createProfile` (7562-7566).

### C4. ModelScope tab flashes "unavailable", and a transient failure latches it for the session — MEDIUM

`_models.html:862-865`: `x-show="!msAvailable"` has no "checking" state;
`msAvailable` starts false and `initMsDownloader` (dashboard.js:12025-12043)
sets it only after an await — the amber banner flashes for the round-trip.
Worse, `msInitialized` is set true **before** the fetch, so one transient
network failure shows "unavailable" all session with no retry path. Fix: gate
the banner on `msInitialized && !msAvailable` + a checking state; don't latch
`msInitialized` on fetch failure.

### C5. Status-tab actions and sub-key delete fail silently — LOW

`clearStats`/`clearAlltimeStats` (dashboard.js:8985-9005) never check
`resp.ok` (and don't redirect on 401); `clearSsdCache`/`clearHotCache`
(9007-9029) log to console only; `deleteSubKey` (6820-6835) has no
user-visible branch for non-401 errors — confirm accepted, nothing happens, no
explanation. Fix: surface `data.detail` in the existing inline error slots.

### C6. Bench "Copy results as text" throws on partial results — LOW

`benchBuildText` (dashboard.js:9814, 9840, 9853) calls
`r.e2e_latency_s.toFixed(3)` / `r.total_throughput.toFixed(1)` unguarded while
every other metric goes through null-safe `benchFmtNum` (9763-9766). A row
with null aggregates (the code comments acknowledge they exist) makes the copy
button throw and copy nothing. Fix: route through `benchFmtNum`.

### C7. Accuracy `done` leaves the running panel up on a "Preparing..." stub — LOW

`dashboard.js:10190-10196`: `done` clears `accProgress` and closes the ES but
not `accRunning`; until the 1s `loadAccQueueStatus` poll lands,
`_bench_accuracy.html:267-273` shows the running panel with the hardcoded
`accProgress?.message || 'Preparing...'` and a 0% bar — and sticks if that
poll fails. Fix: optimistically clear `accRunning` on `done` when the queue
mirror is empty.

### C8. Chat performance panel can display a prefill *percentage* as tok/s — LOW

`getClaimedRequestStats` (chat.html:3707-3719): before the tracker produces a
speed, the fallback is `(prefilling.processed / max(total,1)) * 100` — a
percent — which `computeLiveFooterStats` assigns to `base.avg_prefill_tps`
(3737-3739) and the sidebar renders as "Prefill … tok/s" (2333-2341). Fix:
drop the percent fallback (show 0/— until a real speed exists).

### C9. Bench upload progress renders "0 / 0" — LOW

`dashboard.js:9473-9478`: the upload phase sets `current: 0, total: 0`, so
`_bench.html:261` renders "0 / 0" beside the (also hardcoded, §D4)
"Uploading to community benchmarks...". Fix: omit the counter for this phase.

---

## 5. Theme D — Dashboard: i18n

### D1. Locale files are padded with English copies, hiding translation gaps — MEDIUM

All 9 locale files have perfect key parity (1122 keys), but that parity is
manufactured: values identical to English (len>8, containing spaces) number
**ja 345, ko 288, zh-TW 133, ru 132, zh 93** — e.g. the whole
`status.runtime_cache.*` group and `settings.integrations.websearch.*` ship in
English in most locales. Since `_load_locale` (`routes.py:1142-1156`) already
merges English under the selected locale at request time, the padding is
redundant — its only effect is making missing translations invisible to
tooling while users still see English mid-UI. Fix: strip English-identical
padding from the non-English files (mechanical script), rely on the runtime
fallback, and (optionally) add a CI check reporting per-locale coverage.
Placeholder tokens (`{...}`) were verified consistent across all locales — no
interpolation mismatches exist today; the CI check should keep it that way.

### D2. `models.queue.retry_tooltip` doesn't exist, and the `||` fallback can never fire — LOW

`_models.html:839`:
`:title="window.t('models.queue.retry_tooltip') || 'Retry download'"` —
`window.t` returns the **key** on a miss (base.html:58), which is truthy, so
the tooltip literally reads `models.queue.retry_tooltip` in every locale
(key verified absent from `i18n/en.json`). The MS twin at `_models.html:1263`
uses raw `title="Retry download"`. Fix: add the key, use it in both places;
lint against the `window.t(...) || fallback` pattern (this is the only
instance today — keep it that way).

### D3. The entire Cluster view bypasses translation — MEDIUM

`en.json` contains only 5 `cluster.*` keys (`cluster.pairing.*`). Everything
else in `_cluster.html` — headings ("Use your accelerator pool together",
"Cluster incidents", "Start Cluster", the whole SSH wizard) — and every
cluster string in dashboard.js (`clusterQuickStatus` labels 3597-3768,
`clusterPrimaryActionLabel` 3828-3853, error fallbacks like
`'Something went wrong'` at 1049, the raw
`window.confirm('Regenerating this key disconnects every paired worker…')` at
2311-2314, and the 12 §A1 notification strings) is hardcoded English. A
non-English user gets a fully translated dashboard with one fully-English tab.
Fix: key extraction pass over `_cluster.html` + the cluster sections of
dashboard.js; do it together with A1 (whose fix introduces new user-visible
strings).

### D4. Scattered hardcoded English in otherwise-translated views — LOW

Verified clusters (extraction batch, one PR):
- `_settings.html:755-792` — entire Network card ("Network", "Live",
  "HTTP Proxy", "HTTPS Proxy", "No Proxy", "CA Bundle" + hints); `placeholder="None"` at 652.
- `_models.html` quantizer/uploader: 1305 "(full precision only)", 1317/1324
  "Sensitivity Model" + tooltip, 1329 "None (use source model)", 1429/1438-1440
  "Text Only" + warning, 1591-1748 the whole "About oQ Quantization" panel,
  1947 "Model", 1808/1810 "(personal)"/"(org)".
- `_modal_model_settings.html`: 323 "None", 67 'Fetch latest presets from
  omlx.ai', 95 "no presets available", SpecPrefill block 1068-1111, full
  DFlash block 1121-1287 (labels + hints).
- `_status.html`: 164/214 `'Loading' … ' left'`, 183/235 `'idle'`, 187
  `'est. '`, 274 `'draft scored '/'selected '`, 355 `'prompt: '`, 617 "Host".
- `_logs.html:113` `'Last updated: ' + logLastUpdated`.
- `_bench.html:69-88` "Another throughput benchmark is running" banner + 246
  title attr, 417 "Community Benchmark Upload", 421 "Uploading...", 450
  "Already exists"; `dashboard.js:9475` "Uploading to community benchmarks...";
  `_bench_accuracy.html:273` "Preparing...", 290 "Running: ".
- `chat.html`: 3345 `` `Response ${index+1}` ``, 6819 `'Branch of '`,
  7525/7560/7565 code-block "Render"/"Copy"/"Copied!"/"← Back", 5138
  "New Profile", 5690 'Transcription stream ended without a result'.
- `dashboard.js` action feedback: profile errors 7530/7563/7601/7723/7759,
  HF/MS timeout strings 11180-11183/11756-11758/11916-11918/12084-12087,
  oQ/upload messages 11371-11379/11598/11651-11659; validation field names
  ('Host', 'Port', 'Model Directory'…) interpolated into the translated
  `js.error.required_fields` (6666-6679); task status badges render raw
  backend strings (`x-text="task.status"`, `_models.html:808/1232/1541`).

### D5. Translations interpolated into JS string literals break on quotes/backticks — LOW

`_bench.html:7` embeds `{{ t(tooltip_key) }}` inside a JS backtick literal in
`@mouseenter` — a translation containing `` ` `` or `${` breaks the whole
header row's Alpine expression. `_bench_accuracy.html:50, 95, 438, 491` embed
`{{ t(...) }}` in single-quoted JS strings — any apostrophe (common in
French/Portuguese) breaks the binding. Fix: pass through `| tojson`, or move
to `window.t()` calls inside the expression.

---

## 6. Theme E — Dashboard: accessibility and responsive breakage

### E1. Pill toggles, sort headers, and hover-only menus are unusable by keyboard/screen reader — LOW (bundle)

- ~40 pill toggle buttons (e.g. `_settings.html:121-127`, 291-297, 604-609;
  `_modal_model_settings.html:436-444`) are bare `<button>`s with no
  `role="switch"`, no `aria-checked`, no accessible name; adjacent `<label>`s
  lack `for`/`id` association — screen readers announce an unnamed button.
- Table sort headers are click-only `<div>`s with no
  `role="button"`/`tabindex`/keydown (`_settings.html:1350-1377`,
  `_models.html:105-116`).
- Navbar Models/Settings/Bench dropdowns open on `@mouseenter` only
  (`_navbar.html:57, 127, 192`) — keyboard users cannot reach the sub-items.
  The theme dropdown (258-263) already does it right (`focusin/focusout`,
  escape, `aria-expanded`) — apply that pattern.
- No modal traps focus (concretely: the model-settings modal
  `_modal_model_settings.html:2-11`, HF mirror `dashboard.html:48`, upload
  modal `_models.html:1936`, model-detail modal `_models.html:2027`); the
  uploader refresh button is icon-only with no label/title
  (`_models.html:1829-1833`).
- `_status.html:251-258`: desktop unload button is icon-only with a *sibling*
  tooltip div and no `aria-label` on the button (the mobile variant at 191-195
  does it right with `:aria-label`).
- `_logs.html:28-46`: Lines/Refresh/File `<label>`s not associated with their
  inputs; the log `<textarea>` (102-108) has no accessible name.
- Chat timeline dots are click-only `<div>`s (chat.html:1720-1728); accuracy
  benchmark cards are clickable `<div>` pseudo-checkboxes with no
  `role="checkbox"`/keyboard support (`_bench_accuracy.html:176`).

### E2. Model detail modal hard-coded to 800px — clipped on small viewports — MEDIUM

`_models.html:2031-2032`: `style="width: 800px; height: 70vh;"` (comment even
claims "480px wide"). Below 800px viewport the modal is clipped and the close
button can sit off-screen. Fix: `w-full max-w-[800px]`.

---

## 7. Theme F — macOS app: config and state coherence

### F1. API-key change desyncs menubar + config; a later port change actively reverts the key — HIGH

**What's wrong.** `SecurityScreenVM.applyApiKey`
(`Sources/AppView/ViewModels/SecurityScreenVM.swift:56-77`) only calls
`client.configure(host: client.host, port: client.port, apiKey: key)` — it
never updates `AppServices.config`. Three stale consumers:
(a) `MenubarController` holds `let config` from init and builds
`MenubarStatsPoller(baseURL:apiKey: config.apiKey)`
(`MenubarController.swift:731`); (b) "Open Web Dashboard" builds
`webAdminURL(... apiKey: config.apiKey)` (1010) — auto-login silently falls
back to the login form after a key change; (c) worst:
`AppServices.applyServerEndpoint` calls
`client.configure(host:port:apiKey: updated.apiKey)`
(`AppServices.swift:413`) with the stale key — changing the port after
changing the key **reverts the HTTP client to the old key**; once the session
cookie expires, every admin screen 401s until relaunch.

**Fix.** Route key changes through `services.updateConfig(...)` (single source
of truth) and notify/rebuild the menubar stats poller.

### F2. Stale `terminationHandler` can clobber a freshly started server — MEDIUM

`ServerProcess.handleProcessExit` (`ServerProcess.swift:349-368`)
unconditionally does `process = nil; closeLog()` and, when
`state == .starting`, `tryAutoRestart(...)`. The handler is enqueued via
`DispatchQueue.main.async` (331-334), while `stop()`/`forceRestart()` only
poll `proc.isRunning` (231, 260). If the old process's handler lands after a
new child was spawned, it nulls the new `process` reference, closes the new
log handle, and `tryAutoRestart` schedules a second spawn — orphaning the
first child (menubar stop can't reach it) and typically producing a port
conflict. Fix: capture `proc` in the terminationHandler closure and bail in
`handleProcessExit` unless `self.process === proc`.

### F3. Retrying Start against a persistent failure gives zero feedback — MEDIUM

Alert dedup: `serverPortConflict` suppresses on repeated key
(`guard lastPresentedPortConflictKey != key else { return }`,
`MenubarController.swift:809-815`); `presentServerFailureAlert` guards on
`lastPresentedFailureMessage != message` (948); and `ServerProcess.update`
no-ops on identical state (`guard state != next`, `ServerProcess.swift:504`)
so the header doesn't change either. A second "Start Server" click while the
same process still owns the port does nothing visible. Fix: reset dedup keys
on user-initiated start (or key dedup on time).

### F4. `.unresponsive` shows "Server is off" in Serving Stats and empties the Models submenu — MEDIUM

`rebuildStatsSubmenu` gates on `if case .running = server?.state`, else
inserts `menubar.stats.server_off` (`MenubarController.swift:626-632`);
`rebuildModelsSubmenu` uses the same `serverIsRunning` guard (1044,
1326-1329). `.unresponsive` means "under heavy load, /health delayed"
(`ServerProcess.swift:274-277`) — exactly when users check the menu — yet the
menu claims off while the header says "Server: unresponsive". Fix: treat
running-like states as "on" for these submenus (stale data + a "not
responding" line).

### F5. Menubar Models submenu can stick on "Loading…"; load/unload failures fully silent — MEDIUM

`refreshModels` swallows fetch errors
(`guard let resp = try? await client.listModels() else { return }`,
`MenubarController.swift:1203`) so `modelsFetched` stays false and the submenu
shows `menubar.models.fetching` ("Loading…", rendered at 1047-1054) forever
if `/admin/api/models` keeps failing (e.g. auth broken per F1). `loadModelAction`/`unloadModelAction` catch errors
with only `loadingIDs.remove(id)` / `unloadingIDs.remove(id)` (1223-1224,
1237-1238) — no message, no `rebuildModelsSubmenu()` in the catch, so an open
menu keeps "Loading model…" until the next successful refresh. Fix: disabled
error row in the submenu + rebuild after dropping the pending id.

### F6. Screens show stale data as live when the server dies underneath them — MEDIUM

`StatusScreenVM.tick` keeps previous `stats` on failure
(`StatusScreenVM.swift:92-100`) — uptime, GPU memory, Active Now freeze at
last values; `ModelsScreenVM.refresh` keeps `allModels`
(`ModelsScreenVM.swift:97-105`) so "Active Models · N loaded" persists after a
crash/external kill. Only cue: 11pt red text at the bottom of the scroll view
(`StatusScreen.swift:128-133`, `ModelsScreen.swift:37-43`). The menubar
handles this correctly (`markServerStopped()` blanks rates,
`MenubarMetricsStore.swift:57-60`). Fix: clear or visibly dim screen data when
polling fails or `serverState` leaves running-like.

---

## 8. Theme G — macOS app: error surfacing and stuck states

### G1. Failed update download is invisible — "Download, Install & Relaunch" can silently do nothing — HIGH

`UpdateController.startDownload` `onError` sets
`self.lastError = ...; self.state = .available(info)`
(`Sources/Updater/UpdateController.swift:331-336`), but `UpdatesSection`
renders `updates.lastError` **only** in the `.idle` case
(`StatusScreen.swift:721-724`); in `.available` the secondary line is "Ready
to download · size", and the confirmation sheet is already dismissed
(`confirmUpdate`, `AppView.swift:63-66`). Network drop mid-download → UI
quietly returns to "oMLX X is available"; the promised relaunch never happens,
no error anywhere. Fix: render `lastError` in `.available`/`.ready` too (or
NSAlert on download failure).

### G2. Background polls erase user-action errors within seconds — HIGH

`DownloadsScreenVM.refreshTasks()` sets `self.lastError = nil` on every
successful 1s poll (`DownloadsScreenVM.swift:486`; loop 140-145), wiping the
`startDownload` catch (`self.lastError = error.omlxDescription`, 395) ≤1s
later — a rejected download (bad repo id, gated repo, 500) flashes and
vanishes; the user thinks it started. Same pattern: `ModelsScreenVM.refresh`
(`ModelsScreenVM.swift:101`, 2s loop) erases load/unload/delete errors;
`StatusScreenVM.tick` (`StatusScreenVM.swift:97`) erases
`clearStats`/`clearSsdCache` errors within 5s. (`QuantizationScreenVM` was
checked and does **not** have this bug — it only clears on new start; use it
as the template.) Fix: separate slots for poll errors vs action errors, or
only clear the error the poll itself produced.

### G3. Stopping the server produces a spurious red error on the Server screen — MEDIUM

`ServerScreen` reloads on every state change
(`.onChange(of: services.serverState) { ... vm.load(...) }`,
`ServerScreen.swift:195-199`); on `.stopping`/`.stopped` the
`getGlobalSettings` call fails and `load()` sets
`lastError = error.omlxDescription` (`ServerScreenVM.swift:92-94`) rendered by
`HintFooter` (744-747) — a connection error appears right after the user
deliberately clicked Stop. Fix: skip the reload (or clear the error) when the
new state is not running-like.

### G4. Apply requires a running server without saying so; phantom "pending changes" after a failed load — MEDIUM

`applyServerSettings` always PATCHes over HTTP first
(`ServerScreenVM.swift:262-264`), so with the server stopped every Apply —
including a port-only change that `applyServerEndpoint` can persist offline
(`AppServices.swift:400-420`) — fails with a raw connection error. And when
`load()` failed, baselines keep defaults (`baselinePortText = "8000"`, 45)
while `applyConfig` seeds drafts from the real config (390-398) — a
non-default port makes `hasPendingServerChanges` true (119-131) and Apply
lights up with nothing edited. Fix: baseline from `applyConfig` when `load()`
fails; route offline endpoint changes through the local save path with a
"server offline" notice.

### G5. Restart silently ignores an invalid port draft — MEDIUM

`ServerScreenVM.restart` (467-484):
`let portChanged = parsedPort.map { $0 != effectivePort } ?? false` — garbage
input parses to nil ⇒ `portChanged` false ⇒ both validation guards
unreachable. User types "8O00", clicks Restart → restarts on the old port,
typo still in the field, no error. Fix: validate `portText` independently of
`portChanged` before restarting.

### G6. "Initial Cache Blocks" can never be cleared; Apply never converges — MEDIUM

`PerformanceScreenVM.save` sends the field only when non-nil
(`if initBlocks != loadedInitialCacheBlocks, let n = initBlocks`,
`PerformanceScreenVM.swift:225-227`) and converges the baseline only
`if let n = initBlocks` (253). Emptying the field + Apply sends nothing, the
server keeps the old value, and `hasPendingChanges`
(`parsedInitialCacheBlocks != loadedInitialCacheBlocks`, 68) stays true —
Apply enabled forever, looking like a failed save. Fix: explicit-null patch
(the `PatchOptionalInt` machinery already exists for idle timeout, 145-163).

### G7. Welcome wizard silently drops the user's API key on a slow first boot — MEDIUM

`WelcomeWindow` waits ≤8s for health
(`waitUntilHealthyOrTimeout(proc:timeout: 8)`,
`Sources/Welcome/WelcomeWindow.swift:~375`), then
`_ = await setupServerApiKey(...)` — the helper `catch { return false }`
(~398-408) — and advances to `.complete` regardless. First run on a slow
machine (venv unpack >8s): the typed key is never installed, no message, admin
surfaces later behave keyless. Fix: retry after health, or a non-fatal "API
key not applied yet — set it in Security" notice on the complete step.

### G8. Throughput bench stuck on "Running…" after a server restart — MEDIUM

`pollResults` treats every error as transient
(`ThroughputBenchScreenVM.swift:333-339`); `running` clears only on a terminal
`resp.status`. After a server crash/restart the bench id is gone,
`getBenchResults` 404s forever → permanent running state, Run disabled via
`canRun` (74-80), an HTTP 404 string the only hint (Cancel happens to recover,
261-266). Fix: treat `.http(status: 404)` as terminal (bench lost) and reset
`running`.

---

## 9. Theme H — macOS app: i18n, accessibility, terminology

### H1. Hardcoded English with manual pluralization in menubar Live Activity — LOW

`MenubarStatsPoller.Stats.LiveActivity` builds user-visible strings outside
the catalog: `"\(requestCount) queued request\(requestCount == 1 ? "" : "s")"`
(`MenubarStatsPoller.swift:178`), `"Active request"` (187),
`"\(formatDuration(etaSeconds)) left"` (148), `"tok/s"` fragments — rendered
inside the otherwise-localized Serving Stats submenu
(`MenubarController.swift:648-656`). Also `"No results yet."` in `exportText`
(`ThroughputBenchScreenVM.swift:171`). Fix: `String(localized:)`. (The fixed
LIV/AVG/ALL/PP/TG glyph tags are documented as intentionally unlocalized —
leave them.)

### H2. "Set OMLX_API_KEY to enable stats" placeholder is misleading and effectively unreachable — LOW

`rebuildStatsSubmenu` shows `menubar.stats.no_api_key` only when
`statsPoller == nil` (`MenubarController.swift:638-641`), but the poller is
nil only if `liveBaseURL()` fails (719-720) — never because of a missing key.
With no key, the All-Time rows just show "—" forever with no hint; and the
message references an env var instead of the Security screen that actually
manages keys. Fix: key the placeholder on `config.apiKey == nil` and point at
Settings > Security.

### H3. Image-only buttons lack accessibility labels app-wide — LOW (bundle)

Only two `accessibilityLabel` calls exist in the UI layer (`AppView.swift:407`;
`Theme/Components/CodeChip.swift:68`); status-item buttons are labeled
(`MenubarController.swift:153`, `MenubarMetricItemsController.swift:136`).
Unlabeled: favorite star (`ModelsScreen.swift:216-223`), eject/unload
(127-133), trash/delete (285-297), settings chevron (275-281), Server screen
`plus`/`folder`/`trash` model-dir buttons (`ServerScreen.swift:223-276`),
Status clear-stats trash (`StatusScreen.swift:41-47`). `.help()` is a tooltip,
not a reliable VoiceOver label. Fix: `.accessibilityLabel` matching the
existing `.help` strings.

---

## 10. Theme I — Cross-surface consistency

### I1. Terminology drift, in-app and across surfaces — LOW (reference tables)

Within the mac app:
- Server states: menubar header "oMLX stopped / Server: starting… / running
  (port N) / stopping… / unresponsive (auto-recover or Force Restart) /
  failed — msg" (`MenubarController.swift:458-512`) vs hero pills
  "Running/Starting/Stopping/Stopped/Unresponsive/**Error**" with subtitle
  "**Not running**" (`ServerScreen.swift:387-425`). "Stopped" vs "Not running",
  "failed" vs "Error" coexist.
- "All Time" (`StatusScreen.swift:37-39`) vs "All-Time"
  (`MenubarController.swift:685-687`).
- Model lifecycle: menubar "Load model / Unload model / Loading model…" vs
  screen buttons "Load / Unload"; badges "Loaded / Idle" (ModelsScreen) vs
  "Generating / Waiting / Loaded" (StatusScreen Active Now); menubar sections
  "Loaded / Favorites / Library" vs screen "Active Models / Model Library".

Within the dashboard cluster view, the same machines are called "workers"
(`_cluster.html:1518`), "Macs" (1638), "nodes" (1234), "devices"
(dashboard.js:3644), and "accelerators" (2387) in adjacent cards; the peer is
both "Coordinator · rank 0" and "This Mac". In chat, the streaming header
shows the raw gateway model id (`currentStream()?.sourceModel`,
chat.html:1610) while the finished-message header shows the alias display name
(`variantModelLabel` → `meta.modelDisplay`, 3349-3353) — the model name
visibly changes when a response completes.

Cross-surface: memory-guard terms match (both say "Prefill Memory Guard",
tiers Safe/Balanced/Aggressive/Custom — `PerformanceScreen.swift:150-195` vs
`settings.resource.guard_tier.*`). The mac app has **no cluster/peer UI** (the
only "cluster" hits are CPU E/P-cluster sampling and the CLI shim in
`ShellEnvWriter.swift:62-76`), so cluster terminology lives in the dashboard
alone. Fix: pick one term per concept (suggest: server states
stopped/starting/running/stopping/unresponsive/failed everywhere; "node" for
cluster machines; alias display name in both chat headers) and sweep.

### I2. API keys persisted in plaintext localStorage without disclosure — LOW

`dashboard.js:721` persists a third-party bench provider key
(`omlx_bench_external_api_key`), and chat persists the server API key
(`chat.html:5236` `API_KEY_STORAGE_KEY`, seeded from the server at
3925-3929) in localStorage. Deliberate convenience — but add a "stored in this
browser" note next to both inputs so users on shared machines can decide.

---

## 11. Phased implementation checklist

Ordering: data-loss and dead-interaction fixes first; then silent-failure
surfacing; then form/interaction correctness; then i18n/a11y batches. Effort tags: S(<~1h) / M(half-day) /
L(multi-day). One focused fix per item, with tests where testable (JS: add
targeted Playwright/pytest-driven template checks where cheap; Swift: unit
tests on the VMs, which are already test-covered patterns in
`Tests/oMLXTests/`). All line refs verified 2026-08-24 @ `15c10d85`; re-grep
the quoted identifiers if executing much later.

### Phase 1 — Data loss, dead interactions, wrong state (dashboard)

**DONE 2026-08-25 — all 8 items shipped, commit `6d22a062`, pushed to fork.**
Verified live in a browser (isolated scratch server, Playwright): logged in,
exercised the toast system end-to-end (this caught a real bug — see below),
confirmed zero auto-save on Settings load, confirmed the idle-timeout patch
posts exactly `{"idle_timeout_seconds":...}`, swept Dashboard/Settings/
Cluster/Chat for console errors (one pre-existing, unrelated hit — §A3,
correctly out of Phase 1 scope). 386 admin pytest tests pass. Real bug found
*by* the live-browser step, not by code review: the new toast container's
Tailwind classes (`bottom-4`/`right-4`/`z-[70]`/`w-80`) weren't in the
committed, pre-built `static/css/tailwind.css` (this repo ships a compiled
CSS artifact, not a build-on-serve pipeline) — the toast rendered at `(0,0)`
with `z-index:auto`, invisible behind the nav. Rebuilt with `tailwindcss@3.4.17`
pinned to match the committed file's version exactly (a newer 3.4.19 produces
a semantically-identical but noisily-reordered file); verified via structural
rule-diff, not raw text diff (minified CSS is one line): 6 selectors added, 0
removed, 46 pre-existing rules reordered only (same declarations). **Any
future dashboard.html/dashboard.js change introducing a new Tailwind class
needs the same rebuild step, or it silently no-ops in production.**

- [x] **1.1** [S] Add the missing `>` in the Settings copy-name button
  (`_settings.html:1428-1429`). (§A2)
- [x] **1.2** [M] Implement `showNotification` (or reroute the 12 call sites)
  + `finally` for `clusterSshKeyGenerating`/token/keychain busy flags
  (`dashboard.js:2328-2422`). (§A1) — small bottom-right toast queue
  (`notifications: []`, `showNotification`/`dismissNotification`), rendered
  via `x-for` in `dashboard.html`; the 3 missing-`finally` sites fixed.
- [x] **1.3** [S] Chat delete: confirm gate + honest label (or single-turn
  delete) in `deleteMessage`/`sliceBeforeMessage`
  (`chat.html:6741-6755,3068-3074`). (§B1) — kept the shared
  delete/edit-regenerate truncation semantics (relabeling was the doc's own
  accepted fallback), relabeled "Delete from here", added a `confirm()`
  describing the real scope.
- [x] **1.4** [S] Preserve worker `previous.role` in
  `syncClusterNodesFromPeers` (`dashboard.js:2027`; pattern at 1966) — or
  disable role buttons for workers if headless-only is intended. (§B2)
- [x] **1.5** [S] Null cluster inventory/catalogue only when `nodesChanged`
  (`dashboard.js:1922-1923`; guard pattern at ~2063). Kills the false-"Ready"
  flap and the 10s SSH storm. (§B3)
- [x] **1.6** [S] Accuracy SSE result dedupe by run id, or replace-in-place
  (`dashboard.js:10164-10179`; server appends at
  `accuracy_benchmark.py:648`). (§B4) — replace-in-place (findIndex + splice,
  mirroring the existing `upload`-event handler immediately below it).
- [x] **1.7** [S] Stop rewriting `ssd_cache_max_size` on settings load
  (`dashboard.js:6634-6640,10591-10616,10838-10841`) — slider/input edits
  only. (§B5) — removed the `updateCacheFromSlider()` write-back call from
  `loadGlobalSettings`; `cachePercent` is still computed for the slider's
  initial display, the persisted field is untouched until a real interaction.
- [x] **1.8** [S] Narrow-patch idle timeout instead of full-form
  `saveGlobalSettings` on `@change` (`_settings.html:576-577`; pattern
  `dashboard.js:9701-9718`). (§B6) — new `saveIdleTimeout()`, optimistic
  update + revert-and-toast on failure, mirrors `saveCtxBenchPriority` exactly.

### Phase 2 — Data loss, dead interactions, wrong state (macOS app)

- [x] **2.1** [M] Single source of truth for API key: route
  `SecurityScreenVM.applyApiKey` through `services.updateConfig`, rebuild the
  menubar stats poller, fix `applyServerEndpoint`'s stale-key configure
  (`SecurityScreenVM.swift:56-77`; `AppServices.swift:413`;
  `MenubarController.swift:731,1010`). (§F1)
  Done: `AppServices.updateConfig` now posts `configDidChangeNotification`
  after `client.configure(...)`; `SecurityScreenVM.setupApiKey`/`applyApiKey`
  take `services: AppServices` and call `services.updateConfig` instead of
  writing through `client` directly; `MenubarController` observes the
  notification and rebuilds its stats poller (`config` changed `let` → `var`).
  `applyServerEndpoint` needed no direct change — it already derives from
  `config`, so fixing where the key is set fixed its stale-key read too.
- [x] **2.2** [S] Split poll-error vs action-error slots in
  `DownloadsScreenVM`/`ModelsScreenVM`/`StatusScreenVM`
  (`DownloadsScreenVM.swift:140-145,395,486`; `ModelsScreenVM.swift:101`;
  `StatusScreenVM.swift:97`; `QuantizationScreenVM` is the correct
  template). (§G2)
  Done: each VM's poll-loop refresh function (`refreshTasks`/`refresh`/`tick`)
  takes `clearsError: Bool = true`; the 2s/5s poll loops call it with `false`,
  the existing action methods (load/unload/favorite/remove/clear*) keep the
  default `true` so they still report/clear their own errors.
- [x] **2.3** [S] Show `updates.lastError` in `.available`/`.ready` branches
  of `UpdatesSection` (`StatusScreen.swift:721-724`;
  `UpdateController.swift:331-336`). (§G1)
  Done: both branches now render `updates.lastError ?? <default text>`
  instead of unconditionally showing the default text.
- [x] **2.4** [S] Guard `handleProcessExit` with `self.process === proc`
  (capture in terminationHandler) (`ServerProcess.swift:331-334,349-368`).
  Testable via `ServerProcessIntegrationTests`. (§F2)
  Done: `terminationHandler` captures `proc` and dispatches to a new
  `handleProcessExit(proc:code:)` which no-ops unless `self.process === proc`.
  Verified with the real (non-mocked) `OMLX_INTEGRATION=1` spawn/shutdown
  smoke test, not just the hermetic unit tests.
- [x] **2.5** [S] Reset alert-dedup keys on user-initiated Start
  (`MenubarController.swift:809-815,948`; `ServerProcess.swift:504`). (§F3)
  Done: `startServer()` clears `lastPresentedPortConflictKey` and
  `lastPresentedFailureMessage` before attempting to start, so a repeated
  identical failure re-alerts instead of being deduped away.

### Phase 3 — Silent failures and misleading progress

- [x] **3.1** [M] Throughput/context bench SSE: adopt the accuracy
  reconnect/poll-fallback pattern; special-case cancel so it doesn't render as
  an error (`dashboard.js:9521-9529,9624-9632`; good pattern 10210-10215;
  `_bench.html:285-290`). (§C1)
  Done: backend now emits a distinct `"cancelled"` SSE event type on user
  cancel instead of reusing `"error"` (`benchmark.py` x2, `context_benchmark.py`
  — plus their `_BENCH_TERMINAL_TYPES`/`_CTX_TERMINAL_TYPES` sets, so the new
  type still closes the stream). Frontend handles `cancelled` without setting
  `benchError`/`ctxBenchError`, keeps the `loadModels()` call (backend unloads
  the model on cancel). `es.onerror` on both now polls the existing
  `.../results` REST endpoint every 3s and reconnects the SSE if still
  running, instead of immediately declaring the run dead.
- [x] **3.2** [S] Error + revert on failed Claude Code/Integrations saves
  (`dashboard.js:8779-8797,8845-8878`; `_status.html:675-758`). (§C2)
  Done: `showNotification` toast on failure, revert from a shadow snapshot
  (`_lastSavedClaudeCode`/`_lastSavedIntegrations`, refreshed in
  `loadGlobalSettings` and after each successful save) — deliberately NOT a
  full `loadGlobalSettings()` reload, which `saveIdleTimeout`'s own comment
  documents as clobbering unrelated unsaved edits elsewhere on the tab.
- [x] **3.3** [S] `profileError` on non-OK/throw in `applyTemplateToForm`
  (`dashboard.js:7638-7649`; match 7562-7566). (§C3)
  Done: matches the sibling `createProfile`/`applyProfileToForm` pattern.
- [x] **3.4** [S] MS availability: checking state + no latch-on-failure
  (`_models.html:862-865`; `dashboard.js:12025-12043`). (§C4)
  Done: new `msChecking` flag (also the re-entrancy guard, replacing the
  synchronous `msInitialized = true` set-before-await) drives a neutral
  "Checking…" banner; `msInitialized` only latches on a definitive server
  answer (available true or false), not on a network/HTTP failure, so a later
  tab click retries. Verified live via Playwright: checking→unavailable
  transition, no flash.
- [x] **3.5** [S] Surface `data.detail` in
  `clearStats`/`clearAlltimeStats`/`clearSsdCache`/`clearHotCache`/`deleteSubKey`
  (`dashboard.js:8985-9029,6820-6835`). (§C5)
  Done: all 5 show a `showNotification` toast with `data.detail` on failure.
- [x] **3.6** [S] Menubar models submenu: error row + rebuild in catch
  (`MenubarController.swift:1203,1223-1238,1047-1054`). (§F5)
  Done: `refreshModels()`'s catch sets `modelsFetchFailed` and rebuilds the
  submenu with a "Failed to load models" row — but only when the failure
  wasn't the routine cancellation `scheduleModelsRefresh()` triggers by
  design on every re-open/action (guarded via `Task.isCancelled` +
  `URLError.cancelled`).
- [x] **3.7** [S] Treat running-like states as "on" in
  `rebuildStatsSubmenu`/`rebuildModelsSubmenu`
  (`MenubarController.swift:626-632,1044,1326-1329`). (§F4)
  Done: `serverIsRunning` and `rebuildStatsSubmenu`'s local check now use
  `ServerProcess.State.isRunningLike` (`.running` or `.unresponsive`) instead
  of `.running` only. `menuAvailability(for:)` (browser-link gating)
  deliberately left `.running`-only — its own comment documents that as a
  stricter, intentional gate distinct from process-alive display.
- [x] **3.8** [S] Dim/clear screen data when server leaves running-like
  (`StatusScreenVM.swift:92-100`; `ModelsScreenVM.swift:97-105`; menubar
  pattern `MenubarMetricsStore.swift:57-60`). (§F6)
  Done: added `clearOnServerStopped()` to both VMs (blanks `stats`/
  `allModels` + `lastError`, mirroring `MenubarMetricsStore.markServerStopped()`),
  wired via `.onChange(of: services.serverState)` in both Screens.
- [x] **3.9** [S] Skip ServerScreen reload/error on `.stopping`/`.stopped`
  (`ServerScreen.swift:195-199`; `ServerScreenVM.swift:92-94`). (§G3)
  Done: gated on `newState.isRunningLike` (allow-list, not a
  `.stopping`/`.stopped` deny-list) — `.starting` would hit the same
  connection-refused spurious-error class this item targets.
- [x] **3.10** [S] Bench 404 = terminal in `pollResults`
  (`ThroughputBenchScreenVM.swift:333-339,74-80`). (§G8)
  Done: a 404 (`OMLXClientError.http(404, _)`) stops the poll and surfaces a
  real message instead of retrying forever. Also added `"error"` to the
  terminal-status set — the backend reports failures as `"error"`, never
  `"failed"`, so a server-side error was hitting the identical stuck-`running`
  pathology without needing a 404.
- [x] **3.11** [S] Welcome wizard: retry `setupServerApiKey` after health, or
  visible "key not applied" notice (`WelcomeWindow.swift:~375-408`). (§G7)
  Done: both — distinguishes the benign "already had a key" 400 from a real
  failure, retries once after ~2.5s on real failure (first launches are
  exactly when the 8s health-wait is likeliest to have just missed the server
  coming up), then shows a non-fatal amber notice on the completion step
  pointing to Settings → Security if it's still failed.

### Phase 4 — Form correctness and interaction polish (small design decisions)

- [x] **4.1** [S] Model Settings modal: escape handler + dirty-check on
  backdrop/escape (`_modal_model_settings.html:2-11`; patterns
  `dashboard.html:48`, `_models.html:1936,2027`). (§B8)
  Done: snapshot `this.modelSettings` (JSON.stringify) when the modal opens;
  a new `closeModelSettingsModal()` confirms before closing if it changed,
  wired to all 3 close paths (backdrop, header X, footer Cancel) plus a new
  guarded `@keydown.escape.window`. `saveModelSettings()` still closes
  directly on success (nothing to discard there).
- [x] **4.2** [S] `openModelSettings` sequence counter + loading state
  (`dashboard.js:8076-8122`; `_applySeq` pattern 7570-7577). (§B9)
  Done: a per-call seq threaded as an `isCurrent()` guard into
  `loadProfilesForModel`/`loadTemplates` (both gained an optional
  `isCurrent` param, defaulting to a no-op for their other, non-racing
  callers) so a superseded response can't overwrite a newer call's data —
  not just guarding the final assignment, which alone would've missed the
  two functions' own internal `this.profiles`/`this.templates` writes.
  `modelSettingsLoadingId` drives a per-row spinner + disabled state on both
  "Settings" buttons (`_settings.html`, `_models.html` manager).
- [x] **4.3** [S] `Number.isFinite` instead of `|| null` for
  `ttl_seconds`/`max_context_window`/`max_tokens`
  (`dashboard.js:8271-8281`). (§B14)
  Done, matches the pattern already used for temperature/top_p/etc. right
  below it.
- [x] **4.4** [S] Drift check includes form-only keys; surface backend
  merge-into-profile behavior (`dashboard.js:7048-7060`;
  `routes.py:2758-2795`). (§B10)
  Done: `computeDrift()` now does a symmetric union-of-keys comparison
  (`_profileDrift`), skipping profile values of `null` to match the
  backend's "None = unconstrained" semantics. A second helper
  (`_profileDiverged`) mirrors the backend's own narrower profile-keys-only
  check; `saveModelSettings()` computes both fresh at save time and shows a
  toast when `drift && !diverged` — that combination means the save's only
  effect on the profile is the backend silently merging new fields into it,
  not unlinking it, which the generic drift indicator doesn't communicate.
- [x] **4.5** [S] Rank-preserving sort for trending/created/updated
  (`dashboard.js:11820-11848`; pattern 11738-11745). (§B7)
  Done: `syncTableSortToDropdown()`'s map now points trending/created/updated
  at `col: 'rank'` instead of `col: 'downloads'` (which was silently
  discarding the server's own sort order — the opposite of the function's
  own stated purpose). `hfSearchResults` gets a `rank` field attached in
  server-response order, mirroring the existing trending/popular pattern.
- [x] **4.6** [S] Validate port text independently of `portChanged` in
  `ServerScreenVM.restart` (`ServerScreenVM.swift:467-484`). (§G5)
  Done: validates the port text unconditionally via `guard let`, then
  derives `portChanged` from the already-validated value — the old
  `portChanged = parsedPort.map{...} ?? false` defaulted to `false` on a
  parse failure, which skipped both validation guards below it entirely.
- [x] **4.7** [M] Explicit-null patch for Initial Cache Blocks
  (`PerformanceScreenVM.swift:225-227,253,68`; `PatchOptionalInt`
  145-163). (§G6)
  Done: `GlobalSettingsPatch.initialCacheBlocks` changed from `Int?` to
  `PatchOptionalInt?`, mirroring `idleTimeoutSeconds`'s existing 3-state
  pattern exactly (parse/patch-construction/baseline-convergence). Was
  previously a plain optional with "empty = leave alone" semantics — there
  was no way to actually clear a previously-set value.
- [x] **4.8** [M] ServerScreen offline Apply: baseline from `applyConfig` on
  failed load; offline endpoint changes via local save + notice
  (`ServerScreenVM.swift:45,119-131,262-264,390-398`;
  `AppServices.swift:400-420`). (§G4)
  Done: `applyConfig`'s `!hasLoaded` fallback branch now also calls
  `snapshotApplyBaselines()`, so baselines reflect the real local config
  instead of hardcoded struct defaults when the server was offline on first
  load. `applyServerSettings` now checks `services.serverState.isRunningLike`
  before attempting the live PATCH: non-endpoint field changes (sampling,
  aliases, hfCacheEnabled, storage) fail fast with a clear message instead
  of silently no-op'ing; a port-only change persists via a new
  `AppServices.saveServerEndpointOffline(host:port:)` and sets a
  `offlineApplyNotice` (rendered in `HintFooter`, amber, distinct from
  `lastError` — same pattern as `WelcomeViewModel.apiKeyWarning`). The new
  method deliberately does NOT reuse `applyServerEndpoint`, whose
  managed-server branch calls `server.start()` unconditionally — that's
  correct for its own online reconfigure-and-bounce contract but would
  silently launch the server as a side effect of an offline settings save.
- [x] **4.9** [S] Request-id guards for `loadStats`/`loadLogs`
  (`dashboard.js:8950-8983,10521-10563`). (§B11)
  Done: sequence counters on both, guarding every state-touching point
  after an `await` (including the `finally` blocks) so a superseded
  request can't paint stats for the wrong model or logs for the wrong file.
- [x] **4.10** [S] Persist log auto-refresh on/off separately from interval
  (`dashboard.js:10565-10573,914-918`). (§B13)
  Done: new independent `logAutoRefreshEnabled` boolean (persisted,
  separate toggle switch in `_logs.html`) — previously the only way to
  pause was zeroing the interval, destroying the user's preferred cadence.
  Both `logAutoRefreshEnabled` and `logRefreshInterval` now persist to
  localStorage (clamped/validated on load); the timer gates on both.
- [x] **4.11** [M] Chat: single shared stats poll loop
  (`chat.html:3859,4994,4226,4244`; endpoint cost `routes.py:5131`). (§B12)
  Done, narrower than the title suggests: extracted a small
  `_ensureStatsInterval()` helper for the only actually-duplicated logic
  (interval creation) between `ensureStatsPollingForCurrentChat` and
  `startStatsPolling` — NOT a full merge, since the two have materially
  different preconditions (conditional resume vs. unconditional
  stream-start with its own `recentStats` reset) and merging them risked
  changing behavior at call sites without a full trace of the 8000-line
  file's chat-session lifecycle.
- [x] **4.12** [S] Micro-fixes bundle: `benchFmtNum` in `benchBuildText`
  (`dashboard.js:9814,9840,9853`); clear `accRunning` on `done`
  (`dashboard.js:10190-10196`); drop percent-as-tps fallback
  (`chat.html:3707-3719,3737-3739`); omit "0 / 0" upload counter
  (`dashboard.js:9473-9478`). (§C6-C9)
  Done, all 4: `benchBuildText`'s remaining raw `.toFixed()` calls
  (`e2e_latency_s` x3, `total_throughput`) now route through `benchFmtNum`;
  accuracy bench's `'done'` case sets `accRunning = false` directly instead
  of relying solely on `_pollForNextRun`'s ~1s-later poll; the prefilling
  live-footer speed fallback no longer computes a completion percentage and
  feeds it into `avg_prefill_tps` as if it were tok/s; the upload-summary
  row is gated on `total > 0` instead of just truthiness.
- [x] **4.13** [S] Structured `mtplx_sidecar_available` flag instead of the
  English-substring match (`_modal_model_settings.html:686`;
  `routes.py:713-719`). (§A3)
  Done: `_mtp_compat_for_model` now returns a 3-tuple with a structured
  bool threaded through both response dicts (real models + the virtual
  markitdown entry) and `buildModelSettingsState`. Also fixes the
  pre-existing crash spotted during Phase 1 live-testing
  (`Cannot read properties of undefined (reading 'includes')` when no
  model selected) as a side effect — the new field defaults to `false`
  everywhere instead of the old code calling `.includes()` on a
  possibly-undefined string.

### Phase 5 — i18n

- [x] **5.1** [S] Strip English-identical padding from non-English locale
  files (mechanical; runtime fallback already exists at
  `routes.py:1142-1156`) + CI coverage/placeholder-parity check. (§D1)
  Done: stripped 2773 English-identical entries across the 8 non-English
  locale files (~150-460 each). Replaced
  `test_locale_key_sets_identical`'s strict `keys == base` assertion with
  two invariants: `test_locale_keys_are_subset_of_english` (no orphaned
  keys) and `test_locale_values_are_not_redundant_with_english` (no
  locale value byte-identical to English — the actual anti-padding
  guard). This is a policy change from raw-file presence to
  fallback-aware presence, so 9 other test files that asserted specific
  keys existed in the raw locale JSON (not accounting for the runtime
  fallback) had to be updated to call `omlx.admin.routes._load_locale()`
  instead of `json.loads(path.read_text())` directly — same behavior,
  correct model of what actually ships.
- [x] **5.2** [S] Add `models.queue.retry_tooltip`; use it at
  `_models.html:839` and `1263`; lint the `window.t(...) ||` pattern. (§D2)
  Done: added the key (translated in all 8 locales, not padding), fixed
  both call sites (the second was raw `title=`, not even Alpine-bound),
  removed the dead `|| 'Retry download'` fallback (confirmed via grep it
  was the only instance of that pattern).
- [x] **5.3** [L] Cluster view key-extraction pass (`_cluster.html` + cluster
  strings in dashboard.js incl. the §A1 notifications and the raw
  `confirm()` at 2311-2314). Bundle with 1.2. (§D3)
  Done: full pass over `_cluster.html` (2670 lines) and the cluster
  sections of dashboard.js — ~300 new `cluster.*`/`js.error.*` keys
  (en.json only, per the new no-redundant-value policy; translate later
  as needed). Covers every item the doc named: the SSH pairing wizard,
  the confirm() dialog, all A1 notification strings, `clusterQuickStatus`/
  `clusterPrimaryActionLabel`, the memory planner, shard-balance UI, and
  the eviction-toggle card added this session. Skipped the ~2600-1240
  dead `{# ... #}` Jinja comment block (old shard/runtime console,
  intentionally unrendered — verified via grep that every remaining
  hardcoded string lives inside it). **Known gap, deliberately deferred**:
  `clusterSteps`/`clusterStrategyOptions`/`clusterNodeRoles` labels and
  the fabric metric labels ("Ring latency", "Collective throughput", etc.)
  live as string literals inside dashboard.js *data objects* (not
  template expressions), which the doc's D3 inventory didn't call out —
  left for a follow-up pass rather than expanding this PR further.
  13 test files touched (locale-key assertions plus 3 files running
  cluster JS functions in a bare Node sandbox, which needed a
  `global.window = { t: ... }` stub backed by real `en.json` content —
  base.html's real contract, not a passthrough — since `window` didn't
  exist in that environment before any of these functions called
  `window.t()`). Full suite: 10430 passed. Live-Playwright-verified
  against a scratch server: heading renders translated, no raw `{{ t(`
  leaks, Alpine parsed and ran (tab-switch worked), eviction toggle
  renders. Scratch env has no live cluster state, so
  `clusterQuickStatus`'s dynamic paths need coordinator verification.
- [x] **5.4** [L] Scattered-hardcoded-strings extraction batch per the §D4
  inventory (settings network card, oQ panels, DFlash/SpecPrefill blocks,
  status/bench/chat/logs fragments, JS action feedback, task-status
  badges). (§D4)
  Done: full pass across `_settings.html` (Network card), `_models.html`
  (oQ quantizer fields + the entire "About oQ Quantization" educational
  panel, ~37 keys; uploader's personal/org suffix; the modal "Model"
  label), `_modal_model_settings.html` (presets tooltip/empty-state,
  reasoning-parser "None", the full SpecPrefill and DFlash blocks —
  ~45 keys, largest single cluster in this item), `_status.html` (model
  loading/idle/prefill-progress concatenated fragments), `_logs.html`,
  `_bench.html` + `_bench_accuracy.html` (other-benchmark-running banner,
  community upload panel, progress messages), `chat.html` (code-block
  Render/Copy/Copied!/Back buttons, branch-chat naming, new-profile
  default name, transcription error), and `dashboard.js` (validation
  field names interpolated into `js.error.required_fields`, 3x
  duplicated "Name required", 6x "Failed to {save,apply,update} profile/
  template" fallbacks, 8x HF/MS timeout strings + 6x HF/MS connect-failed
  strings — all deduped to 2 shared keys each, oQ/upload start success
  and failure messages). Also fixed the 4 raw `x-text="task.status"`
  backend-enum renders (`_models.html` HF/MS/oQ/upload queues) via a new
  shared `taskStatusLabel(status)` helper backed by 9 new
  `models.task_status.*` keys, falling back to the raw status string for
  any value without a translation (checks `window._t` directly, not
  `window.t()`, since the latter's key-as-fallback behavior would show a
  raw dotted key name instead of the raw backend status on a miss).
  Deliberately skipped pure technical enum values doc didn't name
  (LLM/VLM/Embedding/Reranker/Audio-STT-TTS-STS model-type options,
  DFlash quant bit-depth/group-size option values, verify-mode algorithm
  names) — industry-standard abbreviations and backend identifiers, not
  prose. New keys en.json only, same policy as 5.1-5.3. Full suite:
  10430 passed. Live-Playwright-verified against a scratch server: no
  raw `{{ t(` leaks across the whole page, zero console errors after
  visiting Settings/Models/Bench/Chat tabs, translated Network-card and
  bench labels render correctly.
- [x] **5.5** [S] `| tojson` (or `window.t()`) for translations inside JS
  literals (`_bench.html:7`; `_bench_accuracy.html:50,95,438,491`). (§D5)
  Done: fixed both doc-named sites, plus a full-codebase sweep turned up
  7 more genuinely-unescaped instances of the exact same bug in
  `_settings.html` the doc didn't enumerate (sub-key unnamed fallback,
  model-dir placeholders, restart-button label ternary) — fixed those
  too. Left `_modal_model_settings.html`'s ~11 instances alone: they
  already use a defensive `|replace('\\','\\\\')|replace("'","\\'")`
  filter chain that correctly escapes backslashes/quotes at render
  time, so they're not broken, just a more fragile pattern than
  `window.t()` — not scope-creeping a working (if ugly) fix. **Caught a
  real regression in my own first-pass fix via live-Playwright, not
  pytest**: initially used `window.t({{ tooltip_key | tojson }})` for
  the `_bench.html` macro, which produces valid JSON but Flask/Jinja's
  `tojson` filter marks its output pre-escaped/"safe" (designed for
  `<script>` blocks), so its raw `"` characters broke out of the
  double-quoted `@mouseenter="..."` HTML attribute — a `pageerror:
  Unexpected token '}'` in the browser console, invisible to every
  Python test (confirmed zero tests reference `bench_th`/`benchTip` at
  all). Fixed by embedding the plain key name directly
  (`window.t('{{ tooltip_key }}')`) instead — safe because `tooltip_key`
  is always a developer-authored dotted identifier, never translated
  text or user data, so it never needs JSON-string escaping in the
  first place. Verified via Python template rendering in isolation
  (caught the bug) and in the full real template context (confirmed
  the fix), then re-confirmed zero console errors live in a browser.
  Full suite: 10430 passed both before and after this fix (expected,
  since nothing in the Python suite exercises this code path).
- [x] **5.6** [S] Mac menubar Live Activity strings → `String(localized:)`
  (`MenubarStatsPoller.swift:148,178,187`;
  `ThroughputBenchScreenVM.swift:171`). (§H1)
  Done: localized the prefill "tok/s" and "left" detail fragments, the
  singular/plural "queued request(s)" text (as two distinct keys, per
  this codebase's own convention of dedicated keys per string state
  rather than embedded conditional logic — see e.g.
  `bench.accuracy.benchmarks.subtitle.empty` vs `.count`), the "Active
  request" fallback, and the throughput-bench export's "No results yet."
  Left the `menuBarTitle` glyph-tag strings (`"GEN …"`, `"WAIT n"`,
  `"RUN …"`, `"PP …%"`) untouched — the doc explicitly documents these
  as intentionally unlocalized compact status codes, distinct from the
  full-prose `detail` text shown in the Serving Stats submenu. Added the
  6 new keys to `Localizable.xcstrings` by hand (this project's
  `extractionState: "manual"` convention — `xcodebuild build` does not
  auto-populate the catalog for new `String(localized:)` call sites in
  this setup) using the catalog's actual persisted format (`%lld`/`%@`
  positional specifiers, not literal Swift `\(...)` interpolation
  syntax) confirmed against existing interpolated entries first. English
  only for the new keys; left ru/zh-Hans absent rather than guess.
- [x] **5.7** [S] Fix the no-API-key stats placeholder condition + wording
  (`MenubarController.swift:638-641,719-720`). (§H2)
  Done: the placeholder now keys on `config.apiKey` being nil/empty
  instead of `statsPoller == nil` (the poller is nil only when
  `liveBaseURL()` fails to resolve a host/port — never because of a
  missing key). Tracing `MenubarStatsPoller.refreshOnce()` surfaced the
  doc's second symptom is a distinct bug, not just wrong wording:
  `sessionStats` comes from the public, unauthenticated status endpoint
  and populates regardless of API key, while `alltimeStats` is gated on
  `hasAPIKey` — so `session == nil && alltime == nil` (the condition
  guarding the whole early-return placeholder) is false as soon as any
  poll succeeds at all, meaning the placeholder path was effectively
  unreachable in the real no-key scenario the doc describes ("All-Time
  rows just show '—' forever with no hint"). Fixed by additionally
  gating the All-Time section itself: renders the same no-API-key
  message in place of the four empty stat rows when `!hasAPIKey`, rather
  than only fixing the (rarely-hit) top-level gate. Corrected the
  doc-suggested "Settings > Security" wording to match this app's actual
  flat-sidebar navigation (`sidebar.security` is a direct top-level
  item, no nested "Settings" parent exists) — "Set an API key in the
  Security screen to enable stats". Updated the pre-existing catalog
  entry's `ru`/`zh-Hans` translations to match (referencing the
  already-translated `sidebar.security` values for terminology
  consistency) rather than leaving them describing the old, wrong
  `OMLX_API_KEY` env-var wording.

### Phase 6 — Accessibility, responsive, consistency

- [x] **6.1** [M] Dashboard a11y bundle: `role="switch"`+`aria-checked`+names
  on pill toggles; keyboard/sort-header semantics; keyboard-accessible navbar
  dropdowns (copy the theme-dropdown pattern, `_navbar.html:258-263`);
  aria-labels on icon-only buttons; label/`for` association in logs; focus
  traps on the four modals listed in §E1; keyboard/role semantics for chat
  timeline dots (`chat.html:1720-1728`) and accuracy benchmark cards
  (`_bench_accuracy.html:176`). (§E1) — Done: 46 knob-style toggles across
  `_settings.html` (17), `_modal_model_settings.html` (20), `_models.html`
  (5), `_cluster.html` (1), `_bench_accuracy.html` (2), `_logs.html` (1) got
  `role="switch"` + `:aria-checked="EXPR ? 'true' : 'false'"` (string, not a
  bare boolean — Alpine drops attributes bound to a literal `false`) copied
  verbatim from each button's existing `:class` condition, plus an
  `aria-label` reusing the nearest existing i18n key (dispatched as 3
  parallel background agents, one per file group, each running its own
  pytest slice — 1610+595+170 passed). The 2 per-row model-table toggles
  (Pin/Visible) got dynamic `:aria-label="window.t('key') + ': ' + model.id"`
  instead of a static label, since a static one would announce identically
  on every row. 10 click-only sort-header `<div>`s (`_settings.html` model
  table ×7, `_models.html` manager table ×3) got
  `role="button" tabindex="0"` + `@keydown.enter`/`@keydown.space.prevent`;
  the `toggleTableSort`-driven headers were already real `<button>`s and
  needed no change. Navbar Models/Settings/Bench dropdowns
  (`_navbar.html:57,127,192`) got the theme dropdown's `@focusin`/
  `@focusout`/`@keydown.escape`/`aria-expanded`/`aria-controls` pattern
  copied verbatim. Icon-only buttons fixed: uploader oQ-list refresh
  (`_models.html`), HF-mirror-settings gear (`_models.html`, found live via
  Playwright — its tooltip was a CSS `group-hover` sibling `<div>`, invisible
  to both keyboard focus and `:has-text` queries), status active-models
  unload button (`_status.html`, now matches the mobile variant which was
  already correct). Logs page: `label for=`/`id` association added for the
  Lines/Refresh-interval/File-select inputs, plus `aria-label` on the
  readonly log `<textarea>` (no visible label existed for it at all).
  4 modal focus traps (HF mirror `dashboard.html`, model-settings
  `_modal_model_settings.html`, upload-confirm `_models.html`, model-detail
  `_models.html`): no Alpine plugin is bundled (only core `alpine.min.js`),
  so hand-rolled a shared `window.trapModalFocus(event)` helper in
  `dashboard.js` (cycles Tab/Shift+Tab between first/last focusable
  descendant) plus `role="dialog"`/`aria-modal`/`aria-label(ledby)` +
  `tabindex="-1"` + `x-effect` to focus the panel on open. Chat timeline
  dots (`chat.html:1721-1734`) got `role="button" tabindex="0"` +
  `@keydown.enter`/`@keydown.space.prevent` + dynamic `aria-label` (new key
  `chat.timeline.jump_to_message`) + `@focus`/`@blur` mirroring the existing
  `@mouseenter`/`@mouseleave` tooltip. Accuracy-bench pseudo-checkbox cards
  (`_bench_accuracy.html:182-190`) got `role="checkbox"` + `:aria-checked` +
  keydown handlers guarded with `$event.target === $el` so Space/Enter on
  the card's nested sample-size `<select>` doesn't double-toggle the parent.
  Live-verified via an isolated scratch server + Playwright: navbar dropdown
  `aria-expanded` flips on focus; a real `Enter` keypress flips a switch's
  `aria-checked` (`false → true`); `Enter` on a sort header actually
  re-sorts (`▲` indicator appears); the HF mirror modal panel auto-focuses
  on open (`document.activeElement === panel`) and `Escape` closes it
  cleanly; zero console/page errors throughout. Full pytest: 10430 passed,
  30 skipped. The model-settings/upload/model-detail modal focus traps
  share the exact same `trapModalFocus`/`x-effect` mechanism proven live on
  the HF mirror modal, but weren't separately live-driven end-to-end (the
  scratch server has no models loaded, so those 3 modals have no trigger
  path without seeding model state) — same disclosed verification-gap
  pattern as Phase 5.
- [x] **6.2** [S] Model detail modal `w-full max-w-[800px]`
  (`_models.html:2031-2032`). (§E2) — Done: replaced the hardcoded
  `style="width: 800px; height: 70vh;"` (whose comment even wrongly claimed
  "480px wide") with `class="... w-full max-w-[800px]"` + `style="height:
  70vh;"`, and added `p-4` to the outer `fixed inset-0` wrapper so the modal
  doesn't touch viewport edges below 800px. Verified `max-w-[800px]` wasn't
  already in the committed `tailwind.css` artifact before editing (Phase 1's
  documented trap — new utility classes silently no-op without a rebuild);
  rebuilt with the pinned `tailwindcss@3.4.17`, verified via structural
  rule-diff (1 rule added — `.max-w-\[800px\]` — 0 removed), not raw text
  diff (single-line minified file, `git diff --stat` shows 1
  insertion/1 deletion since the whole line changes textually even for a
  1-rule addition).
- [x] **6.3** [S] Mac `.accessibilityLabel` bundle per §H3 inventory. (§H3)
  — Done: added `.accessibilityLabel(...)` reusing the exact same
  `String(localized:)` call already used for each control's `.help()`
  tooltip (VoiceOver doesn't read `.help()` text) on: favorite star,
  eject/unload, settings chevron, trash/delete (`ModelsScreen.swift`);
  add/browse/remove model-directory buttons (`ServerScreen.swift`);
  clear-stats trash (`StatusScreen.swift`, label switches between
  all-time/session variants matching the existing `.help()` ternary). No new
  i18n keys — every site had an existing key. `xcodebuild build` and `test`
  both green (290 tests, 1 pre-existing skip, 0 failures).
- [ ] **6.4** [M] Terminology sweep per §I1 tables (server states, model
  lifecycle, cluster machine noun, chat model-name consistency, "All Time"
  vs "All-Time"). (§I1)
- [x] **6.5** [S] "Stored in this browser" note next to persisted API-key
  inputs (`dashboard.js:721`; `chat.html:5236,3925-3929`). (§I2) — Done:
  added 2 new en.json-only keys (`bench.config.external_api_key_storage_note`,
  `chat.api_key_storage_note`) worded as a plain storage-location disclosure
  ("Stored in this browser.") rather than a network-transmission claim —
  the bench external-provider key *is* sent to the oMLX server (which
  proxies the benchmark request to the external endpoint server-side), so a
  "not sent to the server" wording would have been factually wrong. Wired
  under the bench external-API-key input (`_bench.html`) and under the chat
  API-key modal's input (`chat.html`).

---

## 12. Verified non-issues — do not re-investigate

Checked during the audit and found correct:

- **Dashboard:** PUT `/api/models/{id}/settings` request/response shapes match
  the UI exactly (incl. `requires_reload`/`auto_reloaded`/`auto_unloaded`
  alerts); `x-model.number` is used on all numeric modal inputs so the
  `Number.isFinite` guards work; bench SSE replay-dedupe correctly handles
  page-refresh re-attach; download/quantize/upload poll timers stop when
  queues drain; the restart poller's down-then-up transition logic is correct
  against the 202/503 backend contract; cluster incidents use a server-owned
  monotonic merge a failed poll cannot wipe; the log level filter's
  continuation-line handling (tracebacks follow their parent's visibility) is
  correct for the actual `%(asctime)s - %(name)s - %(levelname)s` format.
- **Chat security:** DOMPurify is applied on every HTML-injection path traced
  (renderMarkdown, streaming stable/tail roots, thinking blocks); SVG preview
  uses an allowlisted DOMPurify config behind an off-by-default toggle with a
  warning; web-card links restricted to `http(s)` via `webCardSafeUrl`; the
  stop button explicitly cancels the response reader for WebKit
  (`cancelStreamTransport`, chat.html:3834-3846).
- **i18n mechanics:** all 9 locales have full key parity (1122 keys) and zero
  `{placeholder}` mismatches (the actual problem is padding, §D1);
  `window.t(...) || fallback` occurs exactly once (§D2).
- **macOS:** double-click Load races are server-idempotent;
  `QuantizationScreenVM` does not clear `lastError` on poll (correct pattern
  for 2.2); menubar metrics correctly blank on stop
  (`MenubarMetricsStore.markServerStopped`); the LIV/AVG/ALL/PP/TG glyph tags
  are documented-intentionally unlocalized; the 0.6.2 cluster role
  reserve/flapping fixes hold for the plan-invalidation half — only the
  worker-role field regression in §B2 remains.
