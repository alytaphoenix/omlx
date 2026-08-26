# Dashboard model-config sync: keeping the settings UI honest

Design doc + phased implementation checklist for eliminating silent desync
between the admin dashboard's model-settings UI and the server's actual
per-model configuration. Every file:line reference below was **verified
against HEAD `b32bdaaa` (branch `deploy/session-fixes-v3`) on 2026-08-26**.
Line numbers will drift as the tree moves — treat them as anchors (the quoted
identifiers are the stable handles), and re-locate rather than trust a stale
number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here).

Findings are labeled **CONFIRMED** (the full causal chain was traced in code)
or hedged (**PLAUSIBLE** / judgment) where a claim is a design call rather
than a traced fact.

Status: **triggered by a real incident shape.** Settings were recently
deployed to two nodes by raw `PUT /admin/api/models/{id}/settings` calls over
SSH, bypassing the dashboard. Any dashboard page already open on those nodes
kept rendering — and would happily have *saved* — the pre-PUT values. Nothing
in the current code would have detected or even hinted at that.

---

## 1. Context

### What "desync" actually decomposes into

The request ("make the UI match the model config so they are not desynched")
covers two different problems with different owners:

1. **Single-node UI-vs-backend drift** — one server, one settings store, but
   the browser renders a stale copy and can write that stale copy back. This
   is a UI-sync problem and is the subject of this design.
2. **Cross-node parity** — coordinator and peer each have an independent
   settings store and an independent dashboard; nothing makes them agree.
   This is a config-replication problem, not a UI problem — see Theme D,
   which scopes it out with reasons.

### How `model.settings` actually gets to the screen today (traced)

- `GET /admin/api/models` (`admin/routes.py:1926 list_models`) embeds a
  **snapshot** of each model's persisted settings into the list response:
  `model_data["settings"] = asdict(settings)` (`routes.py:2043`), sourced
  from `ModelSettingsManager.get_all_settings()` (`model_settings.py:653`).
  The store itself is in-memory, persisted to
  `<base_path>/model_settings.json` (`model_settings.py:426`, manager
  constructed at `server.py:1936`).
- The dashboard calls `loadModels()` (`dashboard.js:7532`) **once at init**
  (`dashboard.js:941`) and after explicit mutations it performs itself:
  save (`:9295`), model load/unload (`:7603-7654`), reload (`:7555`),
  mtplx import (`:8951`), bench completion events (`:10366` etc.). There is
  **no timer and no focus/visibility handler that refreshes the models
  list** — the `visibilitychange` handlers at `dashboard.js:961` and `:995`
  only re-check bench state and stats/cluster polling. A dashboard tab left
  open renders `model.settings` as old as the last explicit action.
- `openModelSettings(model)` (`dashboard.js:8851`) builds the form via
  `buildModelSettingsState(model, model.settings || {})` (`:8905`) — i.e.
  **from the cached list snapshot, not a fresh read**. It does fetch fresh
  profiles, templates, parsers, and auto-context on open, but there is no
  fresh settings read to fetch: **no `GET /api/models/{id}/settings`
  endpoint exists at all** — the settings route surface is PUT-only
  (`routes.py:2225`).

### What `computeDrift()` is — and is not

`computeDrift()` (`dashboard.js:7742`) compares the **form** against the
**active profile's saved settings** (`_profileDrift` `:7753`,
`_profileDiverged` `:7770`, backend mirror at `routes.py:2753-2794`). It
drives the profile-drift dot. It has *nothing to do with* UI-vs-backend
staleness: both of its inputs live in the browser. Likewise
`_modelSettingsSnapshot` (`:8917`) is a **dirty-check baseline** for the
discard-confirm on close (`closeModelSettingsModal` `:8929`) — it detects
*the user's own unsaved edits*, not server-side changes. **No mechanism in
the codebase detects that the backend changed underneath the page.**

### Design posture

Reuse the house patterns already paid for: a version handshake carried on
existing responses (precedent: the `X-Omlx-Asset-Version` fetch-wrapper +
`assetStale` reload bar, `dashboard.js:1-30`, `_cluster.html:7`), fetch-fresh
at interaction points, and compare-and-reject at write time. No new push
infrastructure (see § Explicitly not doing). Correctness comes from the
write-time check; everything else is UX polish layered on top.

---

## 2. Theme A — The staleness window (what is actually broken)

### A1. `model.settings` is an unbounded-age snapshot — HIGH (CONFIRMED)

Chain: snapshot embedded at `routes.py:2043` → fetched only at init/explicit
actions (`dashboard.js:941`, `:9295`, ...) → modal form built from it
(`:8905`). Any backend write that this browser tab didn't itself perform —
a second tab, a raw API PUT, an ANE-tuner apply in another tab, a profile
apply — is invisible for the lifetime of the page. The modal can be opened
hours later showing values that were overwritten hours earlier.

### A2. A stale save silently clobbers newer state — HIGH (CONFIRMED)

The backend PUT has patch semantics — only fields in `model_fields_set` are
applied (`routes.py:2264` onward) — but `saveModelSettings()`
(`dashboard.js:9037`) sends **essentially every settings field explicitly**
(`:9097-9290`). So for every field the modal manages, a save is
last-writer-wins against whatever the modal happened to open with. Concrete
failure: open modal → someone PUTs `temperature: 0.4` via API → user toggles
an unrelated checkbox and hits Save → temperature silently reverts to the
stale open-time value. No version, no ETag, no `If-Match`, no conflict
response exists (`ModelSettingsRequest`, `routes.py:115`; persist via
`settings_manager.set_settings`, `:2872`; success response `:2936-2945`).

### A3. The concurrent writers are real, not hypothetical — CONFIRMED

Enumerated writers that mutate settings outside a given page's knowledge:

- Raw API PUTs (`routes.py:2225`) — the triggering incident.
- A second dashboard tab/browser (each caches its own `this.models`).
- ANE-tuner apply — `applyANETuningRecommendation` (`dashboard.js:8749`)
  PUTs a partial patch; run in tab A, tab B never learns.
- Profile apply — `POST .../profiles/{name}/apply` persists server-side
  (`routes.py:3158` → `apply_profile`, `model_settings.py:1167`).
- Quick list toggles — `updateModelSetting` (`dashboard.js:7570`) single-field
  PUTs for pin/default/hidden/favorite.
- Save auto-unload/reload side effects (`routes.py:2893-2934`) change
  *load state*, compounding what a stale page misrenders.

### A4. The dirty-check snapshot has false-positive bugs today — LOW (CONFIRMED)

Two flows persist server-side and update the form but **never refresh
`_modelSettingsSnapshot`**:

- `applyProfileToForm` rebuilds `modelSettings` from the server response
  (`dashboard.js:8323`) — snapshot still holds open-time state, so closing
  after a successful (already persisted!) apply raises the "discard unsaved
  changes?" confirm for nothing.
- `applyANETuningRecommendation` does `Object.assign(this.modelSettings,
  patch)` (`:8811`) after a successful PUT — same false-dirty on close.

Related corner: `importMtplxSidecar` (`:8937`) closes and reopens the modal
via `openModelSettings` (`:8953`), silently discarding any unsaved edits
without the confirm. These are cheap fixes and belong in Phase 1.

---

## 3. Theme B — Detection mechanism (what to build)

### B1. Write-time compare-and-reject is the one mechanism that is *reliable* — HIGH (design; mechanism CONFIRMED buildable)

Polling and push both shrink the staleness window; only a version check at
save time closes it. Mechanism:

- Add `settings_revision: int = 0` to the `ModelSettings` dataclass
  (`model_settings.py:83`). It free-rides through `to_dict`/`from_dict`,
  through `asdict(settings)` in the list response (`routes.py:2043`), and
  through the PUT response (`:2939`) — zero new serialization code.
- Bump it in `ModelSettingsManager` at every site that assigns
  `self._settings[model_id]`: `set_settings` (`model_settings.py:575`),
  `apply_profile` (`:1167`, which does **not** route through `set_settings`
  — it writes `self._settings[model_id]` directly at `:1203`), the
  profile-rename active-name touch-up (`:1118-1122`), and
  `delete_settings`. A `_touch(settings)` helper keeps the list greppable.
  The dataclass field must be stripped from *inbound* profile/preset merges
  so a stale profile dict can't rewind it.
- `ModelSettingsRequest` grows `expected_settings_revision: int | None`.
  When sent and `!= current_settings.settings_revision`, the PUT returns
  **409** with `{detail, current_settings}` and applies nothing. When
  omitted (raw curl users, older clients), behavior is unchanged —
  the check is opt-in by the sender, so this cannot break scripts.
- `saveModelSettings()` sends the revision the form was built from.

Timestamps were considered and rejected as the token: an int compared for
equality is unambiguous, and profiles already use `updated_at` for a
*display* concern (`model_settings.py:1043`) — different job. Note the
revision is per-model and lives in the same JSON file; direct hand-edits of
`model_settings.json` while the server runs bypass the manager entirely and
are out of scope (the manager loads the file at init and is the sole runtime
authority — a raw API PUT goes through it, a text editor does not).

### B2. Fetch-fresh-on-open kills the dominant window for free — HIGH (design)

Add `GET /api/models/{model_id}/settings` returning
`{model_id, settings: to_dict()}` (trivial: `_require_model` +
`get_settings`). `openModelSettings` awaits it inside the existing
`Promise.all` (`dashboard.js:8885`), threaded through the same
`seq`/`isCurrent()` race guard (`:8858`), and builds the form from the fresh
read — falling back to the cached `model.settings` snapshot only if the
fetch fails. Also fold the fresh settings back into `this.models` so list
badges (`matchedPreset`, profile pill) stop lying too. This alone would have
made the modal show the SSH-PUT values in the triggering incident.

### B3. Focus re-check while the modal is open — MEDIUM (design; house precedent CONFIRMED)

The remaining window is "modal open while someone else writes." Precedent:
the dashboard already re-checks bench state on `visibilitychange`
(`dashboard.js:961`). Add: while `showModelSettingsModal` is true, on
visibility→visible re-GET the single-model settings; compare
`settings_revision` against the revision the form was built from.

- Form clean (`JSON.stringify(this.modelSettings) ===
  this._modelSettingsSnapshot`): rebuild form + snapshot silently.
- Form dirty: show a non-blocking banner (Theme C2). Never silently merge
  into a dirty form.

This is advisory polish — B1 already guarantees no clobber even without it.

### B4. Rejected alternatives — no-action (judgment)

- **SSE/WebSocket push**: `EventSource` exists only as per-bench streams
  (`dashboard.js:10322`); there is no admin-wide event bus, and building one
  to invalidate a settings form is disproportionate to a modal that's open
  minutes at a time. B1+B2+B3 achieve the same integrity guarantee.
- **Continuous polling of `/api/models` while idle**: the response is heavy
  (full status + settings for every model) and the page mostly doesn't care;
  the existing design deliberately polls only stats/cluster tabs.
- **ETag/If-Match headers instead of a body field**: the house pattern for
  version handshakes is header-based (`X-Omlx-Asset-Version`), but that one
  rides a *wrapper over many call sites*; here exactly one endpoint and one
  caller are involved, and a body field survives `fetch` plumbing, JSON
  logging, and raw-curl reproduction with less ceremony. Judgment call.

---

## 4. Theme C — UI treatment when staleness is detected

### C1. On save conflict (409): block, explain, offer two exits — HIGH (design)

No silent anything at write time. On 409 the modal shows a conflict dialog
(house pattern: `confirm()`/alert usage is pervasive, but this warrants the
richer in-modal treatment like `profileError` `:8339`):

- **"Load latest"** (default): rebuild the form from the 409 response's
  `current_settings` (it's already in hand — no second fetch), refresh
  `_modelSettingsSnapshot`, keep the modal open so the user re-applies their
  intent on top of truth.
- **"Overwrite"**: resend with the *new* revision from the 409 payload —
  i.e. an explicit, informed last-writer-wins. Never resend without a
  revision; that would reintroduce the silent race the check exists to stop.

A field-level three-way merge (theirs/mine/base) was considered and
rejected — see § Explicitly not doing.

### C2. On focus-detected staleness with a dirty form: banner, not modal — MEDIUM (design)

An amber banner inside the settings modal — same visual grammar as the
`assetStale` reload bar (`_cluster.html:7`) — reading "Settings changed on
the server since this form was opened," with a single **Reload settings**
action (discards local edits after the existing confirm). No blocking: the
user may legitimately want to finish and overwrite, and C1 catches them at
save time anyway. Clean forms never see the banner (B3 refreshes silently).

### C3. Snapshot bookkeeping fixes ride along — LOW (CONFIRMED bugs)

Per A4: refresh `_modelSettingsSnapshot` (and the stored revision) after
`applyProfileToForm` success and after `applyANETuningRecommendation`
success; route `importMtplxSidecar`'s reopen through the dirty-check or
snapshot-refresh first. These make the dirty-check trustworthy, which C2/B3
depend on ("clean form" must mean clean).

---

## 5. Theme D — Cross-node parity: explicitly a different problem

**Out of scope for this design**, with reasons rather than hand-waving:

- Each node runs its own admin server, its own
  `ModelSettingsManager`, its own `model_settings.json` (`server.py:1936`).
  There is no transport between them for settings, and no unified
  multi-node dashboard exists. "The UI matches the config" is satisfiable
  per-node; making two *configs* match is replication, not rendering.
- For **distributed (cluster) serving, parity already doesn't depend on the
  peer's file**: request-time settings come from the coordinator's own
  manager (`get_settings_for_request`, `model_settings.py:548`), and the
  engine-construction fields that reach worker ranks are threaded from the
  coordinator's deployment plan (`cluster/deployment.py:258-270`, mtp fields
  onto every rank's launch argv). The hand-typed double-PUT was only ever
  necessary for the peer serving models **standalone** on its own `:8000`.
- If peer-standalone parity becomes a real need, the right shape is a
  deliberate cluster action ("push this model's settings to peer X", plus a
  `doctor`-style parity report), built on the enrollment/fabric channel —
  a separate doc. The `settings_revision` from B1 is the natural comparison
  key for that future check, which is the only coupling point worth noting.

---

## 6. Explicitly not doing (this design)

- **No admin-wide SSE/WebSocket event bus** for config invalidation (B4).
- **No background polling loop** for the models list on idle pages (B4).
- **No field-level three-way merge UI** on conflict. The settings form is
  ~60 heterogeneous fields with cross-field enable/disable coupling
  (`validateQwenAneSettings` `:8961`, diffusion sanitization
  `routes.py:525`); a merge UI would be large, rarely exercised, and easy to
  get subtly wrong. "Load latest, re-apply intent" is honest and small.
- **No cross-node settings replication** (Theme D — different problem,
  different doc).
- **No watching `model_settings.json` for external file edits.** The manager
  is the runtime authority; editing the file under a live server was never
  supported and stays that way.
- **No BroadcastChannel multi-tab sync.** Two tabs of the same browser are
  just the two-writers case; B1/B3 already handle them, and tab-coupling
  state adds a failure mode of its own.
- **No mandatory revision on the PUT.** Raw-API and script users keep
  working unchanged; the check activates only when the client sends
  `expected_settings_revision`.

---

## 7. Verified non-issues — do not re-investigate

- **The open-race guard works.** Two rapid "Settings" clicks are already
  handled by `seq`/`isCurrent()` (`dashboard.js:8858`, `:7819`, §B9 of the
  merged branch); a superseded response cannot land. B2's fresh settings
  fetch must simply thread through the same guard.
- **Backend patch semantics protect unsent fields.** `model_fields_set`
  gating (`routes.py:2264`) means fields the full-form save never sends —
  `is_pinned`/`is_hidden`/`is_favorite`/`is_default`, and
  `qwen35_ane_prefill_fused_down` (present in `buildModelSettingsState`
  `:8124` and the ANE apply patch `:8762`, absent from the save payload) —
  survive a stale save untouched. The clobber surface is exactly the fields
  the payload sends, which is what A2 describes.
- **The server never disagrees with itself.** All API writers funnel through
  the one locked manager (`model_settings.py:408`, `self._lock`); the
  incident's raw PUTs were fully consistent server-side. The only stale copy
  in the system is the browser's.
- **`computeDrift()` needs no changes for this design.** Profile drift is a
  browser-local comparison; once B2 feeds the form fresh settings, the
  existing call in `openModelSettings` (`:8912`) computes against truth
  automatically.
- **Save already refreshes the list.** `saveModelSettings` →
  `await this.loadModels()` (`:9295`), so post-save badge state is current;
  the gap is pre-save, not post-save.

---

## Phased implementation checklist

### Phase 0 — Revision plumbing (backend only, zero behavior change)

- [x] 0.1 Add `settings_revision: int = 0` to `ModelSettings`
      (`model_settings.py:83`); ensure `from_dict` tolerates absence
      (existing files) and inbound profile/template merges strip it.
- [x] 0.2 Bump via a `_touch` helper at every `self._settings[model_id]`
      assignment: `set_settings` (`:575`), `apply_profile` (`:1203`),
      profile-rename touch-up (`:1118-1122`), `delete_settings`.
- [x] 0.3 Confirm it flows through `GET /api/models` (`routes.py:2043`) and
      the PUT response (`:2939`) with no further changes; unit tests for
      bump-per-writer and file round-trip.

Shipped as jundot/omlx#3154 (branch `feat/model-settings-revision-check`,
built against `origin/main`). Merged locally into `deploy/session-fixes-v3`
(commit `fe07790c`, 2026-08-26) — not yet pushed upstream or deployed to
either node. `EXCLUDED_FROM_PROFILES` in `model_profiles.py` gained
`settings_revision` so a stored profile can't rewind it — the allowlist
already made this safe by construction, but a real completeness test
(`test_all_model_settings_fields_classified`) required the explicit entry.

### Phase 1 — Fresh-read on open + snapshot honesty

- [x] 1.1 Add `GET /api/models/{model_id}/settings` (mirror shape of the
      profile-apply response, `routes.py:3177`).
- [x] 1.2 `openModelSettings`: fetch it inside the `Promise.all`
      (`dashboard.js:8885`) under the existing `isCurrent` guard; build the
      form from the fresh read; fall back to `model.settings` on failure;
      write the fresh dict back into `this.models` for badge consistency;
      record the revision the form was built from.
- [x] 1.3 Fix A4: refresh `_modelSettingsSnapshot` (+ stored revision) after
      `applyProfileToForm` and `applyANETuningRecommendation` succeed; give
      `importMtplxSidecar`'s reopen the dirty-check or a snapshot refresh.
      **Fixed 2026-08-26 on `deploy/session-fixes-v3`.** Re-verified during
      this pass that `_modelSettingsSnapshot` is real on this deploy branch
      (shipped from the dashboard-a11y branch, PR #3143) and that PR #3154's
      merge left it out of sync in three places: `applyProfileToForm`,
      `applyANETuningRecommendation`, and — not originally listed here,
      found during this pass — `handleModelSettingsConflict`'s "Load latest"
      branch (same bug class: rebuilds `modelSettings` from a fresh
      server read without moving the dirty-check baseline). All four now set
      `this._modelSettingsSnapshot = JSON.stringify(this.modelSettings)`
      right after the rebuild. `importMtplxSidecar` now confirms
      (`modal.model_settings.discard_confirm`) before its reopen if the form
      is dirty, matching `closeModelSettingsModal`. Covered by
      `tests/test_admin_model_settings_template.py`
      (`test_apply_profile_refreshes_dirty_check_snapshot`,
      `test_apply_ane_tuning_refreshes_dirty_check_snapshot`,
      `test_mtplx_import_confirms_before_discarding_unsaved_edits`,
      `test_conflict_load_latest_refreshes_dirty_check_snapshot`).

### Phase 2 — Conflict-checked save (the actual guarantee)

- [x] 2.1 `ModelSettingsRequest.expected_settings_revision: int | None`;
      on mismatch return 409 with `{detail, current_settings}` before any
      field is applied (insert before the `sent` loop, `routes.py:2262`).
- [x] 2.2 `saveModelSettings` sends the form's revision; on 409 render the
      C1 dialog — "Load latest" rebuilds form+snapshot from the 409 body,
      "Overwrite" resends with the 409 body's revision. i18n strings for
      both paths. (Shipped as a `confirm()` dialog rather than a richer
      in-modal treatment — smaller diff, matches the pervasive confirm/alert
      house pattern noted in §C1.)
- [x] 2.3 Decide whether `updateModelSetting` quick toggles (`:7570`) also
      send a revision — recommendation: **no** (single-field intent, no
      stale-form to protect; a 409 on a pin click is worse UX than the
      idempotent toggle it prevents). Document the choice in code comment.
      (Decision recorded; left `updateModelSetting` untouched.)

### Phase 3 — Focus re-check banner (advisory; gate on Phases 1-2 shipped)

- [x] 3.1 `visibilitychange` → visible while `showModelSettingsModal`:
      re-GET single-model settings; clean form → silent rebuild; dirty
      form → C2 banner with "Reload settings" action. **Shipped
      2026-08-26 on `deploy/session-fixes-v3`.** Folded into the existing
      house `visibilitychange` listener (`dashboard.js`, the one gating on
      `document.hidden`) rather than a new listener — added
      `checkModelSettingsFreshness()`, called when the modal is open,
      independent of `mainTab`. Compares fresh `settings_revision` against
      `modelSettingsRevision`; identical → no-op; changed + form clean
      (`JSON.stringify(modelSettings) === _modelSettingsSnapshot`) → silent
      rebuild + snapshot refresh; changed + form dirty →
      `modelSettingsStaleBanner = true`, form left untouched.
      `reloadModelSettingsFromServer()` is the banner's action — same
      discard-confirm as `closeModelSettingsModal`, then reuses
      `openModelSettings()` for the actual reload rather than duplicating
      its fetch/rebuild logic.
- [x] 3.2 Reuse the `assetStale` bar's visual pattern (`_cluster.html:7`);
      banner clears on reload-settings, save, or modal close. **Shipped
      alongside 3.1.** New banner block in `_modal_model_settings.html`
      (amber, `data-model-settings-stale-bar`, mirrors the `assetStale`
      blue bar's structure) right below the modal header; two new en.json-only
      keys (`modal.model_settings.stale_banner`,
      `modal.model_settings.stale_banner_reload` — per §I18n-redundancy
      hardening, non-English locales omit them and fall back rather than
      storing English copies). `modelSettingsStaleBanner` cleared in
      `closeModelSettingsModal`, on successful save, and in
      `reloadModelSettingsFromServer`/`handleModelSettingsConflict`'s "Load
      latest" path. Covered by
      `tests/test_admin_model_settings_template.py`
      (`test_visibility_recheck_rebuilds_clean_form_and_banners_dirty_form`,
      `test_reload_settings_banner_action_confirms_before_discarding`,
      `test_close_and_save_clear_stale_banner`,
      `test_stale_banner_wired_in_template`,
      `test_stale_banner_i18n_keys_present`).

All phases now shipped. This design doc's tracked work is complete;
Theme D (cross-node parity) remains explicitly out of scope by design, not
deferred.
