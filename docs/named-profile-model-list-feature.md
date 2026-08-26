# Named profiles in the models list — feature design

Design doc + phased implementation checklist for: *"the ability in the UI to
easily name a profile to show up in the models list as that name."* Every
file:line reference below was **verified against HEAD `6e5c6d4f` on
2026-08-24**. Line numbers will drift as the tree moves — treat them as anchors
(the quoted identifiers are the stable handles), and re-locate rather than
trust a stale number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here).

---

## 1. Context — this feature is ~90% built already

The headline finding of the pre-design survey: oMLX already has a complete
**"exposed profiles"** mechanism, end-to-end, from persistence through routing
to `/v1/models`. What the user is asking for is the last mile: the *naming and
exposing flow in the dashboard is buried and two-step*, exposed profiles are
*invisible as entries in the admin models tables*, and the exposed identity is
always the namespaced `<base>:<api_name>` rather than literally "that name".
This is a completion-and-polish design, not greenfield.

What already exists, verified at HEAD:

- **Data model.** `ModelProfile` carries `name`, `display_name`, `api_name`
  (`omlx/model_profiles.py:143-150`), with a slug validator
  (`validate_profile_name`, :220-233) and a slugifier for deriving `api_name`
  from user-facing text (`slugify_profile_api_name`, :236-247). Profile
  records persisted by the settings manager additionally carry
  `expose_as_model: bool` (`omlx/model_settings.py:1037-1048`). Profiles are
  stored per base model in `model_profiles.json` (versioned, atomic write,
  `omlx/model_settings.py:427,723-738`).
- **Exposed model identity.** An exposed profile is addressable as
  `<base_model_id>:<api_name>` and `<base_alias>:<api_name>`
  (`_profile_model_id` / `_display_profile_model_id_locked`,
  `omlx/model_settings.py:783-798`; full accepted-ID set in
  `_profile_request_ids_locked`, :905-915). `api_name` is auto-deduped per
  model (`_dedupe_profile_api_name`, :740-751).
- **Listing.** `/v1/models` appends exposed profiles as their own entries
  (`omlx/server.py:2919-2938`), and `/v1/models/status` augments them with
  base status + `source_model_id`/`profile_*` metadata
  (`_with_exposed_profile_status`, `omlx/server.py:2646-2685`). The admin
  `/admin/api/models` response attaches an `exposed_profiles` array to each
  base model (`omlx/admin/routes.py:1969-1972`).
- **Routing + loading.** A request for a profile ID resolves to the base
  model with the profile's *universal* fields overlaid at request time
  (`get_settings_for_request`, `omlx/model_settings.py:548-573`) and its
  *engine-construction* fields applied via a transient engine-variant reload
  that never mutates persisted base settings
  (`get_exposed_profile_runtime_settings_for_request`,
  `omlx/model_settings.py:853-877`; consumed in `omlx/server.py:1105-1130`;
  variant reload via `runtime_settings_signature` compare in
  `EnginePool.get_engine`, `omlx/engine_pool.py:1416-1457`).
  `resolve_model_id` resolves profile IDs before aliases
  (`omlx/engine_pool.py:1078-1086`). Even `get_max_context_window` already
  reports the profile's overridden context window for a profile ID
  (`omlx/server.py:1745-1782` → `get_settings_for_request`).
- **Conflict guards.** Both directions exist: saving an exposed profile
  validates its candidate IDs against directory names, model aliases, and
  other exposed profiles (`_validate_exposed_profile_ids_available_locked`,
  `omlx/model_settings.py:917-957`, with `reserved_model_ids` supplied from
  the engine pool at `omlx/admin/routes.py:3002-3004`); and setting a model
  alias validates against exposed profile IDs
  (`_raise_if_alias_conflicts_exposed_profiles`,
  `omlx/admin/routes.py:2923-2967`, called from :2211-2216;
  `_raise_if_profile_id_conflicts_model_id`, :2906-2920).
- **UI (partial).** The model-settings modal has a profiles section with a
  create form (display name + auto-slugged API name,
  `omlx/admin/templates/dashboard/_modal_model_settings.html:214-229`) and an
  edit dialog with an `API ON/OFF` expose toggle (:246-255). The settings-tab
  models table shows exposed profiles as small pills under the base model
  (`omlx/admin/templates/dashboard/_settings.html:1575-1588`). The chat
  playground picker is fed by `/v1/models/status`
  (`omlx/admin/templates/chat.html:4006`), so exposed profiles **already
  appear there**.

The actual gaps, in order of how directly they answer the request:

1. **Exposing is not part of the create flow.** `createProfile()` in
   `omlx/admin/static/js/dashboard.js:7525-7568` never sends
   `expose_as_model` (the backend `CreateProfileRequest` accepts it,
   `omlx/admin/routes.py:194-205`). Today the user must create the profile,
   find the tiny pencil icon, open the edit dialog, toggle `API`, and save —
   the opposite of "easily".
2. **Exposed profiles are not entries in the admin models lists.** The
   settings-tab table (`sortedModels`, `_settings.html:1385`) shows them only
   as pills in the chips row; the Models→Manager list
   (`sortedManagerModels`, `_models.html:124`) has zero profile awareness;
   `_status.html` shows only physical engine entries.
3. **"As that name" is only half true.** The models-list entry is
   `<base|alias>:<api_name>`. The name the user typed appears as the suffix,
   never standing alone. `/v1/models`'s OpenAI schema has no display-name
   field — the ID *is* the name — so if the user literally wants the entry to
   read `coder`, that requires a standalone exposure ID (§3, Theme C, gated
   on Open Question 1).

---

## 2. Data model — the name triad, and what "the name" means

A profile record already has **three** names; the design must be precise
about which one "shows up":

| Field | Set by | Constraints | Where it appears today |
|---|---|---|---|
| `name` | auto-generated by the UI (`'p-' + Date.now()...`, `dashboard.js:7539-7540`) | slug (`validate_profile_name`) | internal key only: URLs of the CRUD endpoints, `active_profile_name` |
| `display_name` | typed by the user in the create form | free text | pill labels in the modal and settings tab; `profile_display_name` in `/v1/models/status` |
| `api_name` | auto-slugged from `display_name`, editable (`_modal_model_settings.html:217-220`) | slug, deduped per model | the exposed-ID suffix: `<base>:<api_name>` in `/v1/models` |

Plus `expose_as_model: bool` (default false) gating all of it.

**Decision D1 — no new fields are needed for Phases 1-2 (UX + visibility).**
The triad composes: `api_name` is "the name" for API identity,
`display_name` is "the name" for dashboard labels. The feature request is
satisfiable by surfacing what exists.

**Decision D2 — one optional new field for Phase 3 (standalone identity),
gated on OQ1:** `exposed_model_alias: str | null` on the profile record — a
profile-level analog of `ModelSettings.model_alias` (which is deliberately
excluded from profile *settings*, `EXCLUDED_FROM_PROFILES`,
`omlx/model_profiles.py:93-108` — that exclusion is about not copying the
*base model's* alias through profiles and stays untouched; this is a new
top-level record key, not a settings key). Semantics mirror the base-model
alias exactly: when set (and `expose_as_model` is true), the profile is
listed under the alias and *additionally* remains accepted under its
namespaced forms, just as a base model's directory name remains accepted
alongside its alias (that precedent is documented in
`_display_profile_model_id_locked`'s docstring,
`omlx/model_settings.py:789-795`).

No other persisted shape changes. `model_profiles.json` stays at
`PROFILES_VERSION` for Phases 1-2; Phase 3 adds one optional key that old
readers ignore and the existing loader defaults to absent (see §6).

---

## 3. Design

### Theme A — one-step "name it and it's in the models list" (create flow)

The create form (`_modal_model_settings.html:214-229`) gains the same
`API ON/OFF` expose toggle that the edit dialog already has (:246-255), and
`createProfile()` (`dashboard.js:7525-7568`) sends
`expose_as_model: this.newProfile.expose_as_model` in the POST body (add the
field to the `newProfile` reset objects at `dashboard.js:281,7558` and the
form-open reset at `_modal_model_settings.html:207`). Backend: zero changes —
`CreateProfileRequest.expose_as_model` (`omlx/admin/routes.py:205`) and
`save_profile(...)` (`omlx/model_settings.py:1009-1055`) already accept and
validate it, returning 409 with a human-readable detail on collision, which
the form already renders (`profileError`, `dashboard.js:7563-7566`).

Interaction detail: when the toggle is ON, show the resulting model ID live
under the API-name input — `<alias-or-dir>:<api_name>` — computed client-side
from `selectedModel.settings?.model_alias || selectedModel.id` plus the
slugged `api_name`. This is the single most effective "easily" improvement:
the user sees exactly what string will appear in the models list *before*
saving.

Second affordance (small, optional but recommended): make the `API` badge on
each profile pill (`_modal_model_settings.html:181-183`) clickable as a
direct expose/unexpose toggle (PUT with `{expose_as_model: !current}` via the
existing `updateProfile` path, `dashboard.js:7701`), so exposing an existing
profile no longer requires the edit dialog either.

Note the deliberate orthogonality to keep in the UI copy: clicking a pill
*applies* the profile to the base model's form/settings (the
`.../profiles/{name}/apply` endpoint, `omlx/admin/routes.py:3084-3105`,
mutates persisted base settings); *exposing* publishes it as a separate model
entry and never mutates the base. These are independent, and the current UI
does not explain that anywhere — one tooltip/hint line each.

### Theme B — exposed profiles as rows in the admin models lists

The data is already delivered: every model object from `/admin/api/models`
carries `exposed_profiles` (`omlx/admin/routes.py:1969-1972`), each profile
dict already containing `model_id` (the advertised exposed ID, injected by
`list_profiles` → `_display_profile_model_id_locked`,
`omlx/model_settings.py:992-1005`), `display_name`, `api_name`,
`has_engine_fields`. Rendering is purely client-side; no new endpoints.

- **Settings-tab models table** (`_settings.html:1385` `sortedModels` loop):
  after each base-model row, render one indented child row per
  `model.exposed_profiles` entry. Label = `display_name`, monospace sub-label
  = the exposed `model_id`, plus a distinct badge (reuse the emerald `API`
  pill style, `_settings.html:1580-1587`; amber-accented when
  `has_engine_fields` — the existing color convention meaning "selecting this
  can trigger an engine variant reload", `_has_engine_fields`,
  `omlx/model_settings.py:978-990`). Row actions: edit (opens the base
  model's settings modal with that profile's edit dialog pre-opened) and
  expose-off. Child rows are display-only composites — favorite/default/
  hidden toggles stay base-model-only (those fields are excluded from
  profiles by design, `omlx/model_profiles.py:93-108`).
- **Models→Manager list** (`_models.html:124`): same child-row treatment,
  minus settings chips. Lower priority than the settings table (OQ2).
- **Status tab**: leave as-is. It reflects physical engines; an exposed
  profile is not a separately loadable engine (see "loaded" caveat below).

Visual distinction rule, applied everywhere: profile rows always show the
`↳` indentation + `API` badge + base-model attribution ("via
`<base display name>`"), so they can never be mistaken for a physical model
directory.

**What happens when the user selects one (chat / API):** already implemented
— the request routes to the base model, universal fields overlay at request
time (`omlx/model_settings.py:548-573`), and if the profile carries
engine-construction fields the pool unloads/reloads the base engine as a
transient variant (`omlx/engine_pool.py:1416-1457`). A subsequent request for
the *base* ID with different persisted settings flips the variant back — the
signature compare is symmetric. The dashboard needs no new load path; a
"chat" action on a profile row just deep-links the chat page with that model
ID preselected (the picker already lists it via `/v1/models/status`).

**Loaded-state fidelity caveat (fix in Phase 2):**
`_with_exposed_profile_status` copies the base row wholesale
(`omlx/server.py:2673-2680`), so a profile row reports `loaded: true`
whenever the base engine is resident — even when the resident variant was
loaded with *different* engine-construction settings and selecting the
profile would actually trigger an unload/reload. Fix: have the status
augmentation compare the entry's `runtime_settings_signature`
(`omlx/engine_pool.py:238`, set at :1456-1457,2690) against the signature the
profile would produce, and emit `loaded: true` + e.g.
`variant_active: false` when they differ (UI renders "loaded (base variant)").
Requires exposing the signature or a boolean through `get_status` — small,
contained.

### Theme C — standalone exposure ID ("literally that name") — gated on OQ1

If OQ1 resolves to "the entry must read exactly the typed name", implement
`exposed_model_alias` (D2). The mechanism composes cleanly with four touch
points, all in `omlx/model_settings.py`:

1. `_profile_request_ids_locked` (:905-915): add the alias to the returned
   set. This **automatically** propagates it into
   `get_exposed_profile_model_ids` (:879-903), into save-time validation
   (`_validate_exposed_profile_ids_available_locked`, :917-957 — directory
   names via `reserved_model_ids`, model aliases at :937-942, other profiles
   at :944-957), and into the admin alias-conflict guard
   (`omlx/admin/routes.py:2930-2934`). One caveat: the reserved-IDs loop
   deliberately skips candidates equal to the profile's own base
   (`candidate_id != model_id`, :930-931) — moot for namespaced IDs, but a
   standalone alias set to **the base's own directory name** would slip
   through and, because profile resolution precedes directory lookup
   (`omlx/server.py:1108-1120`), shadow the base for every request. Phase 3
   must drop that self-exclusion for standalone aliases (the base's own
   `model_alias` is already caught by the :937-942 loop).
2. `_find_exposed_profile_locked` (:800-814): match the alias in addition to
   the namespaced forms (routing + request-time settings then just work).
3. `_display_profile_model_id_locked` (:786-798): prefer the alias as the
   advertised ID, mirroring how the base model's alias is preferred for
   display.
4. Persistence + API: optional `exposed_model_alias` key in the profile
   record (`save_profile`/`update_profile` params,
   :1009-1055/:1057-1140) and on `CreateProfileRequest`/
   `UpdateProfileRequest` (`omlx/admin/routes.py:194-218`); one optional
   input in the modal's edit dialog and create form.

The base-alias-change guard (`_raise_if_alias_conflicts_exposed_profiles`,
`omlx/admin/routes.py:2938-2967`) needs no change for the recomputation loop
— standalone aliases don't depend on the base alias — and its
`exposed_ids` membership check at :2930 picks up standalone aliases for free
via touch point 1.

Charset: recommend the model-ID-compatible superset of the profile slug —
lowercase slug plus `.` (model directories legitimately contain dots, e.g.
`qwen3.5-*`), explicitly rejecting `:` and `/` (both are structurally
meaningful in request-ID candidate splitting,
`omlx/model_settings.py:844-846,864-866`). Exact rule is OQ1b.

Recommended default regardless of OQ1: keep the namespaced form as the
default behavior; standalone naming is opt-in per profile. Namespacing is
what makes the collision story tractable and self-explanatory in client
model pickers.

### Theme D — small fidelity items

- `/v1/models` fidelity for profile entries is already good:
  `max_model_len` resolves through the profile overlay
  (`omlx/server.py:2934`, `get_max_context_window` → 
  `get_settings_for_request`). No work.
- Hidden/favorite semantics are inherited from the base: hidden base models
  exclude their profiles from `/v1/models` (`excluded_model_ids` check,
  `omlx/server.py:2926-2929`). Keep — a hidden base hiding its presets is
  the least surprising rule. Favorites-first sorting (:2946-2947) does not
  apply to profile entries (they're appended after the base loop and their
  IDs aren't in `favorite_ids`); acceptable, or trivially include profile
  IDs of favorited bases.
- i18n: every new label/tooltip needs keys in all nine locale files under
  `omlx/admin/i18n/` (`en.json` + 8 translations), following the existing
  `modal.model_settings.profiles.*` / `settings.models.table.*` namespaces.

---

## 4. Edge cases

- **Name collisions.** Fully guarded at save time in both directions today
  (§1 bullet "Conflict guards"); Phase 3 extends coverage automatically via
  `_profile_request_ids_locked` (§3 Theme C). One genuine hole, pre-existing
  and worth closing in Phase 3: `reserved_model_ids` is checked only at
  profile-save time, so a **newly downloaded model directory** can later
  collide with an existing exposed-profile ID (esp. a standalone alias).
  Resolution order actually *shadows the real model* — profile resolution
  runs before directory/alias lookup (`omlx/server.py:1108-1120`,
  `omlx/engine_pool.py:1078-1086`). Fix: warn loudly at
  discovery/registration time when a new entry's ID is claimed by an exposed
  profile (log + a badge in the models table), rather than trying to
  auto-resolve.
- **Base model deleted.** `delete_settings` removes the model's whole
  profile map (`omlx/model_settings.py:602-626`); exposed entries vanish
  from `/v1/models` on the next list because listing skips profiles whose
  `source_model_id` isn't a live physical model (`omlx/server.py:2924-2930`,
  `2665-2668`). Already correct.
- **Base model directory renamed** (outside oMLX): profiles are keyed by
  `model_id` = directory name, so they orphan exactly like per-model
  settings do. Orphaned profiles are inert (never listed — same
  `source_model_id not in physical_ids` skip). Document; don't build rename
  migration for this feature.
- **Base settings change / reload while a profile variant is loaded.** The
  signature compare in `get_engine` handles every direction: profile request
  after base edit reloads the variant; base request after profile use
  reloads the base variant; profile edit changes the computed signature so
  the next profile request reloads. The only UX consequence is the
  loaded-state caveat (§3 Theme B fix). Note reload requests are refused
  while the entry is busy (`_raise_if_reload_busy`,
  `omlx/engine_pool.py:1429-1432`) — surfaced to clients as 409, already the
  established behavior for settings-triggered reloads.
- **Multiple profiles per model.** Already supported: `api_name` dedupe is
  per-model (`_dedupe_profile_api_name`, :740-751; migration dedupe at
  :685-713), each exposed independently; two profiles selecting different
  engine variants of one base simply ping-pong the variant reload if used
  concurrently — same cost model as today, worth one sentence in user docs.
- **Profile rename.** Internal `name` rename already keeps
  `active_profile_name` in sync (`omlx/model_settings.py:1119-1127`).
  Changing `api_name` (or a Phase-3 alias) changes the public model ID —
  in-flight clients get a 404-with-available-models on the old ID
  (`omlx/server.py:1152-1170`). Acceptable; mirror of alias changes today.

---

## 5. Migration / compatibility

Short and positive:

- **No shape change for Phases 1-2.** `expose_as_model` already exists;
  records without it read falsy (`profile.get("expose_as_model")`
  everywhere, e.g. `omlx/model_settings.py:807,892,925,947`). Profiles
  created before this feature keep behaving identically.
- **`api_name` backfill already shipped.** `_load_profiles` migrates legacy
  records (validating/slugifying/deduping, `omlx/model_settings.py:679-718`)
  under the existing `PROFILES_VERSION` with a warning-not-refusal version
  check (:674-678).
- **Phase 3** adds one optional record key (`exposed_model_alias`) — absent
  key means current behavior; no version bump strictly required, though
  bumping `PROFILES_VERSION` with the same warning-only check is cheap and
  keeps the file self-describing. Older builds reading a newer file ignore
  the key harmlessly (loaders use `.get`).
- **API compatibility.** New request fields are optional on
  `CreateProfileRequest`/`UpdateProfileRequest`; `/v1/models` and
  `/v1/models/status` shapes are unchanged in Phases 1-2 (Phase 2 adds one
  additive boolean to status rows).

---

## 6. Open questions / decisions needed

- **OQ1 — RESOLVED (2026-08-24): namespaced form.** The models-list entry
  reads `<base-or-alias>:<api_name>` (e.g. `qwen3.5-27b:coder`, or with a
  short base alias `q35:coder`) — the scheme already implemented today, not
  the standalone-alias alternative. **Phase 3 (§3 Theme C, the standalone
  exposure ID) is therefore out of scope** — Phases 1-2 are the full
  implementation of this feature. OQ1b is moot.
- **OQ2: which admin surfaces get profile child rows?** Recommendation:
  settings-tab models table definitely; Models→Manager list probably;
  Status tab no. Cheap to add later; agreeing up front avoids churn.
- **OQ3: should a profile row offer explicit "load now"?** Recommendation:
  no dedicated load button in v1 — selection-on-use (chat deep-link / first
  API request) matches how base models behave in the chat flow, and a load
  button would need the whole busy/409 handling UI. Revisit if users ask.
- **OQ4: loaded-state display semantics** for profile rows once the Theme B
  fix lands: is "loaded (base variant)" the right rendering, or should
  variant mismatch show as not-loaded? Recommendation: show loaded with the
  variant qualifier — the memory is genuinely occupied either way.

---

## 7. Phased implementation checklist

Ordering: user-visible value first; everything in Phase 1 is low-risk and
backend-free. Effort tags: S(<~1h) / M(half-day) / L(multi-day). Line refs
verified 2026-08-24 @ `6e5c6d4f`; re-grep the quoted identifiers if executing
much later.

**Status: Phases 1, 2, and 4 shipped 2026-08-26 on `deploy/session-fixes-v3`**
(local, uncommitted upstream, not deployed to either node). Phase 3 is
correctly **out of scope**, not merely deferred — OQ1 resolved on the
namespaced-ID scheme before any of this was built, so there is no standalone
`exposed_model_alias` to implement. Left in this doc verbatim as the design
record for *why* it's out, per house convention (see §6 OQ1).

### Phase 1 — one-step expose + live ID preview (no backend changes)

- [x] **1.1** [S] Add `expose_as_model` toggle to the new-profile form and
  include it in the `createProfile()` POST body plus all three `newProfile`
  reset sites (initial data, template form-open reset, post-create reset).
  Backend already accepted it, unchanged. (§3A)
- [x] **1.2** [S] Live exposed-ID preview under the API-name input while the
  toggle is ON, reading `selectedModel?.settings?.model_alias ||
  selectedModel?.id` plus the live-typed (or auto-slugged) API name. (§3A)
- [x] **1.3** [S] Made the `API` badge a direct expose/unexpose toggle —
  restructured it out of the apply button (it was a non-interactive `<span>`
  nested inside `applyProfileToForm`'s `<button>`, which can't be made
  independently clickable without breaking nesting/a11y) into its own
  sibling `<button>` with `aria-pressed`/`aria-label`, wired to a new
  `toggleProfileExpose(p)` → `updateProfile(p.name, {expose_as_model:
  !p.expose_as_model})`. Errors surface via the existing `profileError`
  slot for free (shared `updateProfile` path). (§3A)
- [x] **1.4** [S] Two new hint strings distinguishing *apply* (loads into
  the form, not saved until Save) from *expose* (publishes a separate model
  entry, never mutates base): `apply_hint` prepended to the pill's existing
  `profileTooltip()`, `expose_hint` on both API-toggle buttons — plus the
  pre-existing but previously-orphaned `expose_engine_fields_hint` key now
  appended to the pill toggle's tooltip specifically when
  `has_engine_fields` is true. en.json only; other locales fall back
  (§I18n-redundancy hardening — a stored English copy is dead weight, see
  docs/dashboard-model-config-sync.md). (§3A, §3D)

### Phase 2 — profiles as entries in the admin models lists

- [x] **2.1** [M] Settings-tab table: indented child row per
  `model.exposed_profiles` entry after each base row — `display_name`
  label, monospace exposed `model_id`, `API` badge (amber when
  `has_engine_fields`), "via `<base>`" attribution, chat/edit/expose-off
  actions. Replaced (not kept alongside) the old read-only chip in the
  settings-summary row — a genuine row now, not a chip pretending to be
  one. Edit opens the base model's settings modal via a new
  `editExposedProfileFromList(model, profile)`, which calls
  `openModelSettings()` then matches the freshly-reloaded profile by name
  and pre-opens its edit dialog. Expose-off is a new standalone
  `unexposeProfileFromList(model, profile)` (doesn't require the modal to
  already be open — `updateProfile()` does). (§3B)
- [x] **2.2** [S] Chat deep-link: profile rows link to
  `/admin/chat?model=<id>`; `chat.html`'s `init()` reads `?model=`, starts
  a fresh chat preselecting it (via the existing
  `isModelAvailableOnServer`/`resolveGatewayModelId` helpers, which already
  handle profile IDs since they're just entries in `availableModels`), then
  strips the param via `history.replaceState` so a reload doesn't keep
  forcing a new chat. No prior URL-param handling existed in chat.html;
  this is new, additive. (§3B)
- [x] **2.3** [M] Models→Manager child rows — same treatment as 2.1 minus
  the settings-summary chips this list doesn't have, sourced via
  `managerModelInfo(model.name)?.exposed_profiles` (cross-references the
  admin models list `sortedManagerModels` doesn't itself carry). (§3B)
- [x] **2.4** [M] Loaded-state fidelity: new `_apply_profile_variant_fidelity()`
  helper in `server.py`, shared by both `_with_exposed_profile_status`
  (`/v1/models/status`) and `admin/routes.py`'s `list_models` (deferred
  import to dodge the circular import — `server.py` imports FROM
  `admin.routes`, matching the existing `from ..server import
  _server_state` pattern already used throughout that module). Resolves
  the profile's merged runtime settings via
  `get_exposed_profile_runtime_settings_for_request`, computes the expected
  signature via the pool's own `_engine_runtime_signature`, and sets
  `variant_active: False` on mismatch (never `True` — absent means no
  claim, so callers that don't know the field see no behavior change).
  Settings-tab and Manager child rows render "loaded (base variant)" when
  `model.loaded && profile.variant_active === false`, plain "loaded" (base
  loaded, variant matches) otherwise. Per OQ4. (§3B)

### Phase 3 — standalone exposure ID — OUT OF SCOPE (OQ1 resolved against it)

Left unimplemented and unstarted, deliberately. OQ1 (§6) resolved on the
namespaced `<base>:<api_name>` scheme *before* Phase 1/2 work began — the
items below describe what standalone `exposed_model_alias` would have
required, kept as the design record in case the decision is ever revisited,
not as pending work.

- [ ] **3.1** [M] `exposed_model_alias` on the profile record + validation:
  extend `_profile_request_ids_locked` (`model_settings.py:905-915`) —
  which auto-propagates into `get_exposed_profile_model_ids` (:879-903),
  save-time validation (:917-957), and the admin alias guard
  (`admin/routes.py:2930-2934`) — plus matching in
  `_find_exposed_profile_locked` (:800-814) and display preference in
  `_display_profile_model_id_locked` (:786-798). Charset per OQ1b. Must
  reject an alias equal to any live model directory ID **including the
  profile's own base** (the `candidate_id != model_id` self-exclusion at
  :930-931 must not apply to standalone aliases). (§3C)
- [ ] **3.2** [S] API + UI plumbing: optional field on
  `CreateProfileRequest`/`UpdateProfileRequest`
  (`admin/routes.py:194-218`), `save_profile`/`update_profile` params
  (`model_settings.py:1009-1140`), one input in the modal create/edit
  forms; optional `PROFILES_VERSION` bump (warning-only check at
  :674-678). (§3C, §5)
- [ ] **3.3** [S] Discovery-time collision warning when a new model
  directory/alias matches an exposed-profile ID (profiles currently shadow
  it: `server.py:1108-1120`, `engine_pool.py:1078-1086`); log + models-table
  badge, no auto-resolution. (§4)

### Phase 4 — polish / deferred

- [x] **4.1** [S] Exposed-profile IDs of a favorited base now sort with it
  in `/v1/models`'s favorites-first sort — a `base_display_id_by_source`
  map built during the base-model pass lets the profile-appending pass look
  up whether its source's *display* ID (post-alias) was favorited. (§3D)
- [x] **4.2** [S] User-docs paragraph shipped in `README.md`'s Profiles
  bullet: corrected the previous unqualified "no extra memory, no reload"
  claim (only true when the profile doesn't touch engine-construction
  fields) and added the variant ping-pong cost for interleaved profiles
  that disagree on those fields. (§4)
- [x] **4.3** [S] Same `README.md` paragraph: directory-rename orphaning
  documented ("profiles simply stop appearing... nothing to migrate, they
  just resume working if the original name comes back"). (§4)
