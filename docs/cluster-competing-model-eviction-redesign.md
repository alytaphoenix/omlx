# Cluster competing-model eviction — redesign per maintainer feedback

Design doc for reworking the feature originally shipped in closed PR
[#2870](https://github.com/jundot/omlx/pull/2870) (`feat(cluster): evict
competing local models after a memory-attributed activation failure`),
addressing the two objections the maintainer raised when closing it. Verified
against HEAD `95f4e294` (branch `deploy/session-fixes-v2`) on 2026-08-25. Line
numbers are anchors — re-locate rather than trust a stale number if the tree
has moved.

Status: **implemented** on `deploy/session-fixes-v2` (2026-08-25) — settings,
structured marker fields, `RankFailure`/`DistributedLaunchError.rank_failures`,
the peer-side eviction script, routes.py orchestration, the admin toggle, and
25 tests (`tests/test_cluster_local_eviction.py`), all matching §3 below as
designed except where noted in the implementation-discovered addendum in §4.
Shipped without C5/admin_port (unmerged) — the §3.4 fallback path: the peer
reads its own `settings.json` port, no guess-fallback.

---

## 1. Why #2870 was closed

Maintainer's closing comment, in full:

> Thanks, the ssh and token mechanics here are careful work. Closing this one
> though: a server-initiated kill of models a user loaded on their own Mac is
> a product call I'm not making right now, and a regex over free-form error
> text is too fragile a base for a destructive action even behind a gate. If
> this comes back later it will need to be opt-in and keyed on structured
> failure attribution, rank and node as fields rather than parsed prose. The
> token minting pattern and C5's advertised admin_port are both worth keeping
> for that day.

Four concrete requirements fall out of this:

1. **Opt-in.** The feature must not run unless a user has explicitly enabled
   it. Not "on by default with an escape hatch" — off by default, full stop.
2. **No regex over free-form error text.** The mechanism that decides *which
   node(s)* get evicted must consume real structured fields (rank, node ID,
   error type), not pattern-match a human-readable string built for a log
   line or an HTTP error detail.
3. **Keep the SSH/token-minting pattern.** The mechanics for reaching a
   peer's admin API without a coordinator→peer HTTP route (SSH in, mint a
   session cookie from the peer's own persisted secret, hit `127.0.0.1`
   loopback) were explicitly called out as sound and worth reusing.
4. **Keep/use C5's advertised `admin_port`.** [`fabric-doctor
   C5`](https://github.com/jundot/omlx/pull/2885) (open, depends on
   [C2](https://github.com/jundot/omlx/pull/2875), also open) adds
   `ClusterStatus.admin_port`, populated from each node's own
   `settings.server.port` and propagated through capability probes — the
   structural answer to "how does the coordinator know which port a peer's
   admin API is listening on," which #2870's peer script solved by reading
   the peer's `settings.json` directly instead.

The rest of this doc addresses each of these against the actual current
codebase.

---

## 2. Root cause of the regex problem (traced, not assumed)

The maintainer's objection is specific and it's worth confirming the exact
mechanism that produced it, because the fix is narrower than "add a
structured exception type" — the structured data mostly already exists and
gets thrown away one hop before the code that needed it.

**The rank-side worker already writes a structured JSON marker per attempt.**
`RuntimeMarker` (`omlx/cluster/inference_worker.py:288-352`) persists
`{deployment_id}-rank-{rank}.json` with a real `rank: int` field baked into
the payload at construction (`inference_worker.py:308`) and a free-form
`**extra` kwarg dict merged in on every `.update(phase, **extra)` call — the
schema is already open for new fields, no format migration needed.

**The failure path collapses the one bit that matters into a string, right
where it's raised.** `inference_worker.py:1241-1250`:

```python
except Exception as exc:
    preserve_failure_marker = True
    with suppress(Exception):
        marker.update(
            "failed",
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )
    raise
```

`type(exc).__name__` — the exact structured type the maintainer wants — is
sitting in a local variable at this exact line, and is discarded into an
f-string before it's written. `InsufficientMemoryError` itself
(`omlx/exceptions.py:536-542`) already carries `.required`/`.current` as real
int attributes, also discarded.

**The launcher then re-flattens the (already partially-lost) marker into one
big joined string.** `DistributedJobSupervisor._runtime_failure_reason()`
(`omlx/cluster/launch.py:2137-2178`) reads every rank's marker — `rank` and
`node_id` are right there as `marker.get("rank")` / `host.node_id` — and
joins them into `f"rank {rank} ({host.node_id}): {error.strip()}"` per
failure, `"; ".join(...)`-ed into one string. That string becomes
`DistributedLaunchError`'s message (a plain `RuntimeError` subclass with no
structured payload at all — `launch.py:50`, raised from ~30 call sites with
bare string messages), which becomes the HTTP 503 `detail`.

**#2870's `routes.py` then regexed that string back apart** to recover the
`rank`/`node_id` it needed
(`_MEMORY_FAILURE_RANK = re.compile(r"rank (\d+) \(([^)]+)\):\s*InsufficientMemoryError")`),
plus a bare substring check (`"InsufficientMemoryError" in detail`) standing
in for "was this a memory failure."

So the actual bug isn't "no structured data exists" — it's a three-hop
game of telephone: real fields → stringified once (rank-side, losing
`error_type`) → stringified again (launcher, losing the marker's own
`rank`/`node_id` fields as *fields*) → parsed back apart with a regex
(routes.py). Fix each hop instead of adding a new side-channel.

---

## 3. Design

### 3.1 Opt-in setting

New `ClusterSettings` dataclass in `omlx/settings.py`, following the exact
pattern every other settings group there already uses (`ServerSettings`,
`MemorySettings`, etc. — `omlx/settings.py:166-910`):

```python
@dataclass
class ClusterSettings:
    """Cluster-subsystem behavior toggles."""

    # Off by default (#2870 review): a server-initiated kill of a model the
    # user loaded locally is a product decision the user opts into, not a
    # default behavior. See docs/cluster-competing-model-eviction-redesign.md.
    auto_evict_competing_local_models: bool = False
```

Wired into `GlobalSettings` (`omlx/settings.py:910`) as `cluster:
ClusterSettings`, same as every other section. Surfaced in the admin
dashboard's Cluster settings panel (`omlx/admin/templates/dashboard/
_cluster.html`, which PR #2870 already touched for unrelated reasons) as a
single labeled toggle, off by default, with copy naming exactly what it does
("Automatically unload models you loaded locally on a peer Mac when cluster
activation fails there for lack of memory") — no vague "auto-recovery"
euphemism; the maintainer's objection was partly about the *nature* of the
action, so the setting's label should not undersell it.

`activate_cluster_deployment` (`omlx/cluster/routes.py:3155`, the
`@router.post("/deployments")` handler) reads this flag before calling any
eviction code at all — the entire recovery path added below is skipped, not
just gated deeper in, when the setting is off. With the setting off, behavior
is byte-for-byte identical to today's un-reworked code: the original 503 and
its `detail` string, nothing appended, nothing evicted.

### 3.2 Structured failure attribution, hop by hop

**Hop 1 — rank-side marker gains a structured error type and, for
`InsufficientMemoryError` specifically, structured size fields.**
`inference_worker.py:1241-1250`:

```python
except Exception as exc:
    preserve_failure_marker = True
    with suppress(Exception):
        # Lazy import matching the existing convention for this exception
        # (omlx/cluster/memory_guard.py:446,614 both import it inside the
        # function that needs it, not at module load) — not currently
        # imported anywhere in this file.
        from omlx.exceptions import InsufficientMemoryError

        extra: dict[str, Any] = {
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, InsufficientMemoryError):
            extra["required_bytes"] = exc.required
            extra["current_bytes"] = exc.current
        marker.update("failed", **extra)
    raise
```

`error` (the existing free-form field) stays — it's still the right thing to
show a human in an incident or a log. `error_type` is new and is what
downstream code should ever branch on. `RuntimeMarker.update`'s `**extra`
signature (`inference_worker.py:320`) already accepts arbitrary new fields
with no schema migration.

**Hop 2 — the launcher exposes structured failures, not just a joined
string.** Add a small frozen dataclass and a second accessor alongside the
existing one on `DistributedJobSupervisor`
(`launch.py`, next to `_runtime_failure_reason`, ~line 2137):

```python
@dataclass(frozen=True)
class RankFailure:
    rank: int
    node_id: str
    error_type: str | None
    error: str
    required_bytes: int | None = None
    current_bytes: int | None = None

def _runtime_failures(self) -> tuple[RankFailure, ...]:
    """Structured counterpart to `_runtime_failure_reason` — same marker
    read, no string join. Rank/node_id come from the marker's own fields,
    the same ones `_runtime_failure_reason` already trusts; this just stops
    throwing them away before the caller can use them as fields."""
    failures: list[RankFailure] = []
    for rank, host in enumerate(self.deployment.hosts):
        # `_read_rank_marker` is new — extract the SSH-vs-local read branch
        # currently inlined in `_runtime_failure_reason`'s loop body
        # (launch.py:2148-2159: local `read_marker` vs. remote
        # `read_remote_marker`) into a shared helper both methods call, so
        # the marker-reading logic exists in exactly one place.
        marker = self._read_rank_marker(rank, host)
        if marker is None or marker.get("phase") not in {
            "failed", "peer_lost", "launcher_lost",
        }:
            continue
        error = marker.get("error")
        if not (isinstance(error, str) and error.strip()):
            continue
        failures.append(RankFailure(
            rank=rank,
            node_id=host.node_id,
            error_type=marker.get("error_type"),
            error=error.strip(),
            required_bytes=marker.get("required_bytes"),
            current_bytes=marker.get("current_bytes"),
        ))
    return tuple(failures)
```

`_runtime_failure_reason()` keeps its current signature and callers
unchanged (it's used for the human-readable `detail` string elsewhere) but
its body becomes `"; ".join(f"rank {f.rank} ({f.node_id}): {f.error}" for f
in self._runtime_failures())[:_LOG_LINE_LIMIT] or None` — same output,
computed from the new structured source instead of a separate parallel
implementation, so the two can't drift.

`DistributedLaunchError` gains an optional structured payload instead of
staying a bare-string `RuntimeError`:

```python
class DistributedLaunchError(RuntimeError):
    def __init__(self, message: str, *, rank_failures: tuple[RankFailure, ...] = ()):
        super().__init__(message)
        self.rank_failures = rank_failures
```

Only the one raise site that matters for this feature needs to pass
`rank_failures=` — the ~30 other `DistributedLaunchError(...)` call sites
across `launch.py` are unaffected (default `()`, same as today's implicit
"no structured info").

**Hop 3 — routes.py consumes fields, deletes the regex.** `error_type` is
matched against a maintained allow-list, not a single literal string — the
extra indirection costs nothing today (one entry) and means a future
memory-shaped exception type is a one-line addition here instead of a
second parallel check somewhere else:

```python
# Exception type names (RuntimeMarker's "error_type" field, set from
# type(exc).__name__ in inference_worker.py) that mean "this rank failed for
# lack of memory, evicting a competing local model might let a retry
# succeed." Extend this, not the matching logic, when a new memory-shaped
# failure type needs the same recovery.
_MEMORY_FAILURE_TYPES = frozenset({"InsufficientMemoryError"})


def _memory_squeezed_hosts(
    deployment: ClusterDeployment, exc: DistributedLaunchError
) -> list[ClusterHost]:
    """Hosts implicated in a memory-attributable activation failure."""
    memory_failures = [
        f for f in exc.rank_failures if f.error_type in _MEMORY_FAILURE_TYPES
    ]
    if not memory_failures:
        return []
    by_node_id = {host.node_id: host for host in deployment.hosts}
    implicated = {
        by_node_id[f.node_id]: None
        for f in memory_failures
        if f.node_id in by_node_id
    }
    return list(implicated)
```

No regex, no rank-index fallback (rank is always a real field now, not
something to reconstruct), no substring check on free text. The "a memory
failure that names no rank still deserves recovery everywhere" fallback from
#2870 is **dropped** — if `error_type` isn't populated (old rank binary via a
staggered upgrade, or a future exception type this doesn't special-case), the
new code should not guess at blast radius by evicting every node in the
deployment. Fail closed: no `error_type` match → no eviction, original 503
unchanged. This is a deliberate behavior narrowing versus #2870, not an
oversight — it's the direct consequence of "keyed on structured failure
attribution" taken seriously.

`activate_cluster_deployment`'s `except DistributedLaunchError` arm passes
`exc` itself (not `str(exc)`) into the recovery path so `exc.rank_failures`
is available; everything downstream of that (the WARN incident, the
freed/pending/pinned/problems bookkeeping, the try/except-must-never-mask
structure) is unchanged from #2870 — none of that was what got the PR closed.

### 3.3 SSH/token-minting pattern — keep as designed

`_REMOTE_LOCAL_EVICTION` (the peer-side Python script embedded as a string
literal) and `evict_remote_local_models()` carry over from #2870 essentially
unchanged: no HTTP route coordinator→peer exists, so SSH + a loopback call
into the peer's own admin API, authenticated by minting a session cookie from
the peer's own persisted `auth.secret_key`, is still the only mechanism
available, and the maintainer confirmed the mechanics are sound. The one
change (below) is *how the script learns which port to hit*, not the
transport or auth model.

### 3.4 Use C5's advertised `admin_port`

C5 (`ClusterStatus.admin_port`, `PR #2885`) is unmerged and depends on C2
(`PR #2875`, also unmerged) — this redesign has a real dependency on
prerequisite work landing first, same shape as this session's own stacked
UI/UX PRs (#3130→#3131→#3132 each depending on the one before). Two options,
not mutually exclusive:

- **Land this feature after C5/C2 merge**, and have the coordinator-side
  eviction dispatch (`_evict_local_models_on_host` in #2870's `routes.py`)
  look up the target host's `admin_port` from the freshest capability-probe
  data the same way `/node-budgets` already does (per C5's own PR
  description) — passed to `evict_remote_local_models(host.ssh,
  admin_port=...)`, script hits that exact port first instead of
  reading `settings.json` peer-side and falling back to `8000, 9000` guesses.
- **If this needs to ship before C5 lands**, keep #2870's peer-side
  `settings.json` read as the *only* port-discovery mechanism (drop the
  `8000, 9000` guess-fallback — guessing a port to send an authenticated
  unload request to is the same class of fragility as the regex, just
  smaller blast radius) and switch to C5's `admin_port` in a follow-up once
  it's available. The peer already knows its own port authoritatively; the
  guess-fallback in #2870 was defense against a `settings.json` read
  failure, not the primary path, so dropping it doesn't reduce reliability
  materially — it just removes a case where the script would confidently
  send a signed request to the wrong port.

Recommendation: land C5 first if there's any flexibility on sequencing —
using the coordinator's own already-known `admin_port` instead of asking the
peer to introspect and report its own port over SSH is strictly less moving
parts, and it's the integration the maintainer specifically flagged as
"worth keeping for that day."

### 3.5 Everything else from #2870 — unchanged

Not touched by this redesign, carried over as-is:

- Pinned models, `source_type == "cluster"` entries, and still-loading
  models skipped on both the coordinator-direct (engine-pool) and peer-SSH
  paths, reported separately (`skipped_pinned` in the outcome dict).
- Recovery runs in its own `try/except` inside the existing
  `except DistributedLaunchError` arm — the original incident and
  `HTTPException` are always recorded/raised first; a crash in recovery adds
  a second WARN incident, never replaces or masks the original 503.
- WARN incident recording (`activation_memory_recovery` /
  `activation_memory_recovery_failed`) and the freed/pending/pinned/problems
  notes appended to the error detail.
- Coordinator-direct path dispatches via the engine pool with no SSH hop
  (`_local_ssh_target(host.ssh)` check); every other host goes through the
  peer script.

---

## 4. What changes vs. #2870, summarized

| Aspect | #2870 (closed) | This redesign |
|---|---|---|
| Default state | Always active | **Off**, explicit opt-in toggle in admin Cluster settings |
| Failure targeting | Regex over `DistributedLaunchError`'s string message | `DistributedLaunchError.rank_failures: tuple[RankFailure, ...]`, populated from the rank marker's own structured `rank`/`node_id`/new `error_type` fields |
| Unnamed-rank fallback | Evict every host in the deployment | **Removed** — no `error_type` match means no eviction, fail closed |
| Peer port discovery | Peer reads its own `settings.json`, falls back to guessing `8000`/`9000` | Prefer C5's advertised `admin_port` (coordinator already knows it); if shipped before C5 lands, keep the `settings.json` read but drop the guess-fallback |
| SSH/token-minting mechanics | As designed | Unchanged |
| Pinned/cluster/loading skip logic | As designed | Unchanged |
| Incident recording, error-detail append, crash-isolation | As designed | Unchanged |

**Implementation-discovered addendum, not in the original design:** the
opt-in gate check itself needs the same crash-isolation discipline as the
recovery it gates. The first cut called `get_settings()` directly in the
`except DistributedLaunchError` handler — `get_settings()` raises
`RuntimeError("Settings not initialized...")` in contexts where
`init_settings()` was never called (worker-only installs, some test apps),
and since that call sat outside any try/except, the exception would have
propagated out of the handler and replaced the intended 503 with an
unrelated 500 — the exact "recovery must never mask the original failure"
principle this whole feature is built around, violated by the gate check
rather than the recovery logic it guards. Fixed by wrapping the settings
read itself in try/except, defaulting to "feature off" (`auto_evict =
False`) on any failure to read it — caught by
`tests/test_cluster_routes.py::test_cluster_activation_rolls_back_when_canary_fails`,
which runs `activate_cluster_deployment` against a test app with settings
uninitialized.

## 5. Testing plan (sketch)

Mirrors #2870's existing test file (`tests/test_cluster_local_eviction.py`,
11 cases) with these additions/changes:

- New: `ClusterSettings.auto_evict_competing_local_models` defaults to
  `False`; `activate_cluster_deployment` under test with the flag unset
  produces byte-identical 503 detail to a build with no eviction code at
  all (regression guard against the flag being checked too late/loosely).
- New: `RuntimeMarker.update("failed", ...)` includes `error_type` for a
  raised `InsufficientMemoryError`, and `required_bytes`/`current_bytes`
  round-trip through the marker JSON.
- New: `DistributedJobSupervisor._runtime_failures()` returns the right
  `RankFailure` tuple from a set of faked per-rank markers, including the
  case where `error_type` is absent (old-format marker / non-memory
  exception) — must NOT be treated as a memory failure.
- Changed: replace #2870's "rank targeting via the exact error format
  observed in production" test (which existed *because* of the regex) with
  a test asserting `_memory_squeezed_hosts` reads `rank_failures` directly
  and produces the same host set — same intent, no format-string coupling.
- New: a failure with `rank_failures` present but none carrying an
  `error_type` in `_MEMORY_FAILURE_TYPES` (e.g. all `ConnectionError`)
  produces zero implicated hosts, not "recover everywhere" — covers the
  fail-closed behavior change from §4.
- New: `_MEMORY_FAILURE_TYPES` is a set, not a single comparison — a test
  adding a second entry (e.g. a fake `SomeOtherMemoryError`) and asserting
  it's matched the same way `InsufficientMemoryError` is, guarding against a
  future edit accidentally reverting the allow-list back to a single
  `==` compare.
- Keep unchanged: the peer-script SSH-boundary tests (report round-trip,
  interpreter fallback, non-JSON rejection, cookie signed against the
  persisted secret, pinned/cluster/loading filtering, unreachable-server
  safety) and the crash-isolation test (recovery exception → second WARN
  incident, original 503 unchanged).

## 6. Decisions (resolved)

Both open questions from the first draft of this doc were resolved directly
by the user rather than left to guesswork:

- **Toggle scope: cluster-wide.** One `ClusterSettings
  .auto_evict_competing_local_models` flag, set from the coordinator,
  applies to every node in the deployment — matching how every other
  cluster behavior in this codebase is already configured (from the
  coordinator, not negotiated per-peer). §3.1's design already assumed
  this; no change needed there. (A per-node override — a peer refusing
  auto-eviction even when the coordinator has the feature on — was
  considered and explicitly not adopted; if that's ever wanted, it would
  need its own peer-side setting, exchanged the way node role or memory
  guard tier already are, and is out of scope for this rework.)
- **`error_type` matching: allow-list, built from the start.** §3.2's
  `_memory_squeezed_hosts` above uses `_MEMORY_FAILURE_TYPES` (a
  `frozenset`), not a single `==` comparison, so a second memory-shaped
  exception type is a one-line addition to the set rather than a rewrite of
  the matching logic. The allow-list is a maintained code constant, not a
  user-facing setting — nothing in this rework calls for making it
  admin-configurable, and adding that surface without a concrete second use
  case would be speculative.
