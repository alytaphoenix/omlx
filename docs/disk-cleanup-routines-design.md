# Disk cleanup routines — design and phased implementation plan

Design doc + phased implementation checklist for keeping oMLX's on-disk
footprint bounded and self-healing. Every file:line reference below was
**verified against HEAD `2718845b` (branch `deploy/session-fixes-v2`) on
2026-08-25**, and every size/count was **measured directly against the live
data directory `/Users/alytaphoenix/.omlx/` on the same date** (read-only:
`du`, `ls`, `find`, `stat`, log grep — nothing was deleted or moved; a live
long-context workload may be using the cache dir). Line numbers will drift as
the tree moves — treat them as anchors (the quoted identifiers are the stable
handles), and re-locate rather than trust a stale number.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here). Live-data paths are
absolute under `~/.omlx/`.

Findings are labeled **CONFIRMED** (the full causal chain was traced in code
and/or observed on disk) or **PLAUSIBLE** (mechanism traced, final runtime
behavior needs a repro or runtime check).

**Unit note (load-bearing):** config size strings (`"92GB"`) are parsed
1024-based (`parse_size`, `paged_ssd_cache.py:941-973`) and `format_bytes`
log output is 1024-based too, so "92 GB" in settings and logs means **92 GiB
= 98.8e9 bytes**. `du` is also 1024-based. This doc writes GiB for 1024-based
figures and raw GB (1e9) only where a `stat` sum is quoted. Without this note
none of the reconciliations below check out — e.g. the scan log's
"gdn_size=52.34 GB" is exactly the measured 56.20e9 logical bytes.

---

## 1. Context

### The on-disk estate, measured 2026-08-25

Volume: 926 GiB, **116 GiB free (88% used)** — the disk is already in the
zone where the cache's own dynamic clamp (§ "what bounds exist") is within 2×
of engaging. `~/.omlx` totals **211 GiB**:

| Path | Size | Files | What it is | Owner code |
|---|---|---|---|---|
| `~/.omlx/models/` | 89 GiB | 6 models | model weights — **load-bearing** | `model_discovery.py`, `admin/hf_downloader.py` |
| `~/.omlx/cache/` | 92 GiB (du) | — | shared paged SSD KV cache, **at its 92 GiB cap** | `cache/paged_ssd_cache.py` |
| — `cache/{0..f}/` | 42.48e9 B logical | 409 `.safetensors` | main KV blocks (335 compatible + 74 incompatible = 4.95 GiB) | same |
| — `cache/_gdn_sidecars/` | 56.20e9 B logical (52.34 GiB) | 365 files in 3 populated digest dirs (24/18/11 GiB) + 2 empty digest dirs | durable per-block GDN recurrent-state checkpoints | same + `boundary_snapshot_store.py` |
| — `cache/_boundary_snapshots/` | 0 B | 0 | ephemeral per-request prefill snapshots (reset at last server start) | `boundary_snapshot_store.py` |
| — `cache/response-state/` | 0 B | 0 | persisted Responses-API state (bounded, count=1000) | `api/responses_utils.py` |
| — `cache/vision_features/` | 0 B | 0 | vision-feature SSD cache (10 GiB own budget) | `cache/vision_feature_cache.py` |
| `~/.omlx/cluster/runtime/prompt-cache-ssd/` | **30 GiB** | 86 files in 8 per-deployment dirs | cluster rank-local prompt snapshots — **all orphaned** (registry empty) | `cluster/prompt_snapshot_cache.py` |
| `~/.omlx/cluster/` (rest) | ~90 KiB | 13 marker JSONs + 4 registry/benchmark JSONs | runtime markers, deployments/incidents/benchmarks | `cluster/runtime.py`, `strategy_benchmarks.py` |
| `~/.omlx/logs/` | 2.3 MiB | 17 | `server.log` + 7 dated rotations (bounded) + 9 ad-hoc `fork-*`/`crash`/`watchdog` logs (unbounded class) | `logging_config.py` + shell redirects |
| `~/.omlx/` loose files | ~120 MiB | ~25 | `omlx-app-backup-20260819-*.tar.gz` (59 MiB), `benchmark_*` py/json/log, `model_settings.json.bak.*` ×6, `settings.json.bak.*`, `cluster_bench_*.json`, `__pycache__/` (56 KiB), `stats.json`, `bin/` | user-created / `server_metrics.py` |

### Growth rate and per-token cost (the numbers that matter)

Surviving-file write dates in `~/.omlx/cache` (main blocks + sidecars,
`stat` mtime — survivors only; true write volume is higher because LRU
unlinks losers):

```
2026-08-20   3.6e9 B   27 files
2026-08-21  50.1e9 B  349 files
2026-08-23  44.2e9 B  393 files
2026-08-24   0.8e9 B    5 files
```

So heavy days write **~45-50e9 bytes/day into the cache**, and the whole
92 GiB budget turns over in roughly two heavy days. The oldest surviving main
block is 2026-08-20 19:11, newest 2026-08-23 17:19 — a ~3-day LRU horizon at
current load. That is the eviction machinery working (see §8), not a leak.

Per-token cost for the dominant workload (Qwen3.6-35B-A3B GDN hybrid,
`paged_cache_block_size` auto-enlarged to **2048** for ArraysCache models —
`_ARRAYS_CACHE_BLOCK_SIZE`, `scheduler.py:2663`, applied at
`_enlarge_block_size_for_arrays_cache`, scheduler.py:2665-2712):

- main KV block: ~104 MB avg / 2048 tokens ≈ **51 KB/token**
- GDN sidecar: ~145-154 MB per block boundary (full recurrent state, near
  constant per model) ≈ **71-75 KB/token**
- combined ≈ **125 KB/token of unique prefix**. A 262k-token chain ≈ 128
  blocks ≈ **13e9 B main + 19e9 B sidecars ≈ 32e9 B per distinct chain** —
  the 92 GiB budget holds about three such chains, which matches the
  observed 3-day horizon and the sidecar/main split on disk.

Cluster boundary files are bigger still: the dead
`Qwen3.8-27B-…-9d18d136930c-rank-0` dir holds 20 GiB in 55 files ≈
**370 MB/file** (a 2048-token KV slab for the rank's layer slice + full
recurrent state, `prompt_snapshot_cache.py:355-371`).

### What bounds exist today (verified — do not re-design these)

- **One shared LRU byte budget over main blocks + incompatible blocks + GDN
  sidecars.** `_tracked_ssd_size` (`paged_ssd_cache.py:2212-2223`) sums all
  three indexes; `_evict_tracked_until_size` (4505-4551) does a single
  deterministic oldest-first walk across them. The live scan log confirms it
  holds: `total_size=91.90 GB` vs `max_size=92.00 GB`
  (`~/.omlx/logs/server.log.2026-08-23`, "SSD cache scan complete /
  PagedSSDCacheManager initialized" lines).
- **A dynamic disk-free clamp at save time.** `_get_effective_max_size`
  (4467-4503) = min(configured cap, 99% of tracked+free,
  `_DISK_SAFE_RATIO` 4465), disk usage cached 30 s; ENOSPC invalidates the
  cache (`_write_block_file`, 3012-3035). Eviction-on-save is bounded to 32
  inline unlinks (`_MAX_INLINE_UNLINKS_PER_SAVE`, 162).
- **Startup convergence.** `_scan_existing_files` (2233-2299) rebuilds all
  three indexes from disk and, if the dir already exceeds the budget, evicts
  unbounded before serving (2295-2299).
- **Crash-consistent writes.** All four writers stage to
  `*_tmp.safetensors`, fsync the file (`_write_safetensors_no_mx`,
  935-936), rename, fsync the parent dir (`_fsync_parent_dir`, 853-874) —
  the qwen35 doc's F1 fsync fix is implemented (payload checksum still
  isn't).
- **Boundary snapshots self-heal at startup.** `reset_boundary_snapshot_root`
  rmtree's `cache/_boundary_snapshots` at server startup and shutdown
  (`boundary_snapshot_store.py:100-114`; called from
  `server.py:370-386,423,598`) — confirmed: the dir is empty on disk.
  Per-request cleanup has real writer-thread barriers
  (`cleanup_request`/`cleanup_all`, boundary_snapshot_store.py:596-804).
- **Read-path self-heal.** `load_block` prunes the index entry on missing
  file (3802-3806, and the racing-eviction `FileNotFoundError` at
  3816-3822) and unlinks + prunes on corrupt payload (3870-3879).
- **Logs:** `server.log` rotates daily with `backupCount=7`
  (`configure_file_logging`, `logging_config.py:228-283`) — the 8 dated
  files on disk are within policy.
- **Response state** is count-bounded (1000, `MAX_STORED_RESPONSES`,
  `api/responses_utils.py:529`) with eviction deleting the JSON files
  (609-612).
- **Vision features** have their own 10 GiB budget + LRU
  (`vision_feature_cache.py:81-82,315+`).
- **Downloads:** cancel/failure cleans HF's staged `._____temp` dir
  (`_cleanup_partial`, `admin/hf_downloader.py:1085-1101`); measured: zero
  `.incomplete` files on disk today.
- **Operator clear:** `POST /api/ssd-cache/clear`
  (`admin/routes.py:5488-5530`) — but see A4.

### Coverage (honesty note)

Read in full: `cache/paged_ssd_cache.py`, `cache/boundary_snapshot_store.py`
(lifecycle halves), `cache/factory.py`, `cache/recovery.py`,
`cluster/prompt_snapshot_cache.py`, `logging_config.py`, the `ResponseStore`.
Targeted reads: `scheduler.py` (snapshot cadence, store/manager
construction, draft manager), `server.py` (lifespan, response-state),
`cluster/{inference_worker,telemetry,launch}.py` (SSD store wiring),
`settings.py`/`config.py` (size defaults), `admin/routes.py` (clear
endpoint), `admin/hf_downloader.py` (partial cleanup),
`cache/vision_feature_cache.py` (scan/eviction). Skimmed only:
`cache/prefix_cache.py` (4.9k lines — its disk interaction is entirely via
the manager API), `cache/paged_cache.py` (in-memory). Claims about skimmed
files are correspondingly weaker.

---

## 2. Theme A — The shared SSD KV cache (`~/.omlx/cache`)

The biggest consumer (92 GiB) and the best-defended one. The findings here
are gaps at the edges of an otherwise working LRU system.

### A1. Startup scan skips-but-keeps unreadable files — an unreapable orphan class — MEDIUM (CONFIRMED mechanism; currently zero on disk)

**What's wrong.** `_scan_existing_files` (`paged_ssd_cache.py:2249-2269`)
globs `*.safetensors` and, for any file whose metadata can't be read,
**continues without indexing and without deleting**: `_read_file_metadata`
returns `None` for (a) truncated/corrupt files, (b) files with a format
version outside `_READABLE_CACHE_FORMAT_VERSIONS` (2910-2919; current
version "3", readable {"2","3","4","5"} — set defined at 179-181), (c)
files missing a `block_hash`.
The same glob also matches leftover `*_tmp.safetensors` staging files
(`_write_block_file` names them `<hash>_tmp.safetensors`, 2987) — a partial
temp from a crash mid-write is unreadable and lands in the same skipped
class. Skipped files are invisible to `_tracked_ssd_size`, to LRU eviction,
and to `clear()` (4748-4791 walks the indexes only) — nothing in the
codebase can ever delete them. The GDN sidecar scan has the same shape
(`_scan_existing_gdn_sidecars` skips malformed digests/names, 2318-2356),
and so does the vision cache scan (`vision_feature_cache.py:394-440`).

A second, subtler member of this class: a **complete** `_tmp` file that was
never renamed (crash between fsync and rename) is *readable*, so the scan
indexes it under its interior `block_hash` — and `PagedSSDCacheIndex.add`
replaces on duplicate hash (1086-1095), so whichever of tmp/final is
scanned second wins and the other file leaks untracked.

**Measured now:** zero — the Aug-23 scan log says `scanned=409, indexed=335,
skipped_incompatible=74, errors=0`, and 409 = the exact file count on disk;
`find ~/.omlx/cache -name "*_tmp.safetensors"` → 0. This is a crash-window
class, not a live leak — but each crash can strand up to ~104 MB (one main
block) or ~150 MB (one sidecar) forever, and version bumps ("1"→"3" already
happened once) strand whole generations.

**Contrast (the correctness rule):** the 74 *incompatible-but-readable*
blocks (4.95 GiB) are **not** orphans — they're indexed in
`_incompatible_index`, count against the budget, and are LRU-evictable
(scan comment at 2233-2239: a shared dir must serve multiple models). A
cleanup routine must preserve exactly this distinction: unreadable ⇒
delete; readable-but-foreign ⇒ keep and let the shared budget manage it.

**Failure scenario.** Power loss mid-write leaves `d/<hash>_tmp.safetensors`
truncated. Every subsequent boot logs "Failed to read …" (as `errors=1`) and
carries the dead bytes forever; the effective budget silently shrinks by the
orphan's size (the file consumes disk that `_get_effective_max_size` counts
as neither tracked nor free-for-cache).

**Fix (S).** Reconciliation sweep in/after `_scan_existing_files` — design
in §6.R1. Effort S: the scan already visits every file and already knows
which ones it refused.

### A2. Main-block LRU recency is not persisted — restart degrades eviction to write-time FIFO and biases it against main blocks — MEDIUM (CONFIRMED code asymmetry; bias magnitude PLAUSIBLE)

**What's wrong.** On a hit, main blocks get only an in-memory
`self._index.touch(block_hash)` (`load_block`, 3763/3783/3859 →
`PagedSSDCacheIndex.touch`, 1143-1154) — the file's mtime is never updated.
At the next startup, `_read_file_metadata` seeds `last_access` from
`st_mtime` (2962), i.e. **write time**. GDN sidecars, by contrast, persist
every LRU touch with `os.utime` (`get_gdn_checkpoint_file_with_diagnostic`,
2509-2515, explicitly "Persist the LRU touch across manager restarts").

Consequences after every restart (and this deployment restarts often —
the log dir shows 5+ starts since Aug 19):

1. Main-block eviction order is FIFO-by-write-date, so a heavily-reused old
   prefix (a long-lived chat session's early blocks) is evicted before a
   written-yesterday-never-hit block.
2. The shared three-index LRU walk (`_evict_tracked_until_size`, 4524-4543)
   compares main-block *write* times against sidecar *access* times —
   systematically favoring sidecars and evicting exactly the main blocks
   whose loss forces a full re-prefill. This compounds A3.

**Fix (S/M).** Persist main-block recency cheaply: throttled `os.utime` on
hit (e.g. only when the on-disk mtime is >N hours stale — one extra syscall
per cold hit, none for hot-cache hits), mirroring the sidecar path. Effort
S. (A sidecar-style always-utime is fine too; the throttle just avoids
mtime churn on hot chains.)

### A3. GDN sidecars consume 57% of the shared budget by policy accident — MEDIUM (CONFIRMED numbers; "right" split is a product decision)

**What's wrong — nothing broken, everything working as wired.** Sidecars
share the main budget deliberately (`_tracked_ssd_size` docstring,
2212-2218). But at ~71-75 KB/token vs the main blocks' ~51 KB/token, the
durable full-recurrent-state-per-2048-token-boundary policy
(`commit_gdn_checkpoint`, scheduler.py:1595-1638 →
`commit_gdn_checkpoint_file`, paged_ssd_cache.py:2359-2446) means sidecars
mathematically outgrow the KV they annotate: measured 52.34 GiB sidecars vs
39.6 GiB main blocks. Combined with A2's eviction bias, the steady state
under budget pressure drifts toward "lots of recurrent-state checkpoints
whose main KV blocks were evicted" — a sidecar without its main block chain
is pure waste (the restore path needs the block chain to reach the
boundary).

**Fix (M, gated on Phase 0 data).** Two candidate policies, pick after
measuring hit-rates per tier: (a) a configurable sidecar budget share
(e.g. sidecars capped at 40% of the budget, enforced by biasing the LRU
walk), or (b) **orphan-sidecar collection**: during the reconciliation
sweep (§6.R1) or on a low-priority timer, drop sidecars whose
`source_block_hash` is no longer present in either main index — those are
unreachable-by-construction today. (b) is correctness-safe and should be
done regardless; effort S once R1 exists.

### A4. `POST /api/ssd-cache/clear` leaves the sidecars behind in the no-model-loaded path — MEDIUM-LOW (CONFIRMED)

**What's wrong.** The endpoint (`admin/routes.py:5488-5530`) uses loaded
managers' `clear()` (which does clear sidecars, paged_ssd_cache.py:
4764-4788) but its filesystem fallback for unloaded models sweeps only the
16 hex subdirs (`for subdir in "0123456789abcdef"`, routes.py:5518). With
no model loaded, "Clear SSD cache" deletes ≤39.6 GiB of blocks and leaves
52.34 GiB of sidecars — the majority of the cache — plus
`_boundary_snapshots` leftovers and `vision_features`. The operator-visible
number ("total_deleted") looks successful.

**Fix (S).** Extend the fallback sweep to `_gdn_sidecars/*/*.safetensors`
(and optionally `vision_features/{0..f}`), using the same
symlink-refusal checks the manager applies (`_is_safe_gdn_sidecar_file`
semantics, 2116-2133).

### A5. Multiple managers over one directory drift out of sync — LOW-MED (CONFIRMED mechanism; consequences self-healing but lossy)

**What's wrong.** Several `PagedSSDCacheManager` instances can own
`~/.omlx/cache` simultaneously, each with its own full index and its own
92 GiB budget over the same files:

- one per loaded model in the engine pool (scheduler
  `_init_paged_ssd_cache`, scheduler.py:12536-12600 — the dir is shared
  flat, no per-model subdir; isolation is metadata-only);
- plus a **draft-model manager** on the same dir when SpecPrefill/DFlash is
  active (scheduler.py:8121-8138);
- cross-process (menu-bar app vs `run-fork-server.sh`) is normally
  prevented by the port-8000 guard in the fork script, but nothing in the
  cache layer itself enforces single-process access.

When manager A evicts+unlinks a file manager B has indexed: B's
`total_size` keeps counting it (under-using B's budget), B's `has_block`
still says True (4098-4118 checks indexes only — no file stat), and B's
prefix-cache chain match then fails at `load_block` (missing-file
self-heal, 3802-3806) → partial restore / recompute mid-request. Nothing
corrupts; work is wasted and budget accounting drifts until each manager
happens to touch the dead entries.

**Fix.** Don't build cross-manager coordination (L, not worth it). Two
cheap mitigations: (S) the periodic disk-pressure tick (§6.R2) calls a new
`reconcile_tracked_sizes()` that stat-validates a bounded batch of LRU-tail
entries per tick, pruning dead ones; (S) document the invariant that *all*
deletion goes through a manager or through the startup-window sweep — which
the routines in §6 obey.

---

## 3. Theme B — Ephemeral boundary snapshots (`cache/_boundary_snapshots`)

This is the local-server store the clustering doc's D2 pointed at (its
cluster twin is Theme D — the two are separate implementations with
different lifecycles; keep them distinct). Verified mechanics: for models
with non-sliceable state, `boundary_enabled` turns on whenever the paged
cache is on (`scheduler.py:3353-3357`), the block size is auto-enlarged to
2048 (scheduler.py:2663), and at **every** 2048-token boundary of prefill
the full non-sliceable state is serialized and staged to
`cache/_boundary_snapshots/<pid>-<uuid>/<sha256(request_id)>/<token_count>.safetensors`
(emit gate `token_count % block_size == 0`, scheduler.py:6230; save path
`boundary_snapshot_store.py:219-373`). Writes are asynchronous with an
inline fallback — D2's "synchronous" applies to the *cluster* twin, not
this one.

### B1. Per-request disk usage is unbounded — the byte cap bounds RAM, not disk — MEDIUM (CONFIRMED mechanism; peak derived, not yet observed)

**What's wrong.** The store's only byte bound is
`_pending_max_bytes` (512 MiB default, `gdn_ssd_pending_max_size`,
`boundary_snapshot_store.py:51,185`) — that throttles the **in-RAM pending
queue**, then everything lands on disk and stays until `cleanup_request`
(fired at request completion/abort, scheduler.py:2467/9049/11112/11434,
deferred until the store future completes). Nothing caps the on-disk
footprint of a request in flight, and nothing caps concurrent requests'
sum. At the measured ~145-154 MB of recurrent state per boundary, a single
262k-token prefill stages ~128 boundaries ≈ **19-20e9 bytes of transient
disk** (minus boundaries handed off early to sidecar promotion via
`take_staged_file`); `max_concurrent_requests=2` doubles that. None of it
is visible to the paged manager's budget or to `_get_effective_max_size` —
it lands in the same filesystem the 99% clamp reads as "free", so a
long-prompt burst can push a nearly-full disk over the edge between two
30-second disk-usage refreshes.

**Measured now:** 0 B (no long prefill in flight at inspection time; the
dir is reset-on-start). The mechanism is fully confirmed; the peak number
should be confirmed once by watching a long run (Phase 0.3).

**Fix (M).** Give the store a disk high-water mark: track per-session bytes
written minus cleaned (it already tracks every file it creates in
`_file_registry`), and when a request's staged bytes exceed a cap
(e.g. `min(free_disk × 0.25, configured)`), degrade gracefully — drop the
*oldest interior* boundaries of that request first (they are only needed
for mid-chain restarts; the final boundary is the one promoted), keeping a
`None` marker so the block store records a placeholder exactly like today's
failed-save path (`scheduler.py:6256-6264` already handles `saved=False`).
This reuses the existing degradation semantics rather than inventing new
failure modes.

### B2. Startup reset can delete another live process's session — LOW (PLAUSIBLE; needs the two-process window to exist)

**What's wrong.** `reset_boundary_snapshot_root` rmtree's the **whole**
`_boundary_snapshots` root — every session — at each server start
(server.py:423) and shutdown (598). Session dirs are per-process
(`<pid>-<uuid>`, boundary_snapshot_store.py:168-170) precisely so multiple
processes can coexist, but the reset ignores that. The fork-server script's
port-8000 guard makes the overlap unlikely (`run-fork-server.sh` refuses to
start if 8000 is bound), but a second server on another port, or a start
racing a slow shutdown, would silently strip an in-flight request's
snapshots — the request then stores placeholder blocks (degraded, not
corrupt: `_NoBoundarySnapshot` handling, scheduler.py:1203-1244).

**Fix (S).** Make the reset per-session-aware: delete only session dirs
whose `<pid>-` prefix names a dead PID (the cluster marker code already has
this exact idiom — `marker_owner_is_live`, `cluster/liveness.py:269-286`).
Fold into the reconciliation sweep (§6.R1).

---

## 4. Theme C — Disk-pressure policy

### C1. Eviction only runs at save time; an idle server never responds to disk pressure — HIGH (design gap; CONFIRMED by grep)

**What's wrong.** The only three triggers of eviction are: a block save
(`_enforce_size_limit_for_new_block`, called from save/enqueue paths), a
sidecar commit (2415), and the startup scan (2298-2299). The public
`enforce_size_limit()` (4617-4649) has **zero callers** in the tree —
periodic enforcement was designed and never wired. Consequences:

- A server that is loaded but idle (overnight, or serving only short cached
  prompts) holds its full 92 GiB while another process (a model download —
  `models/` grew to 89 GiB the same way, Time Machine, the user's own 59 MiB
  backups…) fills the remaining 116 GiB. The cache reacts only at the next
  block write, at which point the effective limit collapses and the write
  path eats a 32-unlink burst per save until reconverged
  (`_MAX_INLINE_UNLINKS_PER_SAVE` bursts, 4585-4615) — eviction latency
  lands exactly on the inference hot path at the worst time.
- There is no **floor** semantics: `_DISK_SAFE_RATIO = 0.99` targets
  "don't fill the disk with cache", but 1% of a 926 GiB volume is 9.3 GiB —
  macOS gets unhappy well before that, and nothing ever *refuses* cache
  writes; ENOSPC is discovered by the writer thread after the fact
  (3012-3035).
- Theme B and D consumers (boundary snapshots, cluster snapshots) don't
  consult any of this — they write into the same free space unmetered.
- Nothing mirrors the memory guard's admission integration: the Metal
  memory guard has soft/hard thresholds and admission refusal
  (`settings.json` `memory.soft_threshold: 0.88 / hard_threshold: 0.93`);
  disk has no analogue.

**Fix (M).** The disk-pressure guard routine — design in §6.R2.

---

## 5. Theme D — Cluster prompt snapshots (`~/.omlx/cluster/runtime/prompt-cache-ssd`)

The cluster twin of Theme B: rank-local chain-of-boundary files written
**synchronously** during distributed prefill (the clustering doc's D2 —
its perf half stands; this section is the disk half).

### D1. Per-deployment directories orphan their contents forever — the self-clean premise is broken — HIGH (CONFIRMED; 30 GiB dead on this machine now)

**What's wrong.** The store's directory embeds the deployment id:
`~/.omlx/cluster/runtime/prompt-cache-ssd/<deployment_id>-rank-<N>`
(`_prompt_cache_ssd_dir`, `cluster/inference_worker.py:470-481`), and the
deployment id is unique per activation. Cleanup exists at exactly two
points, both of which assume the *same directory* is revisited:

- constructor self-clean: "a new store starts by clearing its directory"
  (`SSDPromptSnapshotStore.__init__`, `prompt_snapshot_cache.py:399-405`,
  module docstring 32-36);
- context teardown: `shutil.rmtree(ssd_store.directory)` with the comment
  "a hard crash skips this and **the next store on the same directory**
  reclaims the leftovers instead" (`cluster/telemetry.py:1027-1029`).

But there is never a next store on the same directory — every relaunch
mints a fresh deployment id, so any teardown that doesn't run (crash,
SIGKILL, launcher loss, power cut — precisely the cluster's fail-stop
design) strands the whole directory permanently. No code path ever
enumerates sibling directories.

**Measured (CONFIRMED dead, not inferred):** 8 dirs on this machine;
`~/.omlx/cluster/deployments.json` is `{"deployments": [], …}` (no active
deployment), and the newest file in the two big dirs is 2026-08-18/20:

```
20 GiB / 55 files  Qwen3.8-27B-oQ4e-mtp-9d18d136930c-rank-0
10 GiB / 27 files  Qwen3.8-27B-oQ4e-mtp-8c11ecb99e8d-rank-0
101 MiB / 2 files  ×2  Qwen3.6-35B-A3B-…-rank-0
0 B ×4             NVIDIA-Nemotron-3.5-…-rank-0
```

**30 GiB immediately reclaimable** — the single largest dead weight in
`~/.omlx`. The same leak exists on the peer Mac (rank-1 dirs), so the
cluster-wide figure is roughly double.

**Failure scenario.** Every crash-terminated activation of a
GDN/rotating-window model leaks up to `max_entries × ~370 MB ≈ 24 GiB` per
rank. A few weeks of cluster experimentation fills a disk with invisible
dead state.

**Fix (S).** Deployment reaper — design in §6.R3. Note deletion here is
collectively safe *by design*: `present_boundaries` re-checks `is_file()`
per chain link (prompt_snapshot_cache.py:513-518) and the all-rank boundary
vote (`agree_ssd_boundary`, telemetry.py:687-712) treats a missing file as
"don't vote that boundary" — divergent rank-local disk state is the normal
case the protocol already handles. Reaping *dead deployments'* dirs cannot
desync a live one as long as the reaper never touches the live deployment's
own directory.

### D2. The live store has no byte bound — MEDIUM (CONFIRMED; same as clustering-doc D2's tail note, restated as the disk finding)

`install_server_telemetry` wires `max_entries=64` and never passes
`max_bytes` (`telemetry.py:633-634,679-681`; store default `max_bytes=None`,
`prompt_snapshot_cache.py:379`). 64 entries × ~370 MB ≈ **24 GiB per rank
per live deployment**, invisible to every other budget, on whatever disk
the rank has. `prompt_cache_ssd` defaults ON
(`ExecutionSettings.prompt_cache_ssd = True`, `cluster/performance.py:168`).
**Fix (S):** pass a `max_bytes` derived from rank-local free disk at
activation (e.g. `min(32 GiB, free × 0.2)`); the store's `_evict_locked`
already handles the byte bound (554-561).

### D3. Runtime marker JSONs accumulate one per deployment — LOW (CONFIRMED; trivial bytes)

13 `<model>-<deployment>-rank-0.json` files (8 KiB each) in
`~/.omlx/cluster/runtime/` from past activations. Kept deliberately as
crash evidence (see the marker docstrings), but nothing ever prunes ancient
ones. Fold into R3: keep the newest N per model, delete markers older than
30 days whose owner is dead. Effort S.

---

## 6. Theme E — Logs and small detritus

### E1. Only `server.log` rotates; the ad-hoc log class is unbounded, and there's no total-size cap — LOW-MED (CONFIRMED; currently only 2.3 MiB)

**What's wrong.** Three sub-issues in `~/.omlx/logs`:

1. `TimedRotatingFileHandler` deletes old backups only at rollover
   (`logging_config.py:264-270`) — a process must be alive across midnight
   to prune. A restart-heavy usage pattern (this machine: 5+ starts in a
   week) can accumulate dated files beyond `backupCount` because deletions
   only consider files matching the suffix pattern at an actual rollover.
   Currently within bounds (8 dated files ≤ 7 backups + live) — borderline.
2. The ad-hoc class — `fork-server-*.log`, `fork-mini*.log`, `crash.log`,
   `installed-mini.log`, `watchdog.log` (9 files today) — is written by
   shell redirects and the watchdog script (`omlx-watchdog.sh` appends
   forever, no truncation), with no rotation and no retention. Small today;
   a single crash-looping day with a verbose fork server can make it not
   small.
3. No total-size cap exists for the directory as a whole; nothing protects
   against one runaway logger.

**Fix (S).** Log policy in R4: on server startup (and on the R2 tick),
delete `logs/*` files matching known-transient patterns older than 14 days,
enforce a directory cap (default 500 MiB) oldest-first-by-mtime excluding
the live `server.log`, and truncate `watchdog.log` beyond 5 MiB.

### E2. `ResponseStore` `.tmp` orphans — LOW (CONFIRMED mechanism; zero on disk)

`_persist_record` stages to `<id>.tmp` then replaces
(`api/responses_utils.py:598-601`); `_load_persisted_records` globs
`*.json` only (617), so a crash between write and replace leaves a `.tmp`
forever. Bytes are trivial (KBs). **Fix (S):** unlink `*.tmp` older than
one hour during `_load_persisted_records`.

### E3. Loose user artifacts in `~/.omlx` root — report-only, never auto-delete — LOW

The 59 MiB `omlx-app-backup-20260819-074655.tar.gz`, `benchmark_*` scripts
/results/logs, `model_settings.json.bak.*` ×6, `settings.json.bak.*`,
`cluster_bench_*.json`, `__pycache__/`, the empty `models/nvidia/` dir —
all user- or session-created, none written by current server code paths.
A reaper must **not** delete these (they are someone's experiment record);
the right treatment is visibility: an admin-endpoint report of
"unrecognized files in the data dir, N bytes" so the human decides.
The only safe automatic candidates: `__pycache__` (regenerable; the fork
script already sets `PYTHONDONTWRITEBYTECODE=1`) and HF `._____temp`
download staging dirs with no active download task older than 7 days.

---

## 7. The cleanup routines (design)

Four routines. Common safety rules first — these are the contract every
routine follows:

**Safety rules (all routines):**

1. **Manager-routed or startup-window only.** Files owned by a live
   `PagedSSDCacheManager` are deleted through its API (`delete_block`,
   `forget_gdn_checkpoint`, `enforce_size_limit`) so index and disk move
   together. Direct unlinks happen only (a) inside
   `_scan_existing_files`'s single-threaded window before the writer thread
   starts (`__init__` order: scan at 1743, writer start at 1777-1782), or
   (b) for files provably owned by nobody (dead-PID session dirs, dead
   deployment dirs, unreadable never-indexed files).
2. **Age gates on anything name-pattern-based.** A `_tmp.safetensors` file
   may be a concurrent manager's in-flight rename (A5: multiple managers,
   same dir, same process). Minimum age 10 minutes before a pattern-matched
   delete; deployment dirs require registry absence AND dead marker owner
   AND >1 h idle mtime.
3. **Symlink refusal** exactly as the sidecar unlink path does it
   (`_unlink_gdn_sidecar_file` O_NOFOLLOW discipline, 2135-2210): never
   unlink through a path with a symlinked component inside the cache root.
4. **Never-touch list** (§9): model weights, active configs, user
   artifacts, anything outside the explicit allowlists. Every routine works
   from an allowlist of known-transient patterns, not a denylist.
5. **Count and log everything.** Every routine increments stats
   (`get_stats_dict` already exists, 4911) and logs one summary line —
   reclaimed bytes must be observable, or regressions in the reaper itself
   go unnoticed.

### R1 — Startup reconciliation sweep (fixes A1, A3(b), B2, E2)

**Trigger:** inside `_scan_existing_files`, after the glob loop, before the
writer thread starts (single-writer window). Runs per manager but the
sweep itself is idempotent, so multi-manager double-runs are harmless.

**Actions, in order:**

1. Delete `{0..f}/*_tmp.safetensors` and `_gdn_sidecars/*/*_tmp.safetensors`
   older than 10 min (mtime). These are never legitimate at rest — every
   writer renames within seconds of creation. In the same change, exclude
   `*_tmp.safetensors` from the scan's *indexing* glob entirely — today a
   complete-but-unrenamed tmp gets indexed (A1's second member), and
   deleting it after indexing would leave a dangling index entry and
   tracked-size drift.
2. Delete final-name files the scan just refused: metadata unreadable, or
   `omlx_cache_format_version` outside `_READABLE_CACHE_FORMAT_VERSIONS`,
   or missing `block_hash`. Rationale for safety: finals are only ever
   created by fsync'd atomic rename, so a final-name file that doesn't
   parse is corrupt, not in-flight. Keep the existing behavior for
   readable-but-incompatible blocks (index as incompatible — the
   multi-model contract, A1's correctness rule). Same treatment in the
   sidecar and vision scans.
3. Remove empty `_gdn_sidecars/<digest>/` directories (2 exist today) and
   sidecars whose digest dir name fails the hex/length check (already
   skipped at 2321-2328 — now delete instead).
4. Orphan-sidecar collection (A3(b)): delete indexed sidecars whose
   `source_block_hash` appears in neither main index — with one caveat:
   only when the manager's expected model signature matches the sidecar
   namespace, so one model's manager never GCs another model's sidecars
   (same scoping rule as `invalidate_stale_layer_signature`, 4341-4344).
   Gate behind a config flag for the first release.
5. Boundary-snapshot root: replace the blanket rmtree
   (`reset_boundary_snapshot_root`) with per-session reaping — delete
   session dirs whose `<pid>-` owner is dead (reuse the
   `marker_owner_is_live` idiom); keep the full rmtree behavior when no
   other oMLX process exists (the common case, preserving today's
   guarantees).
6. `response-state/*.tmp` older than 1 h (E2).

**Effort:** S-M total. **Risk:** low — everything deleted is either
age-gated staging, provably corrupt, or provably unowned; the
sidecar-orphan step is the only judgment call and ships flag-gated.

### R2 — Disk-pressure guard (fixes C1; mitigates A5, B1)

**Trigger:** a periodic tick (default 60 s) on the existing scheduler
housekeeping cadence (the process-memory enforcer already runs a 1 s
interval loop — piggyback a 60× subdivision, or a plain asyncio task in
`lifespan`). Also fired immediately on ENOSPC (the writer already
invalidates the disk cache there — add a guard kick).

**Policy — mirror the memory guard's shape** (`settings.json` `memory.*`
precedent: soft 0.88 / hard 0.93):

- `disk.soft_free_floor` (default: max(20 GiB, 5% of volume)): below this,
  each tick calls `enforce_size_limit()` on every live manager (finally
  giving it a caller) with a *reduced* effective cap — scale the configured
  cap by `free/soft_floor` so the cache sheds gradually rather than
  cliff-evicting; log one throttled warning.
- `disk.hard_free_floor` (default: max(10 GiB, 2% of volume)): below this,
  additionally (a) `save_block`/sidecar-commit become no-ops that count a
  `saves_refused_disk_pressure` stat (a cache write is always optional —
  refusing is strictly safe, unlike refusing a request), (b) the boundary
  snapshot store degrades to its failed-save placeholder path (B1's
  machinery), (c) admission is untouched — disk pressure never rejects
  inference, because inference doesn't require disk.
- Interaction with the existing 99% clamp: `_get_effective_max_size` stays
  as-is (it is the *save-time* backstop); the guard is the *idle-time* and
  *cross-consumer* actor. The floors are expressed in absolute free bytes,
  not cache-relative, so they see pressure the clamp attributes to "other
  people's data".
- Eviction order under pressure (one shared LRU walk already exists):
  1. boundary-snapshot sessions of dead PIDs (free wins),
  2. cluster dirs of dead deployments (R3's check, if cluster runtime is
     present on this machine),
  3. normal three-index LRU via `enforce_size_limit()` (which already
     interleaves incompatible blocks and sidecars fairly — after A2's fix,
     fairly in fact and not just in intent),
  4. vision-feature cache LRU (its own manager),
  5. never: logs (bounded separately by R4), models, configs.
- Each tick also runs A5's `reconcile_tracked_sizes()` batch (stat-check
  the 32 LRU-tail entries; prune dead ones) so multi-manager drift decays.

**Effort:** M. **Risk:** low-medium — the new behavior under the floors
must be exercised with a fault-injection test (statvfs monkeypatched), and
the refusal path needs a stat + admin-dashboard surfacing so operators see
"cache disabled by disk pressure" rather than mystery cache misses.

### R3 — Cluster deployment reaper (fixes D1, D2, D3)

**Trigger + placement:** each **worker node** reaps its own disk (the
coordinator can't see peer disks): in `inference_worker.py`, right where
the store directory is computed (`_prompt_cache_ssd_dir`), before
constructing the store — i.e. at every activation, the moment the premise
"next store reclaims" was supposed to hold. Additionally on the
coordinator's R2 tick for the local `~/.omlx/cluster/runtime` (covers the
"cluster abandoned, never activated again" case — exactly today's state).

**Action:** enumerate `prompt-cache-ssd/*/`; delete any directory whose
name's `<deployment_id>` (a) is not in `deployments.json`'s active set,
(b) has no live runtime-marker owner (`marker_owner_is_live`), and (c) has
mtime idle >1 h. The live deployment's own dir is excluded by (a). Also
prune runtime markers per D3 (keep newest N=3 per model, delete >30 days
with dead owner). Wire `max_bytes` into the store construction
(telemetry.py:679-681) per D2.

**Concurrency note:** safe against a live cluster by construction — the
boundary vote treats missing files as unvoted boundaries, and the reaper
never enters a directory whose deployment is registered or whose marker
owner is alive. The only rank that could be writing into a dir is the one
that owns it, and it's alive by definition.

**Effort:** S. **Risk:** low. This is the highest ratio of reclaimed bytes
(30 GiB today, recurring) to effort in the whole plan.

### R4 — Log rotation + transient-artifact reaper (fixes E1, E3)

**Trigger:** server startup + the R2 tick (cheap: one directory listing).

**Policy table:**

| Pattern (allowlist) | Rule |
|---|---|
| `logs/server.log.\d{4}-\d{2}-\d{2}` | keep newest 7 (backstop for TRFH's restart gap) |
| `logs/fork-*.log`, `logs/crash.log`, `logs/installed-*.log` | delete >14 days |
| `logs/watchdog.log` | truncate to last 1 MiB when >5 MiB |
| `logs/` total | cap 500 MiB, delete oldest-by-mtime first, never the live `server.log` |
| `cache/response-state/*.tmp` | >1 h (also in R1) |
| `models/**/.cache/huggingface/download/._____temp/` | delete when no active download task and >7 days |
| `models/**/.cache/**/*.lock` | delete when no active download task and >7 days |
| `~/.omlx/__pycache__/` | delete (regenerable) |
| everything else in `~/.omlx` root | **report only** — admin endpoint lists unrecognized files + sizes; no deletion |

**Effort:** S. **Risk:** minimal; every deletion target is an allowlisted
transient with an age gate.

---

## 8. Phased implementation checklist

Ordering: measure first, then reclaim the confirmed-dead bytes, then the
guards, then policy tuning. Effort tags S(<~1 h) / M(half-day) /
L(multi-day). All line refs verified 2026-08-25 @ `2718845b`. Nothing in
Phase 0 deletes anything.

### Phase 0 — Instrumentation and measurement (repeatable, not one-off)

- [x] **0.1** [S] Add a disk-footprint breakdown to the stats surface:
  extend `get_stats_dict` (paged_ssd_cache.py:4911) / the admin stats
  endpoint with per-tier bytes+counts (main blocks, incompatible,
  sidecars, boundary-snapshot live bytes, and — from the settings layer —
  cluster `prompt-cache-ssd` and `logs/` totals plus volume free bytes).
  This turns this doc's one-off `du`/`stat` numbers into a trackable
  series. Baseline recorded 2026-08-25 in §1's table.
  Implemented 2026-08-26, split across two pieces: `get_stats_dict()` now
  reports `compatible_block_count/size_bytes`,
  `incompatible_block_count/size_bytes`, and `disk_pressure_hard`
  alongside the pre-existing `gdn_sidecar_count/size_bytes` (per-manager,
  per-tier — landed with 2.1 since both touch the same method). The
  cross-consumer half (cluster `prompt-cache-ssd`, `logs/`, volume free
  bytes) is 0.2's `disk_footprint_report.py`, which reports exactly that
  breakdown independent of any one manager. No new admin-dashboard UI —
  both are JSON-only surfaces; a visual banner is a scoped-down follow-up
  if the operator-facing case comes up (see 2.1's note).
- [x] **0.2** [S] Growth-rate probe: a `scripts/disk_footprint_report.py`
  (read-only) that emits the §1 table + per-day mtime histogram
  (`stat -f '%Sm %z' -t '%F'` equivalent) as JSON; run before/after the
  Phase 1 reapers to quantify reclaim. Keep it for regression checks.
  Implemented 2026-08-26. Emits per-area bytes+counts (models, cache main
  blocks, GDN sidecars, boundary snapshots, response-state, vision
  features, cluster prompt-cache-ssd, cluster rest, logs, root loose
  files), volume total/used/free, and a per-day mtime histogram across
  the cache tree. Smoke-tested against a synthetic tree only — per this
  session's own constraint, not run against the real `~/.omlx` (would
  need to happen after this branch actually deploys, as the "after"
  baseline; the doc's own 2026-08-25 numbers remain the "before").
- [ ] **0.3** [S] Observe one long-context request's
  `_boundary_snapshots` high-water mark (watch `du` during a 100k+ prefill
  on the Qwen3.6/3.8 GDN model) to confirm B1's ~150 MB/boundary
  derivation before sizing its cap.
  Not done — genuinely an operator/runtime step (drive a real long-context
  request against a live deployed server and watch disk), not something
  to run from this session per the standing "no deletion/production code
  runs against real ~/.omlx or a live workload this session" constraint.
  Left unchecked deliberately; gates 3.1.
- [x] **0.4** [S] Log tier hit-rates per store: sidecar restore hits vs
  main-block hits (stats already partially exist:
  `gdn_legacy_fp32_fallbacks`, `hits`) to ground A3's budget-share
  decision.
  Implemented 2026-08-26: new `main_block_ssd_hits` and
  `gdn_sidecar_restore_hits` stats, isolating the SSD-tier split from the
  pre-existing `hits`/`hot_cache_hits` counters that lumped every tier
  together. 2 new tests.

### Phase 1 — Reclaim confirmed-dead bytes, lowest risk (≈30 GiB immediate)

- [x] **1.1** [S] Cluster deployment reaper (R3): sibling-dir sweep in
  `inference_worker._prompt_cache_ssd_dir` call site + coordinator-side
  sweep of `~/.omlx/cluster/runtime/prompt-cache-ssd` gated on
  registry-absent + dead-marker + idle >1 h. Reclaims the measured 30 GiB
  (both Macs: ~2×). (§D1)
  Implemented 2026-08-26 on `feat/disk-cleanup-routines` (off `origin/main`):
  `_reap_dead_prompt_cache_ssd_dirs` in `inference_worker.py`, called from
  `main()` right before `install_server_telemetry` (kept out of the pure
  `_prompt_cache_ssd_dir` helper — that helper is called directly by
  existing tests with the real default `~/.omlx/cluster/runtime` state_dir,
  so any I/O inside it would have run against live data under test). Checks
  registry membership (`ClusterRegistry.list()`, aborting the pass entirely
  if `load_error` is set — a corrupt registry must not read as "nothing is
  active"), marker liveness (`marker_owner_is_live`), and a 1 h idle gate.
  9 new tests in `test_cluster_prompt_cache_ssd_reaper.py`.
  **Gap closed 2026-08-26**: the worker-side reap alone only fires on the
  *next* activation of a deployment, which never happens for a cluster
  that was torn down and never relaunched — exactly this doc's measured
  state (registry empty, 30 GiB dead). Added
  `_reap_dead_cluster_prompt_cache_ssd_dirs_for_server()` in `server.py`,
  called from `lifespan()` at every server start (reuses the identical
  worker-side function against `base_path/cluster/runtime`). 3 new tests
  in `test_server.py`. This is the half that actually reaches the headline
  30 GiB.
- [x] **1.2** [S] Pass `max_bytes` to `SSDPromptSnapshotStore` wiring
  (telemetry.py:679-681). (§D2)
  Implemented 2026-08-26: `_ssd_snapshot_max_bytes()` = min(32 GiB, 20% of
  free disk at activation), falls back to the flat cap on `OSError`. 5 new
  tests in `test_cluster_telemetry.py`.
- [x] **1.3** [S] `_tmp.safetensors` + unreadable-final + empty-digest-dir
  sweep in `_scan_existing_files` / `_scan_existing_gdn_sidecars` /
  vision scan (R1 steps 1-3), age-gated 10 min. (§A1)
  Implemented 2026-08-26: `_reap_stale_tmp_staging_files`,
  `_reap_corrupt_final_file`, `_reap_gdn_sidecar_digest_dirs` in
  `paged_ssd_cache.py`, wired into `_scan_existing_files` /
  `_scan_existing_gdn_sidecars`. Flipped two pre-existing tests that
  asserted the old "corrupt file survives forever" behavior. 7 new tests
  in `test_paged_ssd_cache.py`.
  **Gap closed 2026-08-26**: the doc's 1.3 text explicitly includes the
  vision scan ("Same treatment in the sidecar and vision scans") and it
  has the identical A1 orphan mechanism (`vision_feature_cache.py`'s own
  `*_tmp.safetensors` staging + skip-but-keep on unreadable/missing
  metadata). Added the same `_reap_stale_tmp_staging_files` /
  `_reap_corrupt_final_file` pair there, wired into its
  `_scan_existing_files`. 5 new tests in `test_vision_feature_cache.py`.
- [x] **1.4** [S] Extend `/api/ssd-cache/clear`'s filesystem fallback to
  `_gdn_sidecars` (+ `vision_features`), with symlink refusal. (§A4)
  Implemented 2026-08-26 in `admin/routes.py`'s `clear_ssd_cache`, reusing
  the same resolve()-inside-root symlink check pattern. 4 new tests in
  `test_admin_ssd_cache_clear.py`.
- [x] **1.5** [S] `ResponseStore` `.tmp` reaper in
  `_load_persisted_records`. (§E2)
  Implemented 2026-08-26: `_reap_stale_tmp_records`, 1 h age gate. 2 new
  tests in `test_responses.py`.
- [x] **1.6** [S] Runtime-marker pruning (keep newest 3/model, >30 d dead
  owners). (§D3)
  Implemented 2026-08-26: `_prune_runtime_markers` in `inference_worker.py`,
  called unconditionally (independent of the `prompt_cache_ssd` flag)
  alongside 1.1's reap call. Keep-window is per-model and never evicts a
  live-owner marker regardless of age. 7 new tests in
  `test_cluster_runtime_marker_pruning.py`.

Phase 1 total: 1,031 tests passing in the cluster+inference_worker slice
(1,019 baseline + 12 new), 184 passing in the SSD-cache slice (177 + 7),
all green. No deletion code has run against the real `~/.omlx` — every
test uses `tmp_path`-rooted synthetic trees per the session's own
constraint on this work.

### Phase 2 — Guards and recurring hygiene

- [x] **2.1** [M] Disk-pressure guard tick (R2): periodic
  `enforce_size_limit()` (its first caller), soft/hard free floors with
  save-refusal + stat + dashboard surfacing, ENOSPC kick. Fault-injection
  test with mocked `shutil.disk_usage`. (§C1)
  Implemented 2026-08-26. New `DiskSettings` (settings.py, mirrors the
  memory guard's soft/hard-threshold shape: `soft_free_floor_gb`/
  `_fraction`, `hard_free_floor_gb`/`_fraction`, `guard_tick_interval_seconds`,
  default 60s). New `omlx/disk_pressure_guard.py`: a pure, injectable
  tick (`run_disk_pressure_guard_tick`) plus the asyncio loop
  (`disk_pressure_guard_loop`), scheduled as a bare task from
  `lifespan()` (not the memory-enforcer piggyback, per design review) and
  cancelled on shutdown alongside `ttl_task`. On soft breach: scales
  `enforce_size_limit()`'s trigger/target fractions by `free/soft_floor`
  (clamped ≥0.1) across every live manager — `enforce_size_limit` gained
  `trigger_fraction`/`target_fraction` params, default-call behavior
  proven identical via `int(effective_max * 1.0) == effective_max`. On
  hard breach: `manager.set_disk_pressure_hard(True)` +
  `store.set_disk_pressure_hard(True)` — `save_block`, `commit_gdn_checkpoint_file`,
  `_enqueue_ssd_write` (the hot-cache-spill choke point), and
  `BoundarySnapshotSSDStore.save` all become refusal no-ops (new
  `saves_refused_disk_pressure` stat), reusing the existing
  failed-save/placeholder degrade path — never a request refusal.
  ENOSPC "kick": not built as a separate immediate-trigger path — the
  writer thread's existing ENOSPC-invalidates-disk-cache behavior plus
  the 60s tick cadence already bounds exposure; a dedicated kick is a
  scoped-down follow-up, not silently dropped. Dashboard surfacing is
  JSON-only (`get_stats_dict()`'s `disk_pressure_hard` field) — no new
  admin-dashboard banner UI this pass (would reopen the i18n
  split-convention landmine from earlier this session for a `main`-based
  branch; a visible operator banner is a good follow-up once the pattern
  is needed). **Regression caught and fixed during this item**: the
  scheduler-enumeration closure originally reused
  `admin.routes._iter_loaded_schedulers`, which calls the FastAPI
  dependency `get_engine_pool()` — that function *raises* `HTTPException`
  when no pool exists yet (correct inside a route handler; wrong from a
  bare background task, where nothing catches it as an HTTP response, so
  it looked like a tick crash on every server start before any model
  loaded). Fixed with a local, non-raising enumerator reading
  `_server_state.engine_pool` directly; caught by a real-lifespan
  regression test (`TestDiskPressureGuardTaskLifecycle`) that asserts no
  "tick raised" warning is logged — verified the test actually fails
  without the fix, not just that it passes with it. 13 tests in
  `test_disk_pressure_guard.py`, 3 in `test_server.py`, plus save-refusal
  coverage across `test_paged_ssd_cache.py`, `test_gdn_sidecar_index.py`,
  `test_boundary_snapshot_store.py`.
- [x] **2.2** [S] Main-block LRU persistence: throttled `os.utime` on
  cold hit in `load_block`/`load_block_with_metadata`, mirroring the
  sidecar path (2509-2515). Do before or with 2.1 so pressure eviction
  acts on real recency. (§A2)
  Implemented 2026-08-26 (landed before 2.1, per the doc's own ordering
  note): `_persist_main_block_lru_touch`, throttled on the
  previously-recorded `last_access` (captured before `_index.touch()`
  mutates it in place) rather than an extra `stat()` call — one syscall
  per cold hit that actually needed it, none on a chain hit repeatedly
  inside the 1h throttle window. 3 new tests in `test_paged_ssd_cache.py`.
- [x] **2.3** [S] Log/artifact reaper (R4) per the policy table. (§E1, §E3)
  Implemented 2026-08-26: new `omlx/log_artifact_reaper.py`
  (`run_log_artifact_reaper`), covering the full policy table — dated
  `server.log.*` rotation backstop (keep newest 7), ad-hoc log class
  (`fork-*.log`/`crash.log`/`installed-*.log`, >14d), `watchdog.log`
  truncation (>5 MiB → last 1 MiB), `logs/` directory-wide 500 MiB cap
  (oldest-first, never the live `server.log`), HF download staging
  (`._____temp` dirs + `*.lock` files, >7d), and `__pycache__` removal.
  Trigger is "server startup + the R2 tick" exactly as specified: called
  once from `lifespan()` and threaded through
  `disk_pressure_guard_loop`'s new `run_log_reaper` param, run
  independent of pressure tier (log hygiene isn't a pressure response).
  **Known limitation, stated rather than hidden**: the HF staging sweep
  is age-gated only, with no cross-reference to the downloader's live
  task registry (the doc's policy table calls for "no active download
  task" as a joint condition) — an active transfer touches its staging
  files continuously, so 7 days of *silence* is strong evidence of
  abandonment, but this is a weaker guarantee than an explicit registry
  check would give. 16 tests in `test_log_artifact_reaper.py`, 3 more in
  `test_disk_pressure_guard.py` for the tick wiring, 3 in `test_server.py`
  for the startup call.
- [x] **2.4** [S] Per-session boundary-snapshot reset (dead-PID check in
  `reset_boundary_snapshot_root`). (§B2)
  Implemented 2026-08-26: per-session dead-PID check
  (`_session_owner_pid_is_live`, a local copy of
  `cluster/liveness.py`'s `_pid_is_live` rather than an import — same
  reasoning that module states for keeping its own copy local, so
  `cache/` carries no dependency on the optional cluster package). A
  session dir whose name doesn't parse as `<pid>-<uuid>` (foreign debris,
  never created by this module) is reaped outright, same treatment as a
  malformed name anywhere else in the tree (§A1); a dir with a live pid
  survives regardless of session-dir age. In the common single-process
  case every existing session belongs to a now-dead prior invocation, so
  this reduces to exactly the old blanket-rmtree behavior — verified with
  a dedicated test. 5 new tests in `test_boundary_snapshot_store.py`.
- [x] **2.5** [S] `reconcile_tracked_sizes()` LRU-tail stat batch on the
  guard tick. (§A5)
  Implemented 2026-08-26: `PagedSSDCacheManager.reconcile_tracked_sizes()`
  stat-checks a bounded 32-entry LRU-tail batch across all three indexes
  (main/incompatible/sidecar), pruning index entries whose file another
  manager already unlinked. Read-only stat + in-memory index mutation
  only — never unlinks a file (a dead entry is dead precisely because the
  file is already gone, so there's nothing to unlink). Wired into every
  guard tick, independent of pressure tier. 2 new tests in
  `test_paged_ssd_cache.py`, 2 in `test_disk_pressure_guard.py`.

Phase 2 total: full suite green at 10,422 passed (up from the 10,365
Phase-1 baseline), 102 skipped, 77 deselected. Still true throughout:
no deletion/production code has run against the real `~/.omlx` — every
test uses `tmp_path`-rooted synthetic trees.

### Phase 3 — Policy, gated on Phase 0 data

- [ ] **3.1** [M] Boundary-snapshot disk high-water mark with
  oldest-interior-first degradation to the existing placeholder path
  (gate on 0.3's measured peak). (§B1)
- [ ] **3.2** [S] Orphan-sidecar collection (R1 step 4), flag-gated first
  release (gate on 0.4 showing orphans exist in practice). (§A3b)
- [ ] **3.3** [M] Sidecar budget share, only if 0.4 shows main-block
  eviction starving restores after 2.2 landed — 2.2 alone may fix the
  bias. (§A3a)

### Explicitly not doing (this review)

- **Cross-manager index coherence** (shared index / file locks) for the
  multi-manager-same-dir case — self-healing plus 2.5's decay is enough;
  a coordination layer is L-effort and adds failure modes. (§A5)
- **Payload checksums** in the block format — already tracked as the
  open half of qwen35-doc F1; orthogonal to cleanup.
- **The cluster store's synchronous-write performance** — that is
  clustering-doc D2/3.2's item; this doc only bounds its bytes.
- **Auto-deleting anything user-created** in `~/.omlx` root (backups,
  benchmarks, `.bak` files) — report-only, per E3.

---

## 9. Already handled / do-not-touch

### Working machinery a cleanup effort must not re-implement or break

- The **shared LRU budget + deterministic three-index eviction walk**
  (§1 "what bounds exist") — at cap and holding on the live machine.
- The **save-time disk clamp** (`_get_effective_max_size`, 99% ratio, 30 s
  TTL) and its ENOSPC invalidation — R2 layers on top; it does not replace
  them.
- **Atomic, fsync'd writes everywhere** (temp → fsync → rename → dir
  fsync) — any reaper must respect the `_tmp` staging convention and its
  age gate.
- **Startup reset of `_boundary_snapshots`** (server.py:423,598) and the
  writer-barrier request cleanup (boundary_snapshot_store.py:596-804) —
  R1 refines the reset to per-session; do not remove the reset.
- **Read-path self-healing** (missing/corrupt file → index prune, unlink
  on corrupt) — this is what makes out-of-band deletion *survivable*; it
  is not a license to delete out-of-band (has_block staleness, A5).
- **`ResponseStore` count-bound eviction**, **vision-cache own budget**,
  **HF download partial cleanup on cancel**, **`server.log` daily
  rotation** — all present and adequate modulo the listed edge cases.
- **Cluster boundary-vote tolerance of missing files** — the property R3
  relies on; preserve it in any future cluster-store change.

### Load-bearing paths a reaper must never touch (hard denylist)

- `~/.omlx/models/**` except the two allowlisted download-staging
  patterns (`._____temp`, stale `.lock`) — **weights are not cache**.
- `~/.omlx/settings.json`, `model_settings.json`, `model_profiles.json`,
  `stats.json`, `~/.omlx/bin/**` (the `omlx` launcher and watchdog script
  are what restarts the server).
- `~/.omlx/cluster/{deployments,incidents,fabric-intent}.json` and
  `strategy-benchmarks.json` — durable product state (the benchmark store
  feeds strategy selection).
- The **live deployment's** `prompt-cache-ssd/<id>-rank-N` dir and any
  runtime marker with a live owner.
- User artifacts in `~/.omlx` root (backups, `benchmark_*`,
  `model_settings.json.bak.*`, `cluster_bench_*.json`) — report-only.
- Anything inside the installed app bundle (`/Applications/oMLX.app/**`) —
  the packaged app is 50+ files behind this repo and its compiled-kernel
  wrapper files are fragile; no cleanup routine has any business there.
