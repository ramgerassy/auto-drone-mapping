# Project progress & decision log

A running journal of non-obvious decisions, trade-offs, and thoughts worth
remembering — the *why* behind choices that the code and git history don't
capture on their own. Newest entries at the top. Each entry records what was
decided, why, and its current status.

---

## 2026-08-01 — Path-planning algorithm: A\* chosen over octree / potential fields / RRT\* / wavefront

Reviewed the Modern Robotics (Ch. 10) alternatives to grid A\*. **Decision:
plain A\* on the 2D occupancy grid** for Sprint 2's path planner.

The alternatives are each the right tool for a problem we deliberately *don't*
have. Two constraints dissolve most of their motivation:
- **We plan in 2D, not 3D.** The 2.5D map is a 2D grid + per-cell height; drones
  fly at a fixed altitude, so planning is over `(col, row)`. The `kⁿ`
  exponential-scaling worry that motivates octrees/sampling is about high `n`;
  at `n = 2` with ≤250k cells it's milliseconds.
- **Sprint 2 uses teleport movement, no dynamics.** "Smooth flyable trajectories
  / RRT\* + smoothing" assumes dynamic flight — deferred to a possible later
  physics sprint. The drone steps cell-to-cell, so a grid path *is* the output.

Plus: hard determinism requirement, and we already maintain a uniform 2D grid.

Per method:
- **Octree / multi-resolution grid** — solves 3D memory/scaling we don't have;
  would re-architect the (deliberately uniform) mapping grid for zero benefit.
- **Potential fields / navigation functions** — reactive paradigm with local
  minima; nav-functions need the full map to construct anyway; outputs a heading
  not a path; clashes with our deliberative map→frontier→path pipeline.
- **RRT / bi-RRT / RRT\*** — randomized (breaks determinism), only
  *asymptotically* optimal (A\* is exact-optimal on a grid immediately), heavy
  machinery; its wins (high-D, continuous, dynamics-aware) are all out of scope.
- **Wavefront** — closest relative, but it's "A\* with no heuristic": floods the
  whole reachable grid per query, and its compute-once reuse needs many-to-one
  goals (we have each drone to a *different* frontier). A\* dominates point-to-point.

**Revisit:** sampling methods (RRT\*) + path smoothing become genuinely relevant
in the deferred **physics-based flight sprint** — bookmarked there, not now.

---

## 2026-08-01 — Multiprocessing the probability / frontier calc — rejected

Considered detecting the core count, splitting the occupancy grid into
`n_cores` chunks, and computing `probability()` / the frontier masks in
parallel subprocesses. **Rejected.**

Why it loses here:
- The calc is a **vectorized, elementwise numpy pass** — already a tight C loop,
  and **memory-bandwidth bound, not CPU bound**. Extra cores don't widen the
  memory bus, so they'd mostly wait on the same RAM.
- **Overhead dwarfs the work.** The whole `probability()` call is ~1 ms on the
  250k-cell grid. On Windows, multiprocessing uses *spawn* (a fresh interpreter
  + numpy re-import per worker, ~100–300 ms each), plus pickling/copying grid
  chunks in and results out. Net: a ~1 ms op becomes ~100–300 ms — a 100×+
  regression.
- **Determinism + constraints.** CLAUDE.md bans threading/async/multiprocessing
  in the tick loop and requires deterministic runs; a parallel split/reassemble
  invites ordering / floating-point nondeterminism.

Rule of thumb recorded: **parallelize only when the work is large and CPU-bound
and the coordination cost (spawn + IPC) is small relative to it.** Here the ratio
is inverted (~1 ms work vs ~200 ms overhead). If frontier detection ever profiles
as hot, the levers, in order, are: A (skip the `exp`), C (call it less often /
incremental), then a compiled kernel (numba/cython) — more cores would be the
last resort and only for a genuinely CPU-bound, seconds-scale workload.

---

## 2026-08-01 — `detect_frontiers` performance options (A / B / C / D)

Reviewed whether frontier detection is a heavy calculation. It isn't: overall
cost is **O(N + F)** — an O(N) *vectorized* numpy pass over all `N` grid cells
(runs in C) plus O(F) Python work over the frontier *perimeter* `F`, which is
far smaller than `N`. The scary-looking nested loops run over `F`, not `N`.

Options considered, with status:

- **A — skip the sigmoid.** Threshold `grid.log_odds` directly instead of
  calling `grid.probability()`, avoiding one `exp()` + divide over all `N`
  cells (~1 ms on the 250k-cell grid). Output is identical because probability
  is monotonic in log-odds.
  **Status: DEFERRED.** Costs readability (a magic `-0.405` vs a plain
  `p < 0.4`) and leaks grid's internal log-odds encoding into `frontier.py`,
  breaking the clean `probability()` seam. ~1 ms only matters if this is hot.
  Revisit **only if** Feature 4 profiling flags `detect_frontiers` as a
  hotspot (depends on whether the coordinator calls it every tick or only on
  reassignment).

- **B — reuse world coords.** In the representative-cell selection, reuse the
  `world` list already built for the centroid instead of calling
  `grid_to_world` a second time per cell. Pure win, no downside.
  **Status: APPLIED** (commit `a7d79d9`, PR #7).

- **C — incremental detection.** Recompute frontiers only near cells that
  changed this tick and cache the rest (O(N) → O(changed)). Real win, but adds
  dirty-region tracking + cache-invalidation complexity.
  **Status: DEFERRED (likely never).** Premature for 1–5 drones; CLAUDE.md:
  prefer the simple thing, measure first.

- **D — `scipy.ndimage.label` for clustering.** C-level connected components,
  faster than the hand-written BFS, but adds the scipy dependency.
  **Status: DECLINED.** Same scipy option already turned down in the clustering
  decision below; BFS over `F` cells is fast enough.

Guiding principle: **measure before optimizing; don't trade a clean seam or
readability for a millisecond without evidence.**

---

## 2026-08-01 — Frontier clustering algorithm (Sprint 2, Feature 1)

How to group frontier cells into regions. Options weighed: (a) hand-written BFS
connected-components, (b) `scipy.ndimage.label`, (c) `sklearn` DBSCAN.

**Decision: (a) hand-written BFS connected-components**, **8-connectivity**,
`min_region_size = 2`.

Why:
- **No new dependency** — scipy / scikit-learn are heavy and unneeded here.
- **Deterministic by construction** — a hard project requirement (same config +
  seed ⇒ identical run); BFS with fixed iteration/neighbor order guarantees it.
- **Simple** (~20 lines), full control — matches CLAUDE.md "prefer the simple
  thing."

Connectivity detail: the frontier *neighbor test* uses **4-connectivity** (the
standard frontier definition — free cell orthogonally adjacent to unknown),
while *clustering* uses **8-connectivity** so diagonally-touching frontier cells
merge into one region.

Detail doc: [`docs/sprint-2/feature-1-frontier-detection.md`](sprint-2/feature-1-frontier-detection.md).

---

## 2026-08-15 — Frontier scoring + spreading penalty (Sprint 2, Feature 3)

Two decisions for `NearestFrontier`, the first implementation behind the
`FrontierStrategy` seam.

**D1 — what "nearest" measures. Decision: true A\* path cost.** Options weighed:
(a) A\* to every candidate, (b) Euclidean distance to the region centroid,
(c) hybrid — Euclidean-sort then A\* the top *K*.

Why (a):
- **Wall-aware.** Verified on the Feature-3 test grid: a frontier 3.0 m away in
  straight line sits behind a wall at a true cost of **104**, while one 4.0 m
  away down open space costs **40**. Euclidean scoring picks the wrong one; this
  is `test_euclidean_near_frontier_behind_wall_loses`.
- **Reachability for free.** An unreachable frontier returns no path and is
  skipped, instead of being assigned and failing a tick later.
- **Cost is affordable** at 1–5 drones (CLAUDE.md: "don't optimize for 50").
  Option (c) stays available as a pure internal change — it does not touch the
  seam — if the large-indoor scenario (Feature 5) shows a real tick-rate
  problem. **Status: DEFERRED, revisit only with measurements.**

**D2 — spatial spreading penalty. Decision: hard exclusion + soft radius.**
A claimed region is never re-selected; a candidate whose centroid falls within
`spread_radius` of a claimed centroid gets `spread_penalty` added to its *score*.

Why:
- Hard-exclusion-only under-spreads (two drones happily work frontiers 0.5 m
  apart); soft-penalty-only permits the *same* frontier being assigned twice.
- The penalty is **additive and integer** (same 10/14 units as A\*), so ranking
  stays in exact integer arithmetic — no float comparison in a decision path.
- The penalty affects **ranking only**; `FrontierAssignment.cost` reports the
  true unpenalized path cost, keeping it honest for the path-length KPI.
- Defaults (`spread_radius=0.0`, `spread_penalty=0`) leave the penalty off, so
  single-drone behaviour is unchanged until Feature 6 wires up config.

Detail doc: [`docs/sprint-2/feature-3-frontier-strategy.md`](sprint-2/feature-3-frontier-strategy.md).
