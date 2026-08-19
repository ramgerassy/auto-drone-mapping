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

---

## 2026-08-15 — Inter-drone collision avoidance (Sprint 2, Feature 4)

The "zero collisions during nominal ops" KPI had no definition, detector, or
enforcement anywhere in the repo — `collision` appeared only in the KPI lines
themselves. Half of it was already satisfied by construction: A\* traverses only
free cells and refuses corner-cutting, so a planned path cannot clip an
obstacle. The drone↔drone half was unaddressed. Folded into Feature 4, since
multi-drone does not exist before it.

**Decision: wait-on-conflict, higher drone ID has right of way.** When two
drones would occupy the same cell, the lower-ID drone holds position until the
cell clears. Not re-planning: movement is teleport (Decision 2), so there is no
momentum and holding costs nothing, whereas dynamic-obstacle re-planning is
strictly more complex and two drones re-routing around each other can oscillate.

Consequences to honour in the implementation:

- **Precedence must drive iteration order.** Whichever drone is processed first
  reserves its cell first, so an ascending walk would silently make precedence
  first-come-first-served and invert the rule. The tick therefore iterates
  drones **descending by ID** — uniformly, for both frontier assignment and
  movement, so there is one seniority rule rather than two orders in one tick.
- **Swap conflicts.** If drone 3 steps into drone 1's cell while drone 1 steps
  into drone 3's, both targets read as free under a naive occupancy test and
  they pass through each other. *Resolved in the Feature 4 plan without a
  separate check:* seeding the reservation table with every drone's **current**
  cell means an unprocessed drone still blocks its own cell, so the swap can
  never be reserved. One conservative rule covers both conflict classes.
- **Separation is a distance, not a cell.** At `resolution: 0.1` a cell is 10 cm
  and a drone body is several times that, so the threshold is a config value in
  metres converted to a cell radius — never hardcoded to one cell.
- **Deadlock escape.** Two drones head-on in a corridor wait on each other
  forever under a pure wait rule. After N consecutive waited ticks the waiting
  drone drops its frontier claim and re-selects. This is the only place
  re-planning enters, as a remedy rather than the mechanism.

---

## 2026-08-15 — Drones must not be mapped as obstacles (Sprint 2, Feature 4)

Found while reasoning about what happens when two drones approach each other:
`mujoco.mj_ray`'s `bodyexclude` takes a **single** body id — the sensing drone's
own — and `raycaster.py` passes `geomgroup=None` (all groups hittable). With
multiple drones injected as real bodies, drone 1 excludes itself and then ranges
drones 2-5 as though they were walls. Three consequences:

1. **Occupancy.** A false hit needs ~2-3 later free observations to wash out
   (`log_odds_occ` +0.847 vs `log_odds_free` -0.405), but a repeatedly-scanned
   drone saturates at the `+5.0` clamp, from which recovery takes ~14
   consecutive free observations.
2. **Planning.** A\* blocks occupied *and* unknown cells, so a phantom obstacle
   can sever a corridor and delete the frontier cells it sat on — a drone can
   conclude a region is unreachable because a teammate was standing there.
3. **Height — permanent.** `update_occupied` does
   `height[row, col] = max(height[row, col], hit_z)`: monotonic, no decay. A
   drone seen at 1.0 m altitude writes height 1.0 into a floor cell *forever*,
   surviving the occupancy recovery. The occupancy channel self-heals; the
   height channel has no recovery path at all.

The wait-on-conflict rule above **amplifies** this: it pins the lower-ID drone
stationary precisely while the higher-ID drone passes close by, and a stationary
target scanned across consecutive ticks is the fastest route to +5.0 saturation.

**Decision: filter teammate returns in `perception`.** Options weighed:
(a) exclude drones at the ray-cast layer via a `geomgroup` mask, so rays never
hit a drone at all; (b) filter hit points against known teammate poses in
`perception`; (c) give `mapping` the drone positions so it declines to mark
those cells.

Chosen **(b)**. Why:

- **No seam change.** `Sensor.scan(drone_id)` takes only an id, and
  `Rangefinder` already holds the engine reference it uses for `get_pose` — so
  it can query teammates' poses without touching the `Sensor` interface, which
  CLAUDE.md lists as a stability point.
- **Models what a real swarm does.** Filtering known teammate positions out of a
  scan (shared telemetry) is the actual technique. (a) would instead make the
  simulator lie about what the sensor saw, pushing an autonomy concern into the
  environment model.
- **Occlusion stays honest.** A real LiDAR cannot see *through* a teammate.
  Under (a) rays would pass straight through and map the wall behind — better
  coverage, but physically wrong. Under (b) the space behind a teammate stays
  unknown for that tick and is filled in on a later pass. Accepted cost:
  marginally slower coverage, in exchange for a sensor model that does not
  cheat.
- **(c) DECLINED.** It patches the end of the chain, after the bad reading has
  already travelled raycast → perception → mapper, would still need the height
  write suppressed separately, and puts per-tick swarm awareness into the one
  module CLAUDE.md says depends on nothing.

Cost accepted: (b) needs a tuned exclusion radius (drone body half-extent plus a
margin) as a config value, where (a) would have needed none. Ground-truth poses
are exact, so the radius does not need to absorb localization error. A real wall
directly behind a teammate and within that radius is discarded too — that cell
simply stays unknown for the tick.

**Encoding: a filtered ray becomes a MISS with a shortened range.**
`RayObservation` already carries `max_range` alongside `distance`/`hit_point`,
and the mapper traces a MISS out to `obs.max_range` (`mapper.py:73-75`). So a
ray that hits a teammate is emitted as `distance=None, hit_point=None,
max_range=<distance to the teammate>`: cells up to the teammate are marked free
(the ray genuinely travelled that far unobstructed), nothing is marked occupied,
and nothing is claimed beyond. **No change to `mapping`, no change to the
`Sensor` seam.**

Rejected encodings: dropping the ray entirely throws away the legitimate
free-space evidence up to the teammate; emitting a full-range MISS would falsely
mark the occluded cells *behind* the teammate as free.

Tests to write with the fix: two drones in line of sight — assert no occupied
cell appears at the other drone's position, and that its height cell stays
`-inf`; assert the cells between the two drones *are* marked free; assert the
cells beyond the observed drone stay unknown (the occlusion shadow is
preserved).

---

## 2026-08-19 — The drone has a body; the planner assumes a point (Sprint 2, Feature 4c)

Raised while reading `CentralizedMaster._teleport`: if a cell is 10 cm and the
drone is 30 cm, and teleport places the drone's *centre* on the cell centre,
what stops a frontier cell adjacent to a wall from parking the drone half inside
that wall? Nothing does. Confirmed at every link in the chain:

- `_teleport` calls `grid_to_world(*cell)`, which returns the cell **centre**
  (`grid.py:79`), and `set_drone_position` writes it into `qpos[0:3]` — the
  freejoint origin. The body geom carries no `pos` offset
  (`engine.py:42`), so the box centre *is* the body origin. Centre on centre.
- The colliding footprint is `type="box" size="0.15 0.15 0.05"` — MuJoCo sizes
  are half-extents, so **0.30 m x 0.30 m**. (The four rotors reach 0.23 m but
  carry `contype="0" conaffinity="0"`; they are decoration.)
- At `resolution: 0.1` that is **3 cells wide, half-extent 1.5 cells** — while
  every planning decision treats the drone as dimensionless:
  `AStarPlanner.plan`'s `is_free(col, row)` tests one cell
  (`path_planner.py:105`), `NearestFrontier.select` plans to `region.cell` with
  no clearance test, and `assign_all` / `is_assignment_valid` check a single
  `prob[row, col]`. `grep -rn 'inflat|clearance|footprint' src/` returns nothing.

A cell centre is 0.05 m from the cell boundary, so with the wall face on that
boundary the body reaches 0.15 m — **0.10 m inside the wall**. This is not only
the frontier terminus: A\* routes *through* wall-adjacent cells too. MuJoCo will
not object, because `set_drone_position` calls `mj_forward`, never `mj_step` —
there is no contact resolution to push back. The drone simply clips through.

This is a latent Sprint-1 bug as well: the hardcoded patrol used
`x_range: [-8, 8]` inside 10 m walls, so it never came close enough to bite.

**Decision: obstacle inflation (configuration-space expansion) in `planning`.**
`AStarPlanner` gains a clearance radius and builds an inflated-free mask once
per `plan()` call — one pass over a 200x200 grid, cheap and obvious.

Not in `mapping`. The map is a record of the world, not of the body moving
through it; inflating the stored grid would corrupt the exported `.npz` and
break the >=98% per-cell accuracy KPI directly. C-space expansion is a property
of the robot doing the planning, which is exactly where `planning` sits.

**Square inflation is exact here, not conservative.** `set_drone_position`
resets the quaternion to identity on every teleport and no `mj_step` ever runs,
so the drone never rotates. An axis-aligned box against an axis-aligned grid
means Chebyshev (square) inflation of the half-extent is the true C-space
obstacle — no circumscribed-radius slack needed.

**The radius, exactly.** An occupied cell at Chebyshev distance `k` has its near
face at `(k - 0.5) * res`. No overlap requires `(k - 0.5) * res >= h`, so the
minimum legal distance is `k_min = ceil(h/res + 0.5)` and the inflation radius
is `r = k_min - 1`. At `h = 0.15, res = 0.1`: `k_min = 2`, **`r = 1`** — and
that is *grazing*, 0.15 m against 0.15 m with zero clearance. A corridor would
need `2r + 1 = 3` free cells (0.30 m), i.e. exactly the drone's width, touching
both walls. So the value wants a safety margin `m`, giving
`r = ceil((h + m)/res + 0.5) - 1`; a 5 cm margin yields **`r = 2`** and a
5-cell (0.50 m) minimum corridor. *(Correcting my own first pass at this: I
quoted `r = ceil(h/res) = 2` as the exact requirement. The exact requirement is
`r = 1`; `r = 2` is the margin-inclusive recommendation. The distinction is
load-bearing because it is the difference between a 0.30 m and a 0.50 m minimum
doorway in Feature 5.)*

Three consequences that need deciding in the Feature 4c plan, not just coding:

1. **Inflate only *known-occupied* cells, never unknown.** A frontier is by
   definition free-adjacent-to-unknown; inflate unknown and every frontier
   becomes unreachable and exploration halts on tick 1. Accepted cost: a drone
   sitting on a frontier overlaps unknown space that may turn out to be wall —
   inherent to exploration, and corrected as the map fills in.
2. **The start cell needs an escape hatch.** A drone that discovers a wall
   beside itself is suddenly inside the inflated zone, so `plan()` returns
   `None`, `assign_all` gives it no assignment, and if that holds for every
   drone `_complete` flips to `True` — the mission reports success over a
   half-unknown map. The start cell must always be traversable.
3. **It constrains Feature 5.** The large-indoor MJCF's doorways and corridors
   must exceed `2r + 1` cells of *usable* width or the map will be perfectly
   accurate and completely unnavigable.

**Open for the plan doc:** where the half-extent lives. It is currently
hardcoded as `0.15` inside `_build_drone_xml` (`engine.py:42`); putting a
clearance value in YAML duplicates it, and the two silently disagreeing is worse
than either. Candidates: a module constant in `engine.py` that both the MJCF
builder and config validation read, versus a config value validated against it.

**Already correct, for contrast:** `min_separation` is a config value in metres
converted to cells via `resolution` (`master.py:69`), so drone-to-drone spacing
was built with body size in mind from the start. Its floor should be documented
as the body diagonal, `0.30 * sqrt(2) ~= 0.424 m`; the tests' 0.5 m clears it.
Only drone-to-obstacle had no knob at all.

Scoped to its own branch rather than folded into the Feature 4 PR: it modifies
`AStarPlanner`, which is already merged, and touches the config schema.

### Addendum (same day) — the same bug class in `min_separation`

Walking `resolve_moves` line by line turned up the drone-to-drone half of this,
so it folds into 4c rather than getting its own branch.

**What the check actually does.** `master.py:68` converts `min_separation` from
metres to cells (`/ resolution`), and `movement.py:62-67` compares squared
Euclidean distance in cell units against `threshold_sq`. The left-hand side is
pure integer arithmetic, so only the threshold is float and determinism holds.
The forbidden zone around each drone is therefore a **disc** of radius
`min_separation_cells` — not "the same cell", and not a square.

**Correcting my own earlier claim** that this "took the drone size into
account": it does not. Nothing in `movement.py` or `master.py` reads drone
geometry — no half-extent, no reference to the body geom. `resolve_moves`
enforces whatever number it is handed. What is true is narrower: the knob has
the right *units* and the right *shape* — a metric radius that rescales with
`resolution`, rather than a `target != other_cell` test that would have baked in
"one cell is enough" and been wrong at any resolution finer than the drone.
That is the difference from the drone-to-obstacle case above, where there is no
knob at all. But it is a correctly-shaped knob with no correct default.

**Why the floor is the body *diagonal*, 0.424 m — not 0.30 m.** The test is
radial; the body is an axis-aligned box. Two such boxes overlap iff
`max(|dx|, |dy|) < 2h`, a Chebyshev condition, and `L_inf <= L_2` always — so
passing a radial test does *not* imply the boxes are clear. Take
`dx = dy = 0.212`: Euclidean distance is 0.30 (passes a 0.30 m threshold) while
Chebyshev is 0.212, so the boxes overlap by ~0.09 m. A diagonal approach slips
straight through. Making a radial test safe for a square body requires the
**circumscribed** diameter, `2h * sqrt(2) = 0.30 * sqrt(2) ~= 0.424 m`. The
tests' 0.5 m clears it; anything between 0.30 and 0.424 would look reasonable
and silently permit diagonal overlap.

*Rejected:* switching `resolve_moves` to a Chebyshev test, which would be exact
for a non-rotating box and permit tighter packing. Tighter packing is worth
nothing at 1-5 drones, and the change would alter the shipped semantics and the
test that pins them (`test_diagonal_separation_uses_euclidean_distance`). Keep
the radial test; enforce the diagonal as the floor.

**Three guards for 4c:**

1. **`min_separation` floor.** Currently a required arg with no default and no
   validation (`grep` confirms it is only ever passed through). Reject anything
   below the body diagonal at construction.
2. **Start-position validation.** Nothing checks initial spacing, and
   `resolve_moves` only prevents *new* violations — it never repairs an existing
   one. Two drones spawned inside each other's disc find every target blocked,
   burn through `max_wait_ticks`, re-select, and stay frozen: a permanent
   deadlock from config alone. Pairwise check at construction; O(n^2) at n <= 5.
3. **Swept motion, noted not fixed.** Only end-of-tick positions are compared,
   never the motion between them — and movement is teleport, so drones jump.
   What rescues this today is the configured *values*, not the code: a step is
   one cell while the threshold is >= 4.24 cells at `resolution: 0.1`, so a
   drone can never get close enough to tunnel past another. Nothing enforces
   `min_separation_cells > step`. Assert it alongside guard 1 rather than
   building swept-volume collision checking.

**Where the guards live — and why not in `config`.** The natural home for the
physical constants is `simulation`, the module that builds the body: promote the
hardcoded `0.15` in `_build_drone_xml` to a module constant and derive the
diagonal from it. `coordination` may read that (the dependency direction allows
`coordination -> simulation`), so the guards go in `CentralizedMaster.__init__`.
They must **not** go in `config` validation, which would mean
`config -> simulation` and CLAUDE.md is explicit that `config` depends on
nothing. So: `config` carries the user-facing values and checks only
schema-level sanity (present, positive); the physical floors are enforced where
the geometry is visible. This also settles the "where does the half-extent live"
question left open above — one constant in `simulation`, read by consumers,
never duplicated into YAML.

### Addendum (2026-08-20, Feature 4b implemented) — measured, and one vacuous test caught

Implementing the filter let the three predicted consequences be measured rather
than reasoned about. Three drones, 20-tick mission in the 6x6 test room, filter
disabled vs enabled:

```
filter OFF   phantom occupied cells = 0    poisoned height cells = 47
filter ON    phantom occupied cells = 0    poisoned height cells =  0
```

The occupancy prediction was right in a way that matters for testing: a false
occupied reading needs only 2-3 later free observations to wash out, and over a
full mission every drone cell gets them — so **occupancy self-heals to zero even
with the bug present**. A mission-level "no phantom obstacles" assertion is
therefore *vacuous*: it passes with the filter off. It was written, measured,
and deleted.

The height layer is the discriminating signal, exactly as predicted, because
`max(height, hit_z)` is monotonic. `test_master.py` asserts on height;
single-scan occupancy — where the artifact is visible before it washes out — is
asserted in the integration tests instead.

Worth remembering as a testing lesson, not just a mapping one: a self-healing
channel cannot carry a regression test for the thing it heals from.

