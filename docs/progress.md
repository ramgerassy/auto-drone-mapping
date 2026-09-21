# Project progress & decision log

A running journal of non-obvious decisions, trade-offs, and thoughts worth
remembering — the *why* behind choices that the code and git history don't
capture on their own. Entries run oldest to newest. Each entry records what was
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
`dx = dy = 0.2125`: Euclidean distance is 0.3005 (passes a 0.30 m threshold)
while Chebyshev is 0.2125, so the boxes overlap by 0.0875 m. (Corrected
2026-09-19: this originally read `0.212`, whose Euclidean distance is 0.29981
and therefore *fails* a 0.30 m threshold — the break-even is
`0.30/sqrt(2) = 0.21213`. The conclusion was unaffected; the illustration was
wrong.) A diagonal approach slips
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

### Addendum (2026-09-06, PR #11 review) — the accepted cost was mis-stated

A multi-agent review of PR #11 found the "accepted cost" recorded above to be
wrong, and the correction is worth keeping because the mistake was a reasoning
error, not a typo.

**What I claimed:** a real wall inside the exclusion radius "stays unknown for
the tick and is mapped on a later pass".

**What actually happened:** the filtered ray was emitted as a MISS with
`max_range = hit.distance`, and `Mapper._integrate_observation` marks a MISS
free through its **endpoint** — `bresenham_2d` is endpoint-inclusive and the
MISS branch has no `cells[:-1]` exclusion, unlike the HIT branch. So the
discarded cell was not left unknown; it was claimed **free**. Verified: a wall
cell replayed under a filtered ray reaches p = 0.3077 on the second observation
and 0.0376 by the eighth.

**Fix: stop the free trace at the ray's entry into the exclusion sphere**, not
at the hit. The entry point is the last position on the ray that is provably
unobstructed, so it is the furthest we may honestly claim. Head-on against a
teammate 2.0 m away, the trace now stops at 1.70 m rather than 1.85 m.

**But the map-level consequence was overstated, including by me.** The review
argued this erodes real walls into free space, letting A\* plan through them —
a Tier-1 collision violation. I could not reproduce that. Two geometries were
tried: a teammate parked 0.2 m from a wall (2 of 360 rays are filtered wall
hits), and a small pillar sitting inside the exclusion sphere. In both, the
cell's classification was dominated by *other* rays — legitimate occupied hits
in the wall case, discretization free-traces from adjacent rays in the pillar
case — and came out the same with and without the fix. A regression test
written for it passed under both behaviours and was deleted, for the same
reason the phantom-occupancy test was.

So: the fix is correct because claiming space you did not observe is wrong and
the correct version costs nothing. It is **not** known to fix an observable
map defect. The discriminating assertions are at the observation level.

**Two blind spots the review's mutation testing found**, both of which passed
the full suite before it:

- Dropping the `other != drone_id` guard — a drone would filter *itself*,
  blinding it within 0.30 m of its own centre, exactly the near-wall geometry
  where it most needs to see.
- Widening the radius 0.30 -> 0.50. The suite pinned the radius from below
  (corner-on hits) but not from above, so it would silently discard every wall
  hit near any drone.

Both now have tests, as do "every teammate is checked, not just the first" and
the 3D-vs-2D distance rationale, which was documented but untested.

`exclusion_radius` is now validated in `__init__`: negatives are rejected
(squaring silently turned -0.30 into +0.30), and so is any positive value below
the 0.212 m body corner radius, since that re-admits the bug the filter exists
to fix. Exactly 0.0 remains the deliberate, greppable opt-out.

**One arithmetic correction:** a test docstring claimed a single free
observation lands at p = 0.4001, just above `free_threshold`. That came from the
rounded `# ~-0.405` comment. The exact value is `log(0.4/0.6)`, which *is* the
log-odds of p = 0.4 — so one observation lands precisely **on** the threshold
and only the strict `<` keeps the cell unclassified. There is no margin.

---

## 2026-09-19 — Planner clearance implemented (Sprint 2, Feature 4c)

The 2026-08-19 entry and its addendum specified this; what follows is what
implementing it actually taught.

**Shape as planned.** `AStarPlanner` gains a required `clearance_radius` in
metres and inflates known-occupied cells by
`r = ceil(clearance_radius/resolution + 0.5) - 1` cells of Chebyshev dilation,
computed once per `plan()`. Unknown cells are never inflated. The start cell is
exempt from the clearance test — and only that test — so a drone that discovers
a wall beside itself can still plan its way out instead of silently ending the
mission. `CentralizedMaster` gained the three guards: the body-diagonal floor
on `min_separation`, pairwise start-position spacing, and a one-cell minimum.

**D1 resolved: `clearance_radius` is a required keyword argument.** The finding
this feature came from was that a body-size assumption went unstated; a
required argument is the one form that cannot be left unstated. Seven call
sites now say `clearance_radius=0.0` explicitly where they want a point robot.

### The discretization tax on doorways — new, and it constrains Feature 5

`2r + 1` cells of gap is **not** sufficient. A wall face that lands on a cell
boundary has its ray hit point attributed to the cell on the far side, so the
mapped obstacle is up to one cell wider than the wall. Measured on a 6x6 test
room at `resolution` 0.25 with `r = 1`:

```
0.75 m gap (3 cells = 2r+1 exactly)  -> mapped as 2 free cells -> impassable
1.25 m gap (5 cells)                 -> traversable; 558/576 cells mapped
```

So the rule for Feature 5's MJCF is **`2r + 1` cells plus a cell of slop**, and
gaps should be cut on whole-cell boundaries where possible. A doorway sized to
the theoretical minimum will map as one cell narrower than it is and refuse the
drone. This is worth knowing before the 50x50 scene is drawn, not after.

### A body-overlap assertion was written and discarded

The plan's headline test was "no drone's body ever overlaps a known-occupied
cell, every tick". It cannot be satisfied and does not mean what it sounds like:

- At `resolution` 0.25 a 0.30 m body **always** overhangs its own cell by
  0.025 m, wherever it sits. Any drone adjacent to an occupied cell trips the
  check by construction.
- An occupied *cell* is up to half a cell larger than the wall inside it,
  because a ray hit point on a cell boundary is attributed to one side of it.
  The doorway measurement above is the evidence: the divider's face at
  y = -0.25 sits exactly on the row 10/11 boundary and row 11 maps as occupied,
  which is what turns a 3-cell gap into 2 free cells.

So the assertion measures grid discretization, not physical overlap — the
±0.025 m it reported is exactly the overhang. It was also vacuous in the
original test room, which is too open for a drone to ever approach a wall
(measured worst gap +0.225 m with clearance both on and off).

Replaced by the doorway pair above, which is discriminating: the narrow-door
test fails with clearance disabled, and the wide-door test fails if unknown
cells are inflated or the radius is one cell too large. Both mutations verified.

This is the second time in two features that the obvious mission-level
assertion turned out not to discriminate (see the 4b addendum). The pattern is
worth naming: **a mission-level metric aggregates over so many rays, ticks and
cells that a local defect usually washes out of it.** The assertion has to be
placed where the defect is local — at the planner, at the observation — and the
mission-level test is for *liveness*, not correctness.

### Addendum (2026-09-19, PR #12 review) — clearance held at plan time only

A four-agent review of PR #12 found twenty issues; all were reproduced before
being acted on. Three changed the design rather than the code around it.

**1. The feature's guarantee was plan-time only.** `is_assignment_valid` tested
`prob < free_threshold` and nothing else, so a committed path survived the
discovery of a wall beside it:

```
planned while row 0 was unknown: [(1,1) ... (7,1)]
after discovery, re-planning returns: None
is_assignment_valid said:            KEEP
```

The drone flew a route the planner called illegal — the exact pre-4c bug. And
this is the **normal** exploration case: A\* routes only through known-free
cells, so a wall discovered later was unknown at plan time and is never itself
on the path; only its inflation zone touches path cells, which were free and
stay free.

The root cause is a Feature 4 decision that 4c invalidated without noticing.
`assignment.py` justified keeping paths rather than re-planning because "a path
stays valid unless a cell it crosses stops being free, which is exactly the
check below". True before 4c; false after it, because 4c added a second way for
a path to become invalid. **`PathPlanner` now exposes `clearance_mask`, and
`assign_all` re-checks committed paths against it** — one definition of a legal
cell serving both plan time and keep-alive time, rather than `coordination`
re-implementing the body model and drifting.

**2. The start-cell exemption only worked at `r = 1`.** Exempting the start is
useless when every *neighbour* is inflated too, which is the case at `r >= 2` —
and `resolution: 0.1` with a 0.20 m clearance gives exactly `r = 2`. The
liveness property was untested at the radius the project will deploy, and did
not hold there. Replaced by an escape *phase*: while the search is inside the
zone it may move through it; once it reaches open ground it may not re-enter.
Escaping is allowed, loitering is not.

**3. `is_complete` could not tell "explored" from "walled out".** Both end with
every drone unassigned. The distinguishing signal was already being computed in
`_assign` and thrown away — the frontier count. Added `is_blocked` and
`unreachable_frontiers`:

```
wide doorway (5 cells)   complete=True blocked=False unreachable=0 known=96.9%
narrow doorway (3 cells) complete=True blocked=True  unreachable=2 known=58.0%
```

Deliberately **not** added to the `Coordinator` Protocol — that is a named seam,
and Feature 6 should decide what the CLI needs before it is widened.

### Guards that were decoration

`NaN` defeated all four of this PR's guards at once, because every one is a `<`
comparison and every comparison against NaN is `False`. Worse, `min_separation`
= NaN makes `threshold_sq` NaN in `resolve_moves`, so every `d2 < NaN` is False
and **no move is ever blocked** — collision avoidance silently off while
`test_drones_never_come_closer_than_min_separation` kept passing. All four now
check `math.isfinite` first.

The start-separation guard also measured the wrong space: it compared metric
poses, but the invariant lives in cells and `world_to_grid` floors. A pair
0.43 m apart cleared a 0.4243 m threshold and snapped to 0.25 m — two 0.30 m
bodies overlapping before tick 1. Now checked in cell space.

Two more silent-parameter cases, both the same shape as the `min_separation < 1
cell` guard that was already there: an oversized `clearance_radius` blocked
every cell and reported a completed mission on tick 1 (now rejected against the
grid extent), and `clearance_radius <= resolution/2` was silently identical to
0.0 — which undoes D1, since declaring `0.1` greppably states a body and buys
point-robot planning.

### The mutation lesson

Eleven of twenty-two mutations survived the suite that shipped. The pattern
across all of them: **every test sat strictly inside the region it was testing,
never on its boundary.** All three separation guards were `<`, every guard test
used a value well inside the rejection region and every acceptance test a value
well outside it, so flipping any of them to `<=` passed. The inflation formula
was exercised at ratios of 0.0, 0.8 and 1.0 — none near the half-cell step — so
shifting it by half a cell passed. `_dilate`'s zero-padding was argued for in a
comment and never tested, because every fixture has walls on all four borders.

Boundaries are where the bugs are, and behavioural tests reach them only by
accident. The fix was a parametrized test on the arithmetic itself plus an
exact-boundary acceptance case per guard, importing `MIN_SEPARATION_FLOOR`
rather than restating it as `0.4243` — the rounded literal is what had hidden
the boundary.

### One arithmetic correction, inherited from 4b

The worked example `dx = dy = 0.212` was wrong in four files: its Euclidean
distance is 0.29981, which *fails* a 0.30 m threshold rather than passing it.
Break-even is `0.30/sqrt(2) = 0.21213`; the illustration now uses 0.2125. The
`MIN_SEPARATION_FLOOR` derivation was never affected — only the number chosen
to illustrate it — but it had been copied into `progress.md`, the feature doc
and a test docstring, which is what a triplicated derivation always does.

### Addendum — the two planner references

The fix above left `CentralizedMaster` holding a planner that is also inside
the strategy, with nothing tying them together. Documented at first; that was
the weak option, because the failure is silent *and* asymmetric. A master whose
planner has `clearance_radius=0.0` gets an all-False clearance mask, never drops
a committed path, and restores the bug the re-check exists to prevent — with
every test still green, since they all build both references from one variable.

**Not fixed by widening `FrontierStrategy`.** The reason is not that seams are
sacred: it is that a strategy scoring by expected information gain against
straight-line distance has no planner, and the Protocol should not demand one.

**Not fixed by making the mismatch impossible either**, though that option
exists: `FrontierAssignment` could carry the planner that produced it, and the
master could drop the argument entirely. Rejected for now because it changes the
return type of a named seam, needs an Optional for planner-less strategies, and
churns every `FrontierAssignment(...)` in the tests — to solve an aliasing
problem that Feature 6 largely dissolves by constructing both from one config
value.

**Chosen: a duck-typed identity guard** in `CentralizedMaster.__init__`, reading
an optional `planner` property. Costs one `getattr`, raises on the realistic
mistake, and leaves strategies that plan nothing unaffected. Revisit when
Feature 6 wires the config: at that point the guard becomes belt-and-braces and
the assignment-carries-planner shape is the tidier end state.

Worth naming the underlying smell rather than just the fix: what the master
needs is not "the planner", it is "may the body occupy this cell" — a question
about the **body**, not about the search algorithm. `clearance_mask` sitting on
`PathPlanner` is that question answered by the wrong object. If a third consumer
ever needs it, extracting a body/clearance model both depend on is the move.

---

## 2026-09-19 — Large indoor scenario (Sprint 2, Feature 5)

**Resolution 0.2, not `small_indoor`'s 0.1.** Measured cost of one `plan()`
call over a 50x50 m grid:

```
res=0.1   500x500 (250,000 cells)  r=2  mask= 6.0ms  plan= 6.9ms
res=0.2   250x250  (62,500 cells)  r=1  mask= 0.3ms  plan= 3.1ms
```

The clearance mask is 20x more expensive at 0.1, because r = 2 means two
dilations over a 500x500 array, and `plan()` runs once per candidate frontier
per re-selecting drone. That compounds past the CI budget CLAUDE.md sets.

Accepted cost: coarser cells classify wall-adjacent space less precisely, which
bears on the >=98% per-cell accuracy KPI. Not self-defeating, since the
reference map is produced at the same resolution — but if Feature 6's
acceptance run shows the margin is thin, this is the lever.

**Which is why every wall face sits on a 0.2 m lattice point — also a 0.1 m
lattice point.** Dropping the resolution needs no geometry change. The
alignment is not cosmetic either way: a face at a half-cell offset is exactly
what turned a 3-cell gap into 2 free cells in the 4c measurement.

**The geometry was generated from exact rational arithmetic, not typed.** 24
wall segments with door gaps is enough arithmetic that a single transposed
digit would produce a doorway one cell narrow — invisible in the MJCF, and
surfacing much later as a room the swarm never enters. The generator asserted
lattice alignment on every face before emitting; the committed tests re-assert
it against the *parsed model*, since a test that greps XML proves nothing about
what MuJoCo loaded.

**The test worth having is reachability, not dimensions.** Eight room probes
planned from the spawn with `clearance_radius=0.20`; all eight reachable.
Verified non-vacuous by filling four cells of one jamb, taking that doorway
from 6 cells to 2: both rooms behind it become unreachable while the rest of
the plan is untouched. Measured widths: all eight doorways exactly 6 cells
(1.2 m) against a 4-cell floor.

The scene also runs end to end through the Sprint-1 CLI — 9.4% of cells known
from a corridor-confined patrol, which is the expected figure when the rooms
are never entered, and a useful baseline for what frontier exploration should
beat in Feature 6.

---

## 2026-09-20 — Finding 2 resolved: why large_indoor never terminated (Feature 6)

The handoff's leading hypothesis was that the remaining unknown space is wall
interior, that the free cells facing it are therefore *permanent* frontiers,
and that at `r = 1` enough of them stay reachable for the swarm to chase them
forever. **Measured, and it is not that.** At the plateau every surviving
frontier is unreachable from every drone:

```
(213, 36) clearance_blocked=True  reachable_from=[False, False, False]
(90, 135) clearance_blocked=True  reachable_from=[False, False, False]
(136,135) clearance_blocked=True  reachable_from=[False, False, False]
(159,189) clearance_blocked=True  reachable_from=[False, False, False]
(35, 223) clearance_blocked=True  reachable_from=[False, False, False]
```

Two separate mechanisms, both confirmed:

**1. The frontier set never settles.** Wall-surface cells collect *both* free
and occupied evidence at grazing incidence and drift across the 0.4/0.6
classification bands, so tiny regions keep appearing and vanishing — 5 regions
at one sample, 10 twenty ticks later. Traced per cell:

```
(136,135) 0.1041..0.5960  FFFFFF??????   free -> unknown
(159,188) 0.0067..0.9334  OO?FFFFFFFFF   occupied -> unknown -> free
```

**2. Assignments were never re-checked against the target.**
`is_assignment_valid` tested arrival, the wait counter and the next cell, but
never whether the goal was *still a frontier*. Drones were flying 60-100 step
journeys with `still_a_frontier=False` on nearly every sample — arriving at
nothing, re-selecting, repeating. With someone always mid-path,
`all(assignment is None)` is never true, so `is_complete` could never fire.

This is the third time the same defect shape has appeared in this function: the
keep-alive test not re-checking what justified the assignment. 4c added the
clearance re-check; this adds the target re-check.

### What did not work

**`min_frontier_size` (filtering noise regions at source) trades coverage for
speed and cannot buy the KPI.** Measured on large_indoor:

```
size=2   97.6% coverage    (KPI met, slow)
size=4   79.1% coverage
size=12  87.2% coverage    (fast, plateaus)
```

Filtering removes real frontiers along with noise. The knob is kept — plumbed
through `Mapper` and `PlanningSettings`, defaulting to 2, i.e. off — because it
documents a real phenomenon and is the right lever if a future scene is noisier.
It is not the fix.

### What worked

**Admissible lower-bound pruning in `NearestFrontier.select`.** Candidates are
planned in increasing `cost_lower_bound` order and the loop stops once the bound
exceeds the best score found; no real path can undercut the bound and the
spreading penalty only adds, so the selected assignment is *identical*. Pure
cost, no behaviour change — and it is what made the target re-check affordable,
since dropping stale assignments makes re-selection common. **2500 ticks went
from 234s-for-400 to 61s**, roughly 25x.

The bound is declared by the planner (`PathPlanner.cost_lower_bound`) rather
than computed in the strategy. A first attempt computed octile distance inline
and broke a test whose `StubPlanner` returns costs that violate grid geometry —
correctly, because the strategy would have been assuming every planner uses
8-connected octile costs. Letting each planner state its own bound removes the
assumption; a stub returns 0, which disables pruning and is always legal.

**A no-progress stop.** Even with stale targets dropped, transient reachable
frontiers keep appearing, so "every drone unassigned" never holds on a large
scene. `no_progress_ticks` ends the mission when no tick has newly *classified*
a cell for N ticks, reporting `blocked=True`. It counts classified cells rather
than visited ones because the mission's product is the map: a tick that resolves
nothing achieved nothing, whatever the drones did.

This is a heuristic and is labelled one. The principled alternative — teaching
frontier detection not to emit permanently-unobservable space — needs a
definition of "unobservable" that does not require ground-truth geometry, which
is a research question, not a sprint task.

### Result

```
large_indoor  drones=3  ticks=2264  cov=97.606%  complete=True   58s
small_indoor  drones=3  ticks= 245  cov=98.440%  complete=True    8s
small_indoor  drones=1  ticks= 377  cov=98.210%  complete=True    8s
```

Coverage KPI (>=95%) met on both scenarios. Scaling KPI 377/245 = **1.54x**,
just over the 1.5x commitment — a thin margin worth watching rather than
celebrating. The large_indoor acceptance run dropped from 279s to 58s, which is
what keeps it viable as a PR-gate test.

### One acceptance assertion was wrong and got corrected

"Every room physically entered" fails while every room is **100% mapped**: the
rangefinder reaches 12 m and a room is ~11 m across, so a drone in a doorway
resolves the whole room without going in. The assertion was measuring sensor
range, not exploration. Replaced with per-room mapped fraction (the actual
requirement, and per-room so one dark room cannot hide behind a 97% global
figure) plus a weaker traversal check that some doorways are genuinely flown.

### Addendum — drones were flying inside the walls

Found by watching `--view`, not by a test: drones passed very close to walls,
got stuck, and *vanished* from an overhead view. A drone cannot disappear from
above unless it is inside a 3 m wall, occluded by it.

Measured on large_indoor before the fix:

```
drone 0: inside the inflated zone 1252/2264 ticks (55.3%), longest streak 1183
drone 1:                           359/2264 ticks (15.9%), longest streak  249
drone 2:                           151/2264 ticks ( 6.7%), longest streak   61
```

At `resolution` 0.2 a cell at Chebyshev distance 1 from an obstacle puts the
wall face 0.1 m from the drone centre against a 0.15 m half-extent, so this is
genuine overlap with wall geometry, for over half the mission.

**Cause: the 4c escape phase was unbounded.** The rule was "from a blocked cell
you may move to any free cell; from an open cell only to open cells", described
as *escaping is allowed, loitering is not*. But a path that **starts** in the
zone and **never leaves** satisfies it completely — the planner was free to
route along the inside of a wall for the entire journey.

**The test could not catch it.** It asserted that blocked cells form a *prefix*
of the path, which a path that never leaves the zone satisfies trivially. A
prefix property says nothing about length.

**Fix: bound the allowance to the start's neighbourhood** — a blocked cell is
passable only within Chebyshev `r` of the start. Escaping means stepping off
the spot you are standing on, not licence to travel by wall. The test now
asserts the bound itself, at r = 1, 2 and 3.

```
drone 0: 1252 -> 0 ticks in the zone
drone 1:  359 -> 2   (genuine escapes from a newly discovered wall)
drone 2:  151 -> 0
mission: 2264 -> 1498 ticks, coverage unchanged at 97.6%
```

Exploration got **faster**, because a third of the mission had been spent
flying through walls rather than mapping.

Two lessons worth keeping. First, **a prefix assertion is not a bound** — it
constrains shape, not magnitude, and magnitude was the whole property. Second,
this was invisible to 300 tests and obvious within a minute of watching the
thing run. Determinism, coverage and separation were all green throughout;
none of them ask where the drone *is*.

Also corrected here: two assertions treating `is_blocked` as pass/fail. It
fires on successful runs too — clearance leaves wall-adjacent frontiers
permanently visible but unoccupiable, so a fully explored room still ends with
frontiers outstanding. What separates finished from walled-out is magnitude,
which the handoff already recorded and these assertions had ignored.

### Addendum — exhausted frontiers and return-to-base

Two more things the viewer showed that no test asked about.

**A drone looped between the same two rooms.** Wall-surface cells emit
frontiers over space no scan can resolve, so a drone flies to one, fails to
clear it, picks the other, fails, and comes back. The coordinator now remembers
frontiers a drone **reached** without clearing and stops offering them.
"Reached" is the right evidence rather than "targeted": arriving is what proves
a frontier cannot be resolved from close range, and it is the only proof
available without ground-truth geometry.

**An idle drone loitered wherever it stopped.** It is then an obstacle its
teammates route around and a body the separation rule must keep clear of.
After `return_to_base_ticks` unassigned ticks a drone now flies back to its
start position, which construction already validated as clear of geometry. Any
drone that picks up a frontier abandons the trip home immediately, so returning
never competes with exploring.

Expressed in **ticks, not seconds**, despite the request being "60 seconds":
`mj_step` is never called, so simulated time does not advance, and wall-clock
time would make a deterministic run irreproducible. Ticks are the only clock
this system has.

### Combined effect

```
                        before        after
large_indoor 3 drones   2264 ticks    1498 ticks   97.6% coverage
small_indoor 3 drones    245 ticks     175 ticks   98.4%
scaling KPI             1.54x         4.70x        (221 vs 47 ticks to 95%)
```

The scaling KPI went from a hair over its 1.5x commitment to comfortably past
it. That is not a measurement change — it is the wall-hugging fix. A third of
each mission had been spent flying inside walls, and that waste fell hardest on
the multi-drone arm, where three drones were each doing it.

Worth recording as a process note: the wall-hugging, the room-looping and the
loitering were all found by **watching the simulation**, with 300 tests green.
Determinism, coverage, separation and accuracy were all passing throughout.
None of them ask where the drone actually *is*.

### Addendum — where the viewer time actually goes

Reported as "really slow" while watching. Two candidates were measured rather
than assumed, and the obvious one was wrong.

**`probability()` caching: no measurable gain, reverted.** The hypothesis was
that recomputing `exp` over 62,500 cells ~25 times a tick (once per `plan()`,
per clearance mask, per frontier pass, per progress check) dominated. A cache
invalidated by `update_free`/`update_occupied` was implemented and measured:
**46s against 44s — noise.** It was reverted rather than kept: CLAUDE.md names
caches specifically as something to stop before adding, and an optimization
that cannot be measured has not earned its complexity. Ray-casting and map
integration are the real per-tick cost.

**The viewer's own pause was a third of the wall time.** `_FRAME_DELAY_S = 0.02`
per rendered tick is 30 seconds of pure sleep across a 1498-tick mission, on top
of ~46s of compute. On a large scene the tick already takes ~30 ms, which paces
the animation at a watchable ~30 fps by itself.

Default is now 0, with `--view-delay SECONDS` for small scenes that would
otherwise finish before you can look at them. **This changes public CLI
behaviour** (flagged per CLAUDE.md): a `--view` run of `large_indoor` is now
about 40% shorter in wall time, and identical in output — the delay never
touched the map.


### Addendum — the visit heatmap, and what it shows

Added on the user's suggestion after watching a run: `--visit-heatmaps` writes
one PNG per drone counting how many ticks it spent in each cell, drawn over the
occupancy map. Counts are on a **log** ramp because they are heavy-tailed — a
drone parked at base accrues dozens of visits to one cell while a corridor it
swept once has a single visit, and a linear ramp renders everything but the
parking spot identical.

Counted in the CLI, not the coordinator: it is a diagnostic, and `coordination`
should not carry state only a debug flag reads.

**It immediately showed three things coverage had hidden.** On large_indoor,
3 drones, 1498 ticks, 97.6% coverage:

```
Drone 0: 1103 cells visited, 221 revisited, worst cell  61 times
Drone 1: 1204 cells visited, 198 revisited, worst cell   6 times
Drone 2:  770 cells visited, 264 revisited, worst cell  16 times
```

1. **Work is divided badly.** Drone 0's map covers the western half and nothing
   east; drone 2 never leaves the corridor junction and the one room beside it.
   The spreading penalty separates *targets*, but nothing gives a drone a
   region to own, so two drones can spend a mission in the same quadrant.
2. **Drone 2 retreads 34% of the cells it visits** (264 of 770), concentrated
   at the corridor crossing — it shuttles in and out rather than progressing.
3. **Long diagonal traverses across already-explored rooms.** Drone 0's trace
   is a zigzag spanning the whole west half repeatedly. `NearestFrontier` picks
   the cheapest *reachable* frontier, and once a neighbourhood is cleared the
   cheapest remaining one is often across the map — so the drone transits
   explored space instead of sweeping.

All three are the same underlying gap the user identified from the viewer: the
strategy plans to *reach* a frontier cell when a 12 m sensor only needs a
vantage point from which the frontier becomes *observable*. Frontier cells are
adjacent to unknown space, which is usually against a wall, so "go to the
frontier" means "cross the map to a wall" by construction.

Next-best-view selection — score candidate *viewpoints* by expected information
gain within sensor range, rather than scoring the frontier cells themselves —
is the fix, and it is a `FrontierStrategy` change large enough to want its own
plan rather than a late-sprint improvisation.

---

## 2026-09-20 — The map is 2D, not 2.5D

Raised by the user reading `map.png`: an obstacle in one room was missing, and
they inferred that if the drone only ever sees walls tall enough to cross its
flight plane, "this is not 2.5D, it's 2D as long as the room is smaller than
12 m". That inference is correct, and the measurement is worse than the
inference.

**The height layer holds exactly one value.**

```
height layer: 3961 cells written, 58539 still -inf
  distinct height values: [1.]
```

1.0 m is the flight altitude. `Rangefinder._compute_directions` builds every ray
with body-frame `z = 0` — a purely horizontal sweep — so every `hit_point[2]`
equals the drone's own altitude, and `update_occupied(col, row, hit_z)` stamps
that same number into every occupied cell it ever writes. The per-cell height
channel cannot record anything but the plane the rays were cast in.

**And obstacles below the flight plane are invisible.** Measured against
`large_indoor`'s four crates, with the drone at 1.0 m:

```
geom          z span        ray at z=1.0 hits it?
crate_ne   0.00 .. 0.80     False   <- the "missing object"
crate_sw   0.00 .. 0.80     False
crate_nw   0.00 .. 1.20     True
crate_se   0.00 .. 1.00     True
```

The two crates the user could not find are exactly the two shorter than the
flight altitude. The two taller ones map correctly, appearing as unknown
interiors ringed by occupied cells.

**This contradicts a stated non-negotiable.** CLAUDE.md commits to "2.5D
mapping only. 2D occupancy grid + per-cell height" and names it a constraint
that shapes every decision. What ships is a 2D occupancy grid plus a constant.

The gap is not in `mapping` — `OccupancyGrid.update_occupied` takes and stores a
height faithfully, and `save_png` shades by it. It is in `perception`: a single
horizontal ray plane cannot produce varying `hit_z`. Genuine 2.5D needs rays
spread over elevation as well as azimuth, so `hit_point[2]` varies with what was
struck. That is a `Rangefinder` change — the `Sensor` seam's shape already
allows it, since `scan()` returns `RayObservation`s carrying full 3D hit points
and the mapper already reads `hit_point[2]`.

Two consequences worth stating plainly:

1. **The height channel has never been exercised.** Every test that asserts on
   it asserts on `-inf` (absent) or on the flight altitude, so it has been
   green while measuring nothing. The Feature 4b teammate-filter tests are the
   clearest case: "the height cell stays `-inf`" was a real regression test for
   a real bug, but it could never have caught a *wrong* height, only a present
   one.
2. **Obstacle detection is altitude-dependent in a way nothing documents.** A
   scenario author placing a 0.8 m crate has placed a decoration, not an
   obstacle, and nothing in the config or the scene tells them so.

Also settled in the same exchange: the `map.png` the user was reading came from
the `--assignment global` run (87.4% coverage), not the 97.6% baseline — the
under-explored wedges they noticed are the global-allocation failure already
recorded above, not a mapping fault. The 97.6% map is near-uniformly free with
both visible crates present.

---

## 2026-09-20 — Elevation sweep: the map is 2.5D now (Sprint 3, Feature 8)

Before: the height channel held **one** distinct value, 1.0, the flight
altitude. After, on `small_indoor`:

```
occupied cells with a height: 1735
distinct heights: 163   range 0.30 .. 3.00 m
```

And on `large_indoor`, every crate is found with its true top:

```
crate       true top   mapped?   recorded height
crate_ne       0.8 m      yes           0.80 m   <- previously invisible
crate_nw       1.2 m      yes           1.20 m
crate_se       1.0 m      yes           1.00 m
crate_sw       0.8 m      yes           0.80 m   <- previously invisible
```

**Two changes, solving different halves.**

*Fly low.* Altitude 1.0 -> 0.3 m. A horizontal ray detects everything taller
than the altitude **at any range**, which an angled fan cannot: from 1.0 m a
-10° ray only reaches down to 0.47 m at 3 m, and less further out.
Height-independent detection is the stronger guarantee, and it is what the low
plane buys. It also makes the user's original two-phase proposal unnecessary.

*Fan upward.* Elevation bands 0° to 20°, so `hit_point[2]` varies with what was
struck. **Upward only**: a downward ray strikes the floor and the mapper would
record a ring of phantom walls around every drone. Flying low is what makes an
upward-only fan sufficient, so the two halves depend on each other.

### The rule that stops it erasing what it finds

The mapper projects every ray to 2D and marks all cells before the endpoint
free. An upward ray passing *over* a 0.8 m crate and striking a wall ten metres
beyond would mark the crate's own cell free — and at one occupied update
(+0.847) against four free ones (-1.62) per scan, the crate loses. The feature
would have deleted exactly the obstacles it was added to find.

So: **only navigation-plane rays write free space.** `RayObservation` carries
`navigation_plane`; elevated rays contribute occupancy and height and nothing
else. This is not a workaround — an elevated ray genuinely carries no
information about the ground beneath it, and claiming otherwise was always
wrong. Verified discriminating: removing the rule fails the test.

### What it cost, and what it changed

Rays per scan go from 72 to 360 (azimuth x bands), and ray-casting was already
the dominant per-tick cost. Against that, `large_indoor` finished in **1074
ticks against 1498**, because the drones now see more per scan. Coverage moved
97.6% -> 95.7%, still over the KPI: the crates are real obstacles now and
occupy cells that used to be flown over.

**Every previous benchmark baseline is superseded.** The scenarios pose a
different problem now — four obstacles that must be routed around rather than
ignored.

### The narrowing worth naming

With the drone unable to climb over anything, height stops being a navigation
input and becomes map *output* — a property a consumer reads, not something the
planner consults. This is 2.5D-for-mapping, not 2.5D-for-planning. Worth
stating because CLAUDE.md's "2.5D mapping only" does not distinguish them, and
the difference decides whether a height channel is load-bearing or descriptive.

### The tests were green while measuring nothing

Every prior assertion on the height layer checked `-inf` (absent) or the flight
altitude. Feature 4b's "the teammate's height cell stays `-inf`" was a real
regression test for a real bug, but it could only ever catch a height that was
*present*, never one that was *wrong*. A channel with one possible value cannot
fail an equality check. The new tests are the first that could.

---

## 2026-09-20 — The num_rays anomaly was two bugs and a red herring

"More rays gives worse coverage" turned out not to be a statement about rays.
Swept across a wide range on `large_indoor`, it is not monotonic at all:

```
 rays    cov   stopped        gap between rays at 12 m
   12   1.33%  stalled        31.4 cells
   24  97.36%  stalled        15.7
   36  97.34%  stalled        10.5
   72  95.71%  stalled         5.2
  144  70.74%  stalled         2.6
  288  97.35%  stalled         1.3
```

24 rays beats 144. Six times the rays, a third of the map. Whatever this is, it
is not ray density — and every run stopped "stalled", which was the clue.

### Bug 1 — `no_progress_ticks: 200` silently truncates missions

```
 rays  no_progress  ticks     cov   stopped
   72          200   1074  95.71%  stalled
   72          800   1674  95.71%  stalled
   72            0   4000  95.71%      cap     <- more time changes nothing
  144          200   1197  70.74%  stalled
  144          800   4000  97.38%      cap     <- +26.6 points, same code
  144            0   4000  97.38%      cap
```

At 144 rays the stop ended the mission with a third of the map unfound, and the
run **reported itself finished**. Raised to 800 across all scenarios.

This is a heuristic I added to make `large_indoor` terminate, and it is worth
being clear about what it is: a stand-in for "the swarm has nothing left to do"
that cannot distinguish that from "the swarm is between discoveries". `max_ticks`
is the real backstop; this only exists to avoid burning it.

### Bug 2 — nothing. 72 rays really does plateau at 95.71%

Unchanged at 200, 800, or with the check disabled for 4000 ticks. That one is a
genuine property of the run, not a truncation.

### The red herring

Once truncation is excluded, coverage across 24-288 rays sits at 95.7-97.4%.
That spread is **which trajectory a configuration happens to take**, not how
densely it scans. Changing ray count changes what is seen first, which changes
every frontier decision after it. Two runs of the same code over the same map
diverge because the sensing perturbed the sequence, not because one sensed
better.

12 rays is the one real geometric failure: 31-cell gaps at range, 1.33%
coverage, correctly hopeless.

### And it overturned the allocation verdict — again

The allocation benchmark was run before the elevation sweep. Re-run on current
code:

```
variant    ticks     cov  revis   bal  union
baseline    1074  95.71%    295   96%   2343   <- now the WORST coverage
B           1049  97.36%    229   92%   2573
A: global   1177  97.37%    423   93%   2682
A+B         1112  97.36%    253   92%   2704
```

**A no longer fails the coverage KPI.** It was rejected on 87.4%; it now reaches
97.37%. Identical at `no_progress_ticks` 200 and 800, so this is not the
truncation — it is the elevation sweep. Better sensing changed which allocation
is better, which in hindsight is unsurprising: allocation decides where drones
go, and what they can see decides what is worth going to.

All three variants now beat the baseline on coverage. The baseline's one
remaining win is workload balance (96%), and it explores the least ground of
the four (2343 cells against 2704).

**The lesson is about method, not allocation.** Every benchmark conclusion in
this project has been conditional on the sensing model, and the sensing model
had a bug that made a third of the obstacles invisible. Comparing coordination
strategies on top of that measured the wrong system. Conclusions drawn before
2026-09-20 should be re-derived, not cited.

### The allocation benchmark, re-measured on the fixed sensor

```
scenario       variant    ticks     cov  revis   bal  union
comb_indoor    baseline     389   19.0%    135   81%    992
comb_indoor    B            372   19.1%    244   85%    831
comb_indoor    A            389   19.0%    135   81%    992   <- identical to baseline
comb_indoor    A+B          440   19.1%    320   84%    887
large_indoor   baseline    1674   95.7%    295   96%   2343   <- worst coverage
large_indoor   B           1049   97.4%    229   92%   2573
large_indoor   A           1177   97.4%    423   92%   2682
large_indoor   A+B         1112   97.4%    253   92%   2704
small_indoor   baseline     316   98.4%     77   58%    651
small_indoor   B            223   97.2%      9   36%    389   <- 40% idle
small_indoor   A            193   98.2%     19   94%    556
small_indoor   A+B          197   98.3%     18   89%    554
```

**A+B adopted**, and set in all three scenario configs. It is the only variant
that beats the baseline on every map and has no bad case: 34% faster on
`large_indoor` at 97.4% against 95.7%, exploring the most ground of the four,
and 38% faster on `small_indoor` with a quarter of the revisits.

B alone is fastest on `large_indoor` but leaves drones idle 40% of the time on
`small_indoor` with a 36% workload split. A alone now reaches 97.4% — it was
rejected on 87.4%, and that number moved because the *sensor* changed, not the
allocation.

`comb_indoor` has stopped discriminating: baseline and A produce byte-identical
runs, and mission length fell from ~1100 ticks to ~390. It was built to expose
thrashing that the elevation sweep largely removed. It still earns its place as
a clearance and doorway test; it is no longer an allocation benchmark.

**Three verdicts on the same question, and only the middle one was an error.**
Revision 1 recommended A+B on a metric confounded by coverage. Revision 2
rejected everything after that metric was corrected. Revision 3 recommends A+B
again — not because revision 2 was wrong about the data it had, but because the
system underneath it changed. Worth separating: a wrong measurement is a
mistake, a changed system is not.

---

## 2026-09-20 — Scaling KPI, and splitting one number into two causes

`large_indoor` showed drones leaving a region and returning 600-1000 ticks
later, which had been loosely called "wasted travel". Sweeping drone count and
classifying each long-gap revisit by *where* it happened shows it was never one
phenomenon.

```
drones  ticks  t@95%   speedup   corridor rev  room rev   balance
     1   3414   2371        —              89       225     100%
     2   1664   1311     1.81x             74        73      96%
     3   1112    811     2.92x             47        49      92%
     4   1123    625     3.79x             65        54      87%
     5    894    517     4.59x            103       134      79%
```

### The scaling KPI passes

4.59x from one drone to five, against CLAUDE.md's >=2x commitment, with
per-drone efficiency between 0.90 and 0.97 throughout — near-linear. 1->3 is
2.92x against the >=1.5x figure. This is the first time the KPI has been
measured beyond three drones; `large_indoor`'s config only defined three start
positions, and `build_mission` refuses to invent more (correctly — a start
position has to be collision-free, in bounds and clear of its neighbours). Two
were added on the east-west corridor arm.

### Two causes, pulling opposite ways

**Room revisits invert with drone count**: 225 at one drone, 49 at three. If
backtracking were contention it would rise, not fall. A single drone has nobody
to contend with, so those 225 returns are the *strategy* sending it back across
the map — `NearestFrontier` picks the cheapest reachable frontier, and once a
neighbourhood is cleared the cheapest remaining one is often somewhere it has
already been.

**Corridor revisits rise again past three drones**: 47 at three, 103 at five,
with balance decaying 100% -> 79%. `large_indoor` is a corridor *cross* — one
junction, four quadrants — so every inter-quadrant trip crosses the same cells.
That is the topology's cost, not a strategy failure, and it is what contention
looks like when it arrives.

Three drones happens to sit at the crossover for this map. Note that
ticks-to-completion *worsens* from 3 to 4 drones (1112 -> 1123) while
time-to-95% keeps improving: the tail is contention, not exploration.

### Why this matters for what to fix

Neither allocation change (global assignment, target tolerance) moved the
long-gap numbers much, and now it is clear why: both change *who goes where*,
and the dominant cause at low drone counts is *what is worth going to*. That is
`FrontierStrategy`'s job, and it is where Feature 7 aimed before failing on
cost.

It also means **the single "revisited cells" figure used in every benchmark so
far conflated a topology cost with a strategy defect**, and they move in
opposite directions. Reporting them separately is not a refinement; without it
the aggregate can stay flat while both halves change.

---

## 2026-09-20 — The backtracking was the floor plan, not the strategy

Correcting the entry above. It concluded that `large_indoor`'s long-gap revisits
were a `NearestFrontier` defect, on the grounds that room revisits *invert* with
drone count (225 at one drone, 49 at three) — which rules out contention. The
inversion is real. The attribution was wrong.

`loop_indoor` was built as the control: same 50x50 extent, same config, same
everything but the floor plan — a racetrack corridor with twelve rooms, every
one with two doorways, and **no cut vertex anywhere**. Its test suite proves the
property rather than asserting it: plan a route, wall off its middle with a
block wider than any passage in the scene, re-plan, and assert a second
essentially disjoint route exists (measured: 272 vs 271 cells, 2 shared). The
same procedure on `large_indoor` returns `None` for the second route. That
contrast is the experiment.

```
                  room cells   room revisits (1 drone)   per 1000 cells
large_indoor           50090                       225              4.5
loop_indoor            43546                        15             0.34
```

**Thirteen times fewer.** Corridor share differs between the maps (13.3% vs
22.3% of free space) so corridor counts are not directly comparable, but the
room figure is, and a 13x gap is not explained by a 1.7x difference in zone
share.

### Why

`large_indoor` is a **tree**: one junction, four quadrants, no second route
anywhere. Finishing one quadrant and starting another forces a return through
the junction and back across space already visited. That return is structural —
no target-selection policy avoids it, because there is no other way through.
`loop_indoor` lets a drone circulate forward and it simply does not backtrack.

### What this costs the previous conclusion

The strategy is not exonerated — 15 revisits is not 0, and the mechanism
described earlier (once a neighbourhood is cleared, the cheapest remaining
frontier is often back the way you came) is real. But it is a minority of the
effect on the map where it was measured, and **Feature 7 was aimed at something
that was never the dominant cause**. Its failure cost less than it appeared to.

### The methodological point

Three diagnostics in a row have now attributed an effect to the wrong layer:
the division-of-labour metric confounded by coverage, the num_rays anomaly that
was a premature stop, and this. The common shape is measuring one system and
concluding about another. The control here — an identical configuration over a
different topology — is what separated them, and it is cheap. **A benchmark with
one map cannot distinguish a strategy property from a map property.**

### Also measured

`loop_indoor` at one drone finishes in 2296 ticks against `large_indoor`'s 3414,
consistent with the backtracking account. At three drones it shows *more*
corridor revisits (144 against 47) — the ring is a shared thoroughfare every
drone circulates, where the cross is a junction they pass through. Different
topologies, different bottlenecks, and neither is a strategy defect.

### Addendum — five drones on the loop, and a confound in the comparison

```
loop_indoor   drones  ticks  t@95%   corridor rev  room rev  balance
                   1   2296   2047             31        15     100%
                   3   1432    807            144        38      85%
                   5    688    641              1         0      86%
```

Five drones: **one corridor revisit, zero room revisits**, and 3.19x scaling
1->5. That is not drone count doing the work. The five spawns sit on four ring
legs plus a corner, so each drone owns a sector and never traverses; three
spawns cover three of four legs, so they must travel to reach the south.

**Which exposes a confound in the cross-map comparison above.**
`large_indoor` clusters every spawn at the corridor junction; `loop_indoor`
spreads them around the ring. The one-drone rows are clean — a single spawn
either way — so the 13x room-revisit finding stands. The multi-drone rows were
never like-for-like and should not be read as topology alone.

Tested directly on `large_indoor`, three drones:

```
spawn layout            ticks  t@95%   corridor rev  room rev
clustered (current)      1112    811             47        49
spread along the arm     1152   1084             33        37
```

Spreading cuts revisits by a third and costs **33% more time to 95%**. So there
is no general "spread the spawns" rule: the loop's result came from spawns
matching that topology's natural sectors, and a cross has no sectors to match.
Spawn placement is a real lever, it interacts with the floor plan, and it is not
free.

Worth noting what this means for the KPI numbers: **start positions are a tuned
parameter of every scenario**, as load-bearing as the allocation policy, and
nothing in the config says so. A scenario author picking spawns is choosing part
of the result.

---

## 2026-09-21 — We tested the next-best-view hypothesis instead of building it

**The hypothesis**, carried since Sprint 2 and restated at the close of Sprint
2.5: `NearestFrontier` wastes travel because it ranks frontiers on distance
alone and ignores how much each would reveal, so a strategy weighing expected
information gain would explore measurably faster.

That claim had already cost three abandoned attempts at
`InformationGainFrontier`, each killed on cost — tracing line-of-sight from
every candidate against a *belief* map ran ~17x slower per tick for no measured
benefit. A fourth attempt would have been the third time this project built
something before measuring whether it was worth building. So we tested the
hypothesis directly.

### The instrument

`benchmarks/oracle.py` — a `FrontierStrategy` that cheats. It scores each
candidate with the *exact* number of unknown cells a visit would reveal, traced
over the ground-truth scene rather than estimated from the map so far. No
implementable strategy can score better than exactly right, so it bounds the
whole family from above: whatever it fails to win is not available at any
price.

It deliberately does not cheat at two things. **Routing** stays on the belief
grid through known-free cells, exactly as the shipped system flies — an oracle
taking shortcuts through unmapped space would be measuring a different system.
**Candidates** come from the live map — it chooses better among the same
options, it does not invent options. So the bound covers target selection and
nothing else, which is exactly what a `FrontierStrategy` controls.

The instrument is tested (`tests/unit/test_benchmarks/test_oracle.py`, 10
tests) for a specific reason: three diagnostics in this project have now
produced confident wrong conclusions because the *instrument* was wrong, not
the system. One test asserts the visible set of an open room is a complete
disc, because too sparse a ray fan would silently under-count gain at range and
quietly deflate the bound in exactly the direction that flatters the verdict.

### Finding 1 — the strategy seam selects nothing in any shipped scenario

Found before the sweep could run, by instrumenting the live call. Over four
missions on two maps at one and three drones: **718 selection calls, every one
with exactly one candidate.** Tick counts reproduced the known baselines
exactly (3414 / 1112 / 2296 / 1432), so the probe was not perturbing the runs.

The cause is `assignment: global`, which all four scenarios ship. In
`_assign_globally` the target is chosen by `coordination/allocation.py` — a
Dijkstra cost field plus a swarm-wide matching — and the strategy is then handed
a one-element list:

```python
assignment = strategy.select(grid, [by_cell[target]], state.cell, claimed)
```

`NearestFrontier` is a router. Its cost ranking, its lower-bound pruning and
its spreading penalty never execute in any shipped configuration. The control
confirms the mode is the cause — same map under `greedy`:

```
mode                 calls with >1 candidate   oracle would differ
global (shipped)                          0%                     —
greedy, 1 drone             99% (up to 16)                    89%
greedy, 3 drones                       100%                    30%
```

**This reframes all three earlier failures.** `InformationGainFrontier` was not
merely aimed at a minority cause; it was aimed at a seam the shipped system
routes around. It would have been invisible even had it worked.

It is also an architecture-level flag: CLAUDE.md names `FrontierStrategy` as
one of four stability points and describes it as the extensibility seam for
exploration policy. In the code as shipped, it does not select. The swarm's
actual exploration policy is "nearest frontier, globally matched", and there is
no notion of expected information anywhere in the system.

### Finding 2 — a perfect strategy is roughly break-even

Swept on `large_indoor` under `greedy`, the only mode where a strategy can act.
`decay` weights distance against gain; `decay=0` is pure information gain and a
large `decay` collapses onto nearest-frontier, so the family contains the
baseline at one end and the sweep is a bound over it rather than a verdict on
one weighting.

```
drones  variant          ticks   t@95%      cov   revis   path
     1  baseline          3414    2371   97.37%     503   3413
     1  oracle d=0        4000       —   91.43%     992   4000   capped
     1  oracle d=0.05     4000    2043   97.34%     591   4000   capped
     1  oracle d=0.1      3862    1934   97.36%     664   3861
     1  oracle d=0.2      3303    1976   97.36%     460   3302
     1  oracle d=0.5      3205       —   88.76%     232   3205   stalled
     3  baseline          1049     743   97.36%     229   2864
     3  oracle d=0        2987    1398   97.35%    1991   8754
     3  oracle d=0.05     1427     686   97.35%     529   4217
     3  oracle d=0.1      2522     619   97.34%     666   4509   stalled
     3  oracle d=0.2      1411     617   97.36%     508   4076
     3  oracle d=0.5      1157     666   97.36%     308   3375
```

Best completed oracle run against baseline:

```
                time to 95%   total ticks   distance travelled
1 drone  d=0.2         -17%           -3%                 -3%
3 drones d=0.2         -17%          +35%                +42%
```

**The hypothesis does not survive.** A perfect, unimplementable oracle reaches
95% coverage 17% sooner and pays for it with a 35% longer mission and 42% more
travel at three drones. It front-loads the big reveals and then pays to mop up
what it scattered. That is the ceiling; every real implementation sits below
it, and the one that was attempted ran 17x slower per tick, which swallows a
17% gain several times over.

`d=0` — pure information gain, distance ignored — is catastrophic at both swarm
sizes: capped at 91% coverage with one drone, 8.7x the revisits with three.
Distance dominates gain on these maps. That is a property of the problem, not a
tuning artifact.

Caveats kept in the open: three of twelve runs ended capped or stalled, so
their tick counts are censored rather than measured and are excluded from the
comparison; and this is one map, so it is a statement about `large_indoor`
rather than a law.

### Decision

**Do not build a next-best-view strategy.** It is bypassed in the shipped
configuration, and where it is not bypassed a perfect version is break-even at
best. Recorded as a closed question rather than deferred work.

The instrument is kept rather than deleted. A decision not to build is only
worth as much as the evidence behind it, and in three weeks the claim "we
measured the ceiling at 17%" is unverifiable without the code that measured it.

### What this opens instead

Two real levers, both cheaper than NBV:

1. **The allocator, not the strategy.** If expected information gain is worth
   having at all, it belongs in `allocate`'s cost function, which is currently
   pure distance. That is where target selection actually happens.
2. **`greedy` may now beat `global`.** On `large_indoor` at three drones,
   `greedy` finished in **1049** ticks against `global`'s **1112** at identical
   coverage. A+B was adopted before the elevation-sweep sensor landed, so the
   allocator may have quietly regressed against a sensor that no longer exists.
   One run, so it is a question rather than a finding — but a cheap one to
   settle.

### Methodological note

This is the first time the project has answered "should we build X" by building
an instrument instead of building X. It cost a day and closed a question that
had already consumed three attempts. The generalisation worth keeping: **an
upper bound is usually cheaper to measure than a feature is to build, and it
can only be measured before the feature exists.**

---

## 2026-09-21 — Sprint 3 — the swarm survives losing a drone

A drone can now fail mid-mission — go silent, or stop responding to motion
commands — and the master notices from symptoms alone, releases its frontier,
routes around the wreck and finishes the map. The mechanics are in
`design.md` §5.8–5.9. This entry records what the code and git history do not:
why a few choices went the way they did, and what the reviews caught.

### D1 — a tick had no duration

The Tier-2 KPI says "< 2 s". Nothing in the system had seconds. `tick()` never
advances MuJoCo time — `mj_step` is never called — and a step is a one-cell
teleport, so a tick is a unit of *decision*, not of time. Measuring the KPI
meant first deciding what a second is.

`drones.cruise_speed` (m/s, required) is the answer, and
`tick_seconds = map.resolution / cruise_speed` is derived rather than stored so
it can never disagree with the grid. At 1.0 m/s on `large_indoor`'s 0.2 m cells
a tick is 0.2 s, so the KPI is < 10 ticks. Timeouts stay in ticks, like every
other `*_ticks` setting.

Worth being plain about what this buys: the seconds are **nominal**. They are a
configured speed applied to a grid step, not a measurement of anything
physical. A latency of "0.40 s" means "2 ticks at the speed we said drones fly".
That is the honest unit for a teleporting simulator, and it is not the same
claim as a real swarm reassigning in 0.4 s.

### Observe first, and what it bought

The tick used to be sense → assign → move. Failure detection needed a place,
and the obvious one — at the end of `_move`, right where the unrealized move is
seen — was wrong. A drone declared there would release its frontier *after*
that tick's assignment pass, adding a tick of latency to every reassignment.

So declarations moved into a new first phase: **observe → sense → assign →
move**. A drone declared on tick *t* releases its frontier before tick *t*'s
assignment runs, and a teammate can take it the same tick. That is also why
detection latency and reassignment latency are the same number in this system.

### The master trusted its own commands

Stuck detection needs the master to notice that a commanded move did not
happen. It could not: `_move` teleported the drone and wrote the *commanded*
cell straight into `DroneState.cell`. The master's belief about where a drone
was came from its own command log, not from the `Localizer`.

That was a latent bug independent of this sprint — it had never bitten only
because a healthy teleport always lands. It is fixed by reading the pose back
after every move and letting the state follow it. The zero-regression gate was
that this must change nothing for a healthy swarm, and it changed nothing:

```
small_indoor   drones   ticks   coverage   map hash (local)
before              1     498     0.9799   09f950b017aff423
                    2     254     0.9822   4c8826303df6733d
                    3     197     0.9831   ced633862f48d60e
after          identical on all three
```

CI asserts the ticks and coverage; the hash is checked locally only, because
float results can differ across CPUs.

### The CI gate had never run the whole suite

Found while promoting Sprint 2's tests to regression, before any feature work.
`CURRENT_SPRINT` was still `2` — never bumped when Sprint 2 closed, nor after
2.5 — so all 283 of their tests were still "progression", and **every push this
sprint had run 103 of the 365 tests**. Worse, the push gate ran
`regression or sanity` and the PR gate ran `progression or sanity`: the two
*split* the suite, and no gate had ever run all of it. It went unnoticed because
Sprint 1 is small.

The fix makes the labels label rather than select: both run on every push as
separate steps (358 regression tests at the switch), acceptance on PRs to
`main`. A test tagged with a sprint ahead of `CURRENT_SPRINT` now fails
collection with a message to bump it, which is exactly when a forgotten bump
first shows. It would have caught this the first time a `sprint(3)` test was
written.

Running everything had a price: with coverage on, the push gate measured
**6 min 9 s** against the 5-minute budget. Decided: coverage is collected only
on PRs to `main`; pushes run Regression and Progression without `--cov`
(PR #24). Measured after that change: the 358 regression tests in 2 min 42 s
instead of 6 min 9 s, and the whole push job in 4 min 37 s. The margin did not
last. Feature 11 added a ~54 s integration module, and the first push after it
merged took **6 min 15 s**, progression alone 3 min 13 s. The push gate is over
budget again without coverage, and that is open.

Also found while writing this entry: the pre-commit hook ran
`regression or sanity`, which after the bump had quietly grown to include the
7 acceptance tests — minutes on every commit. Same root cause as the CI gate:
a selection written when the regression label held no acceptance tests. Fixed in PR #26,
which makes it `(regression or sanity) and not acceptance`.

Also decided: the console's Streamlit UI tests are not CI-gated. The console is an
optional extra, not core — it needs to work, not to gate a push — so CI
installs without the `ui` extra and those tests skip there, while the
plain-Python console tests still run.

### Three plan tests that could not fail

The Feature 10 test list was written before the code, and review found three
of those tests weaker than their names. All three were weaknesses of the plan,
not of the implementation.

- **Test 7 — "the released frontier can be handed out the same tick".** The
  plan's test asserted that the frontier was *eventually* taken over or
  mapped, which passes whatever the tick order. Replaced by a test that finds
  a real same-tick handout — start positions found by brute-force search over
  a small room, so no private state is poked — and asserts that on the
  declaring tick the failed drone holds nothing and the only holder of its
  frontier is an `ACTIVE` teammate. Moving `_observe` after `_assign` now
  fails it.
- **Test 11 — "a wreck in a corridor is passed without thrashing".** The plan's
  test used an open room and checked clearance only; there was no corridor and
  no bound. Added a room split by a wall with a single gap and a wreck in the
  gap. Measured: 4 target changes after the declaration, the mission finishes
  on tick 26, and the surviving drone crosses at Chebyshev distance 3. With the
  overlay disabled, the same run hits the 600-tick cap. Candidly, the
  target-change bound (8, twice the measured count) barely discriminates —
  3–9 changes with the overlay, 5–9 without, across configurations. The path
  clearance and finish-before-the-cap assertions carry the test; the bound only
  guards against runaway thrash.
- **Test 8 — "a failed drone is never sent home".** For a stuck drone this was
  undetectable as written. Now checked by event: with `return_to_base_ticks: 1`,
  no `returning_to_base` record names the failed drone from its `drone_failed`
  record onward, while a healthy twin run shows the same drone *is* sent home.
  The mutation run found something worth recording: the test fails only when
  **both** the `ACTIVE`-only guard and the wreck overlay are removed — the
  overlay alone puts the wreck's own cell inside its footprint, so no home
  route exists. The guard is defence in depth, and no test isolates it.

One more, from a different test: "a yielding drone is never stuck" passed
vacuously at the plan's 0.75 m start spacing, because no drone ever yielded
(0 wait-ticks). At 0.5 m there are 9. The test keeps an assertion that a wait
actually happened.

The pattern is Sprint 2's (`design.md` §7.3) in a new form: a test written from
a *description* of the behaviour asserts something the behaviour implies, and
the implication is often true for other reasons too.

### Measured: the latency KPI and what losing a drone costs

`benchmarks/failure_recovery.py`, `large_indoor`, 5 drones, drone 1 failed at
tick 300 (mid-mission: the healthy swarm reaches 95% at tick 517):

```
scenario            ticks  t@95%     cov   health  latency
large_indoor          894    517  97.37%        —        —
failure_injection    1289    579  97.37%     lost   2 ticks = 0.40 s
failure_stuck        1369    579  97.37%    stuck   3 ticks = 0.60 s
```

**The KPI is met in both modes**, and coverage with a drone lost equals the
healthy run. The healthy row reproduces the known 894 / 517 exactly, which is
an independent check that the scenario files and the tick derivation are
right.

The cost is mission length: **+44%** total ticks for silent, **+53%** for stuck,
and **+12%** time to 95%. The swarm reaches the coverage target nearly on time
and pays in the tail. Stuck costs more than silent here; nothing was measured
that would separate the causes, so this entry does not guess at one.

On `small_indoor` (3 drones, drone 2 failed at tick 40) the same recovery is
guarded on every push. It measured 2 ticks silent and 3 ticks stuck (0.1 s
ticks there); what it asserts is latency under 2 s and coverage ≥95%. The
module first took ~79 s against the plan's ~20 s estimate — each real 3-drone
mission costs ~20 s, and the estimate did not account for that. A
module-scoped fixture cut it from four missions to three, and it now takes
~54 s.

### What stays open

- **A stuck drone that is never commanded cannot be detected.** By design: the
  detector needs a move to fail, and a motor fault on an idle drone harms
  nothing. Reported as "undetected (never commanded)", not as a KPI miss.
- **The `FrontierStrategy` seam is still bypassed under `assignment: global`**
  (2026-09-21, above), and the `greedy`-versus-`global` question carried from
  Sprint 2.5 is still unsettled.
- **Console minors, deferred to the sprint's final review:** the non-zero-exit
  warning names "blocked" although a blocked run exits 0; every rerun
  re-validates every scenario, MuJoCo contact checks included; a browser
  reload loses the handle on a live run; a failed launch surfaces as a
  Streamlit traceback; a non-default `SWARM_ASSETS_DIR` is a test hook only.
- **`run.json`'s config snapshot is not loadable as a scenario YAML** — two
  keys are spelled differently. No re-run feature needs it yet, so the inverse
  mapping was not built.
- **Two acceptance tests call `master.tick()` directly**, bypassing failure
  injection. Harmless today — neither scenario schedules a failure — but a
  failure scenario added to them would silently run healthy.
- **The CI push gate is over its 5-minute budget again** — 6 min 15 s after
  Feature 11, without coverage. No decision yet.
