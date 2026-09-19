# Feature 4c — Planner clearance & separation guards

Branch: `feat/planner-clearance`
Depends on: Feature 2 (`AStarPlanner`), Feature 4 (`CentralizedMaster`)
Decision log: [`docs/progress.md`](../progress.md), 2026-08-19 entry + addendum

---

## Goal

The drone has a body; nothing in the system plans for one. Close both halves:

- **Drone ↔ obstacle.** `AStarPlanner` treats the drone as a dimensionless
  point, so a path may put its centre one cell from a wall — 0.10 m inside it.
- **Drone ↔ drone.** `min_separation` has the right units and no guard, so a
  mis-set value silently permits overlap, and nothing checks start positions.

## The bug, restated

`_teleport` puts the drone's **centre** on a cell centre (`grid_to_world`
returns the centre; the body geom has no `pos` offset). The body is a 0.30 m
box and a cell is 0.10 m, so the drone is 3 cells wide while `is_free(col, row)`
(`path_planner.py:105`) tests one cell. A cell centre is 0.05 m from the cell
boundary, so with a wall face on that boundary the body reaches 0.15 m —
**0.10 m inside the wall**. `set_drone_position` calls `mj_forward`, never
`mj_step`, so there is no contact resolution to push back.

This affects the whole path, not just the frontier terminus: A\* routes
*through* wall-adjacent cells too.

## Design — obstacle inflation

A cell is traversable only if no **known-occupied** cell lies within the
clearance radius. Computed once per `plan()` call as a boolean mask.

**Square (Chebyshev) inflation is exact here, not conservative.**
`set_drone_position` resets the quaternion to identity every teleport and no
`mj_step` runs, so the drone never rotates. An axis-aligned box against an
axis-aligned grid means Chebyshev dilation of the half-extent *is* the true
configuration-space obstacle — no circumscribed-radius slack needed.

Implementation: `r` iterations of 8-neighbour dilation over the occupied mask.
`r` is 1-3, the grid is 200×200 bools, so this is a handful of shifted ORs —
obvious, deterministic, and not worth vectorising further.

**Only known-occupied cells are inflated, never unknown.** A frontier is by
definition free-adjacent-to-unknown; inflate unknown and every frontier becomes
unreachable and exploration halts on tick 1. Threshold `0.6`, matching
`mapping/frontier.py`'s `occ_threshold`.

**The start cell is always traversable.** A drone that discovers a wall beside
itself is otherwise inside the inflated zone, `plan()` returns `None`,
`assign_all` gives it no assignment, and if that holds for every drone
`_complete` flips to `True` — the mission reports success over a half-unknown
map. The escape hatch is required for liveness, and it is safe: the drone is
already there.

**The goal cell is *not* exempt.** An unreachable wall-hugging frontier is
correct: the drone genuinely does not fit. Coverage is measured in cells
*mapped*, not cells *visited*, and the sensor reaches 12 m — the drone maps that
frontier from a cell it can legally occupy.

### The radius

An occupied cell at Chebyshev distance `k` has its near face at
`(k - 0.5) * res`. No overlap requires `(k - 0.5) * res >= h`, so the minimum
legal distance is `k_min = ceil(h/res + 0.5)` and the inflation radius is
`r = k_min - 1`.

At `h = 0.15, res = 0.1`: `k_min = 2`, **`r = 1`** — and that is *grazing*,
0.15 m against 0.15 m with zero clearance, giving a 3-cell (0.30 m) minimum
corridor, exactly the drone's width against both walls. With a safety margin
`m`, `r = ceil((h + m)/res + 0.5) - 1`; a 5 cm margin gives **`r = 2`** and a
5-cell (0.50 m) minimum corridor. The default is the margin-inclusive value.

## Design — separation guards

Three checks in `CentralizedMaster.__init__`, all fail-fast:

1. **`min_separation` floor = the body diagonal, 0.424 m.** `resolve_moves` is
   a *radial* test but the body is a square, and `L_inf <= L_2` — so passing a
   radial test does not imply the boxes are clear. `dx = dy = 0.212` gives a
   Euclidean distance of 0.30 (passing a 0.30 m threshold) while Chebyshev is
   0.212, so the boxes overlap by ~0.09 m. A radial test needs the
   **circumscribed** diameter, `0.30 * sqrt(2) ~= 0.4243`.
2. **Start positions pairwise >= `min_separation`.** `resolve_moves` only
   prevents *new* violations; it never repairs an existing one. Two drones
   spawned inside each other's disc find every target blocked, burn through
   `max_wait_ticks`, re-select, and stay frozen — a permanent deadlock from
   config alone. O(n²) at n <= 5.
3. **`min_separation_cells >= 1.0`.** Below one cell the radial check can never
   block anything the "don't enter an occupied cell" seeding does not already
   block, so the separation parameter is silently dead. Raising says so.

## Public API

```python
# src/swarm_mapping/planning/path_planner.py
class AStarPlanner:
    def __init__(
        self,
        free_threshold: float = 0.4,
        clearance_radius: float,         # metres — REQUIRED, see D1
        occupied_threshold: float = 0.6,
    ) -> None: ...
```

**`planning` must not import `simulation`.** The dependency direction allows
`planning -> mapping` only, so the radius arrives as a **parameter**, never as
an import of `DRONE_HALF_EXTENT`. (`engine.py`'s constant comment was corrected
in PR #11 for exactly this reason.) The value is in **metres**, converted to
cells inside `plan()` via `grid.config.resolution` — the grid already carries
the resolution, so no caller has to do the conversion, matching how
`CentralizedMaster` handles `min_separation`.

`simulation` needs no change: 4b already promoted `DRONE_HALF_EXTENT`.

## ✅ DECISION D1 (resolved) — required argument

`exclusion_radius` in 4b took a safe default. `clearance_radius` cannot: a
default in `planning` would hardcode a drone dimension that `planning` is not
allowed to know.

- **(a) Default `0.0`** (no inflation). Every existing call site and test keeps
  working; the fix is opt-in, and stays Python-only until Feature 6 wires the
  config. Risk: shipping the sprint with the bug live because nobody wired it.
- **(b) Required argument.** Forces the decision at each of the 7 construction
  sites (4 test files), and `0.0` becomes an explicit, greppable choice — the
  pattern 4b settled on for disabling a safety filter.

**Resolved: (b), required.** The whole finding was that a body-size assumption
went unstated; a required argument is the one form that cannot be left unstated.

## ❓ DECISION D2 — does `NearestFrontier` need to change?

`NearestFrontier` calls `planner.plan(...)` and scores by the returned cost. If
a frontier is now unreachable, `plan()` returns `None` and the existing `is
None` skip already handles it. **Expected: no change.** Flagged because
`FrontierStrategy` is a named seam and I want the "no change" on the record
rather than assumed.

## Test plan (review these before I write them)

**Inflation — `tests/unit/test_planning/test_path_planner.py`, `sprint(2)`**

1. `clearance_radius=0.0` reproduces today's paths exactly (the regression
   guard for every existing planner test).
2. A path that hugged a wall at `r=0` is pushed off it at `r=1`.
3. A corridor narrower than `2r + 1` cells becomes impassable — `plan()`
   returns `None` where it previously returned a path.
4. A corridor of exactly `2r + 1` cells is still passable, down its centre
   line (pins the radius from above; without this, `r` could grow unnoticed).
5. **Unknown cells are not inflated** — a frontier-like goal beside unknown
   space is still reachable. This is the one that would halt exploration.
6. The start cell is traversable even when fully inside the inflated zone, and
   the path leads *out* of it.
7. A goal inside the inflated zone is `None` — the drone does not fit.
8. Determinism: same grid and radius, identical path.

**Separation guards — `tests/unit/test_coordination/test_master.py`**

9. `min_separation` below 0.424 raises `ValueError` naming the floor.
10. `min_separation` at/above the floor is accepted.
11. Start positions closer than `min_separation` raise `ValueError` naming the
    offending pair.
12. `min_separation_cells < 1.0` (coarse grid) raises.

**Integration**

13. A drone flies through a 1.25 m (5-cell) doorway and maps both rooms — the
    liveness check that inflation has not walled off reachable space.
14. ~~No drone's body ever overlaps a known-occupied cell.~~ **Written and
    discarded**: at `resolution` 0.25 a 0.30 m body always overhangs its own
    cell by 0.025 m, and an occupied cell is up to half a cell larger than the
    wall inside it, so the assertion measures discretization rather than
    physical overlap. It was also vacuous in the open test room. Replaced by
    **a 0.75 m (3-cell) doorway being refused**, which fails with clearance
    disabled. See progress.md.

## Files

| File | Change |
| --- | --- |
| `src/swarm_mapping/planning/path_planner.py` | `clearance_radius`, `occupied_threshold`, inflated mask in `plan()` |
| `src/swarm_mapping/coordination/master.py` | three constructor guards |
| `tests/unit/test_planning/test_path_planner.py` | tests 1-8 |
| `tests/unit/test_coordination/test_master.py` | tests 9-14 |
| `tests/unit/test_planning/test_frontier_strategy.py`, `tests/unit/test_coordination/test_assignment.py` | call-site updates if D1 = (b) |
| `docs/sprint-2-plan.md` | mark 4c done; note 4c's clearance constrains Feature 5's doorways |

## Done when

- All 14 tests green, full suite green, ruff + mypy clean.
- Test 14 verified to fail with clearance disabled.
- `planning` still imports nothing but `mapping`.

## Not in this feature

Wiring `clearance_radius` into the config schema and CLI — that is Feature 6,
alongside the multi-drone schema it already owns. Feature 5's MJCF must cut
doorways and corridors wider than `2r + 1` cells of usable space.
