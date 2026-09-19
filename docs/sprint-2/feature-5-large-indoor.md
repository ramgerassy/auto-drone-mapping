# Feature 5 — Large indoor scenario

Branch: `feat/large-indoor`
Depends on: Feature 4c (the clearance rule constrains every doorway)
Decision log: [`docs/progress.md`](../progress.md), 2026-09-19 entry

---

## Goal

A 50 m x 50 m indoor floor plan with corridors and doorways — the scenario that
makes coordination matter, where `small_indoor`'s single open room does not.
Deliverable is the MJCF plus its config; the acceptance run against it is
Feature 6.

## Layout — a corridor cross with rooms off it

```
  +25 +-------------------------------------------+
      |  NW-far    |     |         |    NE-far    |
      |------D-----|     |         |-----D--------|
      |  NW-near   |  N-S corridor |    NE-near   |
      |            |    (3 m)      |              |
      |=====D======|               |======D=======|
      |                                           |   E-W corridor (3 m)
      |=====D======|               |======D=======|
      |  SW-near   |               |    SE-near   |
      |------D-----|     |         |-----D--------|
      |  SW-far    |     |         |    SE-far    |
  -25 +-------------------------------------------+
```

Eight rooms, each reached through a doorway, plus a 3 m corridor cross. A drone
must pass a doorway to explore any room, so frontier assignment has to route
through constrictions rather than across open floor — which is the behaviour
`small_indoor` cannot exercise.

## The constraint that drives every dimension

Feature 4c measured that a gap needs **`2r + 1` cells plus one cell of
discretization slop**: a wall face landing on a cell boundary has its ray hit
point attributed to the far cell, so the mapped gap is one cell narrower than
the real one.

At the chosen `resolution` 0.2 with `clearance_radius` 0.20, `r = 1`:

| quantity | cells | metres |
| --- | --- | --- |
| theoretical minimum `2r + 1` | 3 | 0.6 |
| plus discretization slop | 4 | 0.8 |
| **doorways in this scene** | **6** | **1.2** |
| **corridors in this scene** | **15** | **3.0** |

Doorways are 50% wider than the measured floor, so the scene stays traversable
if `clearance_radius` is later raised to cover a margin, or if `r` moves to 2.

**All geometry lies on a 0.2 m lattice**, so every wall face is a true cell
boundary at `resolution` 0.2 *and* 0.1. That is not cosmetic: a face at a
half-cell offset is what turned a 3-cell gap into 2 free cells in the 4c
measurement.

## ✅ DECISION D1 (resolved) — `resolution: 0.2`, not 0.1

`small_indoor` uses 0.1. Measured cost of a single `plan()` call over a 50x50 m
grid:

```
res=0.1   500x500 (250,000 cells)  r=2  mask= 6.0ms  plan= 6.9ms
res=0.2   250x250  (62,500 cells)  r=1  mask= 0.3ms  plan= 3.1ms
```

`plan()` runs once per candidate frontier per re-selecting drone, so the
20x difference in mask cost compounds across a multi-drone mission. CLAUDE.md
budgets unit + integration CI at under five minutes for the whole suite.

**Accepted cost:** coarser cells classify wall-adjacent space less precisely,
which bears on the >=98% per-cell accuracy KPI. The KPI is measured against a
reference map produced at the same resolution, so it is not self-defeating —
but if Feature 6's acceptance run shows the margin is thin, the lever is this
value, and the geometry already supports 0.1 without moving a wall.

## Files

| File | Change |
| --- | --- |
| `src/swarm_mapping/simulation/assets/large_indoor.xml` | NEW — the floor plan |
| `scenarios/large_indoor/config.yaml` | NEW — scenario config |
| `tests/unit/test_simulation/test_large_indoor.py` | NEW — geometry invariants |

The scene follows `small_indoor.xml`'s conventions: open top (no ceiling geom,
so `--view` looks in from above), two directional lights, opaque walls, and a
`<statistic>`/`<visual>` block so MuJoCo's own viewer frames it sensibly. No
drone bodies — `SimulationEngine` injects them.

## Config

Written in the **Sprint-1 schema** that `load_config` reads today, with the
multi-drone keys Feature 6 will introduce left out rather than guessed at. The
one exception is `planning.clearance_radius`, recorded because the scene's
doorways are sized against it and the number would otherwise live only in this
document. Feature 6 owns validation and the multi-drone shape.

## Test plan (review these before I write them)

Geometry invariants, asserted against the parsed model rather than the file
text — a test that greps XML proves nothing about what MuJoCo loaded.

1. The scene loads and `SimulationEngine` injects drones into it.
2. **Every wall face lies on a 0.2 m lattice point.** The invariant that makes
   doorway widths mean what they say.
3. **Every doorway is at least 6 cells wide** at `resolution` 0.2 — asserted by
   ray-casting across each doorway and measuring the gap, not by re-reading the
   dimensions the file declares.
4. **Every room is reachable from the spawn point** under
   `AStarPlanner(clearance_radius=0.20)` on a ground-truth occupancy grid built
   by ray-casting the scene. This is the test that would have caught a doorway
   one cell too narrow, and it is the reason the scene exists.
5. The same check **fails** with a doorway artificially narrowed — so case 4 is
   not vacuous.
6. No geom overlaps the spawn point.
7. The config parses, and its `map` extents cover the full 50 m x 50 m.

## Done when

- All 7 tests green, full suite green, ruff + mypy clean.
- Case 4 verified to fail against a narrowed doorway before case 5 is kept.
- `--view` shows the floor plan from above.

## Not in this feature

The multi-drone config schema, CLI wiring, the acceptance run and the scaling
KPI — all Feature 6. This feature ships the environment they will run against.
