# Feature 4 — Centralized master (`coordination.CentralizedMaster`)

Branch: `feat/coordination-master` (off `sprint-2`)
Status: **☐ planned — awaiting decision on D1/D2, then tests, then implementation**

> `Coordinator` is the last of the four **named SOLID seams**. Once this
> signature lands, `visualization` and `cli` depend on the Protocol, never on
> `CentralizedMaster`.

This is the feature where Sprint 2 becomes real: Features 1–3 built the parts
(frontier detection, A\*, frontier selection) and nothing yet drives them.

## Goal

Multiple drones explore an unknown environment autonomously. Each tick the
master senses with every drone, integrates the scans into the shared map,
assigns unclaimed frontiers, and steps each drone one cell along its path —
without two drones ever colliding.

## Public API

```python
# src/swarm_mapping/coordination/protocols.py
class Coordinator(Protocol):
    @property
    def is_complete(self) -> bool: ...
    @property
    def drone_states(self) -> Mapping[int, DroneState]: ...   # read-only, for visualization
    def tick(self) -> None: ...

# src/swarm_mapping/coordination/types.py
@dataclass(frozen=True)
class DroneState:
    drone_id: int
    cell: Cell
    assignment: FrontierAssignment | None
    path_index: int
    waited_ticks: int

# src/swarm_mapping/coordination/master.py
class CentralizedMaster:
    def __init__(
        self,
        engine: SimulationEngine,
        sensor: Sensor,
        mapper: Mapper,
        strategy: FrontierStrategy,
        altitude: float,
        min_separation: float,      # metres between drone centres
        max_wait_ticks: int,        # deadlock escape threshold
    ) -> None: ...
```

All collaborators are injected — the master owns orchestration, not
construction. `drone_states` exists because CLAUDE.md has `visualization`
reading `coordination` read-only; it is the only state the seam exposes.

## Design — the tick

Three phases, each a complete pass over all drones before the next begins.
Phase separation matters: if drones sensed and moved interleaved, drone 1 would
plan against a map that already included drone 5's newest scan while drone 5
planned against a staler one. Separate phases give every drone the same map
snapshot to reason about.

**Phase 1 — sense.** For each drone: `sensor.scan(id)` → `mapper.integrate_scan`.

**Phase 2 — assign.** `mapper.get_frontiers()` once. For each drone lacking a
valid assignment, call `strategy.select(grid, frontiers, cell, claimed)` and
append the chosen region to `claimed`, so the next drone sees it as taken. An
assignment is dropped (forcing re-selection) when:

- the path is exhausted — the drone arrived;
- the next cell on the path is no longer free — something was discovered there;
- `waited_ticks > max_wait_ticks` — the deadlock escape.

Paths are **not** re-planned every tick. A\* routes only through known-free
cells, so a planned path stays valid unless a cell becomes occupied — which the
per-step check above catches. Re-planning every tick would be wasted work and
would cause target thrash as frontier centroids shift.

**Phase 3 — move.** The conflict resolution, below.

## Design — conflict resolution (settled, see `docs/progress.md`)

Decided already: **wait-on-conflict, higher drone ID has right of way**, drones
processed **descending by ID** so precedence and processing order agree.

The implementation is a single conservative rule. Keep a `reserved` map of each
drone's end-of-tick position, initialised to every drone's *current* position.
Walk drones descending by ID; a drone may advance only if its target is at least
`min_separation` from every *other* drone's reserved position:

```
reserved = {id: current_cell for every drone}
for drone in sorted(ids, reverse=True):
    target = next cell on path
    if all(dist(target, reserved[other]) >= min_separation for other != drone):
        reserved[drone] = target          # advance
    else:
        pass                              # hold; reserved stays at current cell
```

Because unprocessed drones are still represented by their *current* cells, this
**subsumes the swap conflict** with no separate check: drone 5 cannot move into
drone 1's cell while drone 1 moves into drone 5's, since drone 1's current cell
blocks drone 5 outright. One rule, both conflict classes.

Known limitation to document rather than fix: a convoy travelling in ascending-ID
order advances at half speed, because the follower is processed *before* the
leader vacates. At 1–5 drones this costs a tick here and there and is not worth
the complexity of a dependency-ordered resolution.

**Separation is a distance, not a cell.** At `resolution: 0.1` a cell is 10 cm
and a drone body is larger, so `min_separation` is metres, compared against cell
centres in world coordinates.

**Extraction for testability:** this resolution is a *pure function* of
(current cells, desired cells, separation) → final cells. It lives in
`movement.py` and is tested exhaustively without MuJoCo, per the testing
philosophy. The master just calls it and teleports accordingly.

## ⚠️ DECISION D1 (needs your call) — the `Coordinator` seam shape

| Option | Signature | Notes |
| --- | --- | --- |
| **A. `tick()` + `is_complete`** (recommended) | caller owns the loop | `--view` can render between ticks; the renderer reads `drone_states` per frame; tests can step one tick at a time |
| B. `run(max_ticks)` | coordinator owns the loop | simplest call site, but the live viewer cannot draw mid-mission and tests cannot inspect intermediate state |
| C. Both | `run()` as a loop over `tick()` | superset, but adds a seam method with no second implementation needing it |

**Recommendation: A.** Sprint 1's `--view` already renders inside the loop, and
Sprint 2 keeps multi-drone `--view` (Feature 6). Option B would make that
impossible. A `run()` convenience, if wanted, belongs in the CLI, not the seam.

## ⚠️ DECISION D2 (needs your call) — how much of `cli.py` moves now

The sprint plan puts "refactor tick loop out of `cli.py`" in this feature, but
`run_pipeline` currently hardcodes the Sprint-1 patrol.

| Option | Feature 4 does | Risk |
| --- | --- | --- |
| **A. Build + unit-test only** (recommended) | `CentralizedMaster` fully built and tested; `cli.py` untouched | Feature 4 ships nothing user-visible; Feature 6 wires it |
| B. Full rewire now | `run_pipeline` drops patrol, drives the coordinator | Feature 4's PR grows to include CLI, config schema and the e2e test — the things Feature 6 exists to do |

**Recommendation: A.** `tests/unit/test_cli.py` only covers `interpolate_segment`
(a pure helper), so nothing breaks either way — but the integration test
`tests/integration/test_e2e.py` drives `run_pipeline`, and rewiring it now drags
multi-drone config schema into this branch. Feature 6 already owns that.

## Determinism (hard requirement)

- Drones iterated in **sorted (descending) id order** in every phase — never
  dict insertion order.
- Conflict resolution is order-deterministic by construction (descending walk,
  reservation table).
- Frontier assignment inherits Feature 3's total ordering.
- No wall-clock time in decisions; the `--view` pacing sleeps stay view-only.
- Same config + seed ⇒ identical drone trajectories, tick for tick.

## Test plan (review these before I write them)

`tests/unit/test_coordination/`, tagged `pytest.mark.sprint(2)`.
`test_movement.py` is pure logic, no simulator. `test_master.py` uses a tiny
inline MJCF with 2–3 drones (never a MuJoCo mock, per CLAUDE.md).
`*` = candidate `sanity`.

**Move resolution — pure, no MuJoCo**
1. \* no conflicts → every drone advances one cell
2. two drones target the same cell → higher id advances, lower holds
3. swap attempt → both blocked by the conservative rule, neither passes through
4. target within `min_separation` but not the same cell → still blocked
5. a holding drone's reserved position stays its current cell
6. a drone with no assignment holds position
7. precedence follows descending id, not input order (pass ids out of order)
8. identical inputs → identical output, repeated calls

**Assignment**
9. an unassigned drone receives the cheapest reachable frontier
10. two drones receive *different* frontiers (claimed list accumulates)
11. an in-progress assignment survives the next tick (no re-planning)
12. assignment cleared once the path is exhausted
13. assignment cleared when the next path cell is discovered occupied
14. no frontiers at all → drone idles, no exception

**Deadlock escape**
15. `waited_ticks > max_wait_ticks` → assignment dropped
16. after dropping, the drone selects a *different* frontier

**Tick loop / mission — tiny inline MJCF**
17. one tick scans with every drone and the map gains known cells
18. `is_complete` is False while frontiers remain
19. mission reaches `is_complete` when reachable frontiers are exhausted
20. **invariant:** no tick ever ends with two drones closer than
    `min_separation` — this is the "zero collisions" KPI as an assertion
21. determinism: two runs from identical config produce identical trajectories

**Protocol**
22. `CentralizedMaster` satisfies `Coordinator`

## Files

- `src/swarm_mapping/coordination/protocols.py` — `Coordinator`
- `src/swarm_mapping/coordination/types.py` — `DroneState`
- `src/swarm_mapping/coordination/movement.py` — pure conflict resolution
- `src/swarm_mapping/coordination/master.py` — `CentralizedMaster`
- `tests/unit/test_coordination/{__init__.py, test_movement.py, test_master.py}`

Layout follows `perception`/`simulation` (`protocols.py` + `types.py` +
implementation) rather than `planning`'s inline-Protocol style, because there
are four collaborating types here rather than one.

## Known dependency — teammate sensing

The perception filter (drones must not be mapped as obstacles, see
`docs/progress.md`) is deferred to its own branch. Until it lands, a multi-drone
run maps teammates as phantom obstacles. The unit tests above are unaffected
(they use hand-built grids), but the inline-MJCF tests must not assert on map
*accuracy* — only on coverage growth, completion and separation. Accuracy
assertions wait for the filter.

## Done when

- Multiple drones explore autonomously to completion, never violating
  `min_separation`, deterministically; coverage counts toward `coordination`'s
  ≥70%; ruff + mypy clean; merged into `sprint-2`.

## Not in this feature

Config schema for drone count / planning params, CLI wiring, multi-drone
`--view`, the e2e and acceptance tests, and the scaling KPI (all Feature 6);
the perception teammate filter (own branch); failure handling and heartbeats
(Sprint 3).
