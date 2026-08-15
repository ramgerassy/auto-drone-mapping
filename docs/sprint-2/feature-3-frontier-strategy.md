# Feature 3 — Frontier strategy (`planning.NearestFrontier`)

Branch: `feat/frontier-strategy` (off `sprint-2`)
Status: **✅ implemented — 20 tests green, `planning` at 100% coverage (pending merge into `sprint-2`)**

> `FrontierStrategy` is one of the four **named SOLID seams** (CLAUDE.md) — a
> stability point. Once this signature lands, `coordination` depends on the
> Protocol, never on `NearestFrontier`. Getting the signature right matters more
> than getting the scoring right; scoring is swappable behind the seam.

## Goal

Given the map, a drone's current cell, and the frontiers other drones have
already claimed, pick **which frontier this drone should explore next** and the
path to it. This is the decision layer that turns Feature 1's detected regions
and Feature 2's path planner into an actual exploration policy.

Stays stateless and a pure function of its inputs (CLAUDE.md: "`planning` in
particular should be stateless"). No memory of past assignments between calls —
the coordinator owns that state and passes it in as `claimed`.

## Public API

```python
# src/swarm_mapping/planning/frontier_strategy.py

@dataclass(frozen=True)
class FrontierAssignment:
    region: FrontierRegion   # the chosen frontier
    path: list[Cell]         # start → region.cell inclusive
    cost: int                # A* path cost in 10/14 units (logging + KPIs)

class FrontierStrategy(Protocol):
    def select(
        self,
        grid: OccupancyGrid,
        frontiers: Sequence[FrontierRegion],
        start: Cell,
        claimed: Sequence[FrontierRegion] = (),
    ) -> FrontierAssignment | None: ...

class NearestFrontier:
    def __init__(
        self,
        planner: PathPlanner,          # injected — the seam consumes the seam
        spread_radius: float = 0.0,    # world metres; 0 disables the penalty
        spread_penalty: int = 0,       # cost units added within the radius
    ) -> None: ...
```

- Returns `None` when no frontier is reachable (fully explored, or everything
  left is walled off) — the coordinator reads that as "this drone idles".
- `planner` is **injected**, not constructed internally: the strategy is tested
  against a stub planner, and swapping A\* for something else never touches this
  file (open-closed).

## Design

**Why the strategy returns the path, not just the region.** Scoring a frontier
by true travel cost *requires* planning to it. Returning the path the scoring
already computed avoids the coordinator re-planning the same route a second
time, and matches CLAUDE.md's description of `planning`: "given map + drone
state + claimed frontiers → return frontier + path".

**Reachability falls out for free.** A frontier behind a sealed wall returns
`None` from the planner and is skipped. With straight-line scoring we would
assign it, then discover the failure a tick later and waste the tick.

**Deliberately *not* in `NearestFrontier`:** `FrontierRegion.size` weighting.
"Bigger frontier = more new information" is exactly the `InformationGainFrontier`
stretch goal named in CLAUDE.md. Mixing it in here would blur the seam that
justifies having two implementations.

## ✅ DECISION D1 (resolved) — what "nearest" measures

Chosen: **Option A (true A\* path cost)**. Verified on the test grid: a frontier
3.0 m away in straight line but behind a wall costs **104**, while one 4.0 m away
down open space costs **40** — Euclidean scoring picks the wrong one.

| Option | Metric | Cost per drone per tick | Blind spot |
| --- | --- | --- | --- |
| **A. True path cost** (recommended) | A\* to every candidate; cheapest wins | `n_frontiers` × A\* | slowest; A\* over a large grid × ~dozens of frontiers |
| B. Euclidean to centroid | `hypot` on world coords | negligible | a frontier 2 m away through a wall outranks one 5 m down the corridor |
| C. Hybrid | Euclidean-sort, then A\* the top *K*, take first reachable | ≤ *K* × A\* | can miss the true optimum when walls reorder the top *K* |

**Recommendation: A.** With 1–5 drones (an explicit non-negotiable — "don't
optimize for 50"), and A\* already integer-exact and fast, correctness beats the
micro-optimization. Option C is the escape hatch if the large-indoor scenario
(Feature 5) actually shows a tick-rate problem — and C is a change *inside*
`NearestFrontier`, not to the seam, so deferring it costs nothing.

## ✅ DECISION D2 (resolved) — the spatial spreading penalty

Chosen: **Option A (hard exclude + soft radius)**. The penalty affects *ranking*
only — `FrontierAssignment.cost` reports the true unpenalized path cost.

Purpose: stop all drones converging on the same corner of the map. Applied on
top of D1's score.

| Option | Rule | Behaviour |
| --- | --- | --- |
| **A. Hard exclude + soft radius** (recommended) | A claimed region is never re-picked. Any candidate whose centroid is within `spread_radius` of a claimed centroid gets `+spread_penalty` added to its cost. | Drones spread out, but a *penalized* frontier is still taken when it's the only thing left. |
| B. Hard exclude only | Skip claimed regions; no proximity term. | Two drones happily work adjacent frontiers 0.5 m apart — no real spreading. |
| C. Soft penalty only | Penalize by distance to nearest claimed, no exclusion. | Two drones can be assigned the *same* frontier when it dominates. |

**Recommendation: A.** B under-spreads and C permits duplicate assignment; A is
the only option where "spread out" and "never strand a frontier" both hold.
Penalty is **additive and integer** (same 10/14 units as A\* cost) so scoring
stays in exact integer arithmetic — no float comparison in a decision path.

Defaults (`spread_radius=0.0`, `spread_penalty=0`) make the penalty **off**
unless configured, so single-drone Sprint-1 behaviour is unchanged.

## Determinism (hard requirement)

- Candidates are iterated in `frontiers`' existing order — `get_frontiers()`
  already sorts by `(row, col)`.
- Score comparison is integer (`cost + penalty`); ties break on the region's
  `(row, col)` cell, never on iteration order.
- `claimed` is a `Sequence`, not a `set` — no unordered iteration in the
  decision path.
- Same grid + start + frontiers + claimed ⇒ identical assignment, every run.

## Test plan (review these before I write them)

`tests/unit/test_planning/test_frontier_strategy.py`, tagged
`pytest.mark.sprint(2)`. Pure logic on hand-built `OccupancyGrid`s, no MuJoCo.
Selection tests use a **stub planner** so scoring is tested independently of A\*;
two integration-flavoured tests use the real `AStarPlanner`.

**Selection core**
1. \* single reachable frontier → returned, with a path ending at `region.cell`
2. two frontiers → the cheaper-to-reach one wins
3. no frontiers at all → `None`
4. every frontier unreachable (walled off) → `None`
5. one of two unreachable → the reachable one is chosen, not the nearer-in-
   straight-line unreachable one
6. returned `path` starts at `start` and ends at `region.cell`
7. returned `cost` equals the planner's path cost

**Wall-awareness (real `AStarPlanner`)**
8. frontier that is Euclidean-near but behind a wall loses to one that is
   Euclidean-far but down an open corridor — *this is the test that fails under
   D1-option B, and is the reason for the recommendation*

**Spreading penalty**
9. a claimed region is never re-selected (hard exclude)
10. candidate within `spread_radius` of a claimed centroid loses to an equal-cost
    candidate outside it
11. a penalized candidate is still selected when it is the only one left
12. `spread_radius=0` / `spread_penalty=0` → claimed-proximity has no effect
    (only hard exclusion remains)
13. `claimed=()` (default) behaves identically to no penalty configured

**Determinism / contract**
14. two equal-cost candidates → tie broken by `(row, col)`, stable across runs
15. repeated identical calls return an identical assignment
16. `select` does not mutate `grid`, `frontiers`, or `claimed` (statelessness)

`*` = candidate `sanity` marker.

## Files

- `src/swarm_mapping/planning/frontier_strategy.py` — `FrontierAssignment`,
  `FrontierStrategy`, `NearestFrontier`
- `src/swarm_mapping/planning/__init__.py` — exports
- `tests/unit/test_planning/test_frontier_strategy.py` — `sprint(2)`

## Done when

- Selects the cheapest *reachable* unclaimed frontier, spreads drones apart, and
  returns `None` when nothing is left; deterministic across runs; coverage counts
  toward `planning`'s ≥70%; ruff + mypy clean; merged into `sprint-2`.

## Not in this feature

`InformationGainFrontier` (stretch goal, later), the coordinator's claimed-list
bookkeeping and reassignment-on-failure (Feature 4), and config plumbing for
`spread_radius` / `spread_penalty` (Feature 6).
