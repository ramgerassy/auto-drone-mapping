# Feature 2 — A\* path planner (`planning.AStarPlanner`)

Branch: `feat/path-planner` (off `sprint-2`)
Status: **✅ implemented — 12 tests green, `path_planner.py` 100% coverage (pending merge into `sprint-2`)**

> **Movement/heuristic decision (resolved):** Option 1 — **8-connected + octile**,
> integer costs **10** (orthogonal) / **14** (diagonal), **no corner-cutting**
> (a diagonal is allowed only if *both* shared orthogonal neighbors are free).

## Goal

Given the occupancy grid, a start cell, and a goal cell (a frontier region's
representative cell), compute the **shortest safe path** of grid cells the drone
will follow. First feature in the `planning` module. Reads `mapping`
(`OccupancyGrid`) read-only — `planning` never imports `coordination`.

## Public API

```python
# src/swarm_mapping/planning/path_planner.py
class PathPlanner(Protocol):
    def plan(
        self, grid: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None: ...

class AStarPlanner:
    def plan(self, grid, start, goal) -> list[tuple[int, int]] | None: ...
```

- `start` / `goal` are `(col, row)` grid cells. The coordinator converts the
  drone's world pose → cell; `goal` comes straight from `FrontierRegion.cell`.
- Returns the path as `(col, row)` cells from `start` to `goal` **inclusive**,
  or `None` if the goal is unreachable.

## Design

- **Nodes** = grid cells.
- **Traversable = free cells only** (`p < free_threshold`). Occupied *and*
  unknown cells are blocked — the drone never plans a route through unmapped
  space (safety: an unknown cell might hide an obstacle). The goal is a frontier
  cell (free by construction) and the start is the drone's cell (free), so a
  path through known-free space always suffices.
- **A\***: `f = g + h`; expand the lowest-`f` node from a heap until the goal
  pops; reconstruct via parent pointers.
- **Connectivity / edge cost / heuristic** — the pending decision below.
- **Unreachable** (goal blocked, or no free path) → `None`.
- **`start == goal`** → `[start]`.

## ✅ DECISION (resolved) — movement model + heuristic

Chosen: **Option 1 (8-connected + octile)**. The heuristic must match how the
drone is allowed to move (an admissible `h` never overestimates the true
remaining cost, which is what keeps A\* optimal).

| Option | Moves | Heuristic `h(n)` | Notes |
| --- | --- | --- | --- |
| **1. 8-conn + octile** (recommended) | N/S/E/W + diagonals | `10·max(dx,dy) + 4·min(dx,dy)` | natural drone paths; tight admissible heuristic → fewest node expansions |
| 2. 4-conn + Manhattan | N/S/E/W only | `10·(dx + dy)` | simplest; blocky L-shaped paths |
| 3. 8-conn + Euclidean | N/S/E/W + diagonals | `10·√(dx²+dy²)` | natural paths but looser bound → more expansions (still optimal) |

**Recommendation: Option 1.** A drone is holonomic — diagonal moves are
natural, and octile is the exact admissible heuristic for 8-connected movement.

Sub-choices that come with Option 1:
- **Integer edge costs `10` (orthogonal) / `14` (diagonal ≈ 10√2).** Keeps A\*
  in integer arithmetic → perfectly deterministic, no float-comparison edge
  cases.
- **No corner-cutting**: a diagonal move is allowed only if *both* shared
  orthogonal neighbors are free — so the drone never clips a wall corner or
  squeezes through a diagonal gap between two walls.

## Determinism (hard requirement)

- Heap priority is the tuple `(f, h, cell)` → a total order, so ties never
  depend on heap internals.
- Integer `g`/`h` (the `10`/`14` scaling) → no floating-point comparison.
- Path reconstructed by following parent pointers (order-independent).
- Same grid + start + goal ⇒ identical path, every run.

## Test plan (write signatures first, review, then implement)

`tests/unit/test_planning/test_path_planner.py`, tagged `pytest.mark.sprint(2)`;
pure-logic on hand-built `OccupancyGrid`s (no MuJoCo). `*` = candidate `sanity`.

1. \* straight shot on an empty grid → shortest, contiguous path
2. path detours around a wall obstacle
3. goal walled off → `None`
4. `start == goal` → `[start]`
5. goal cell is occupied → `None`
6. path never enters an occupied or unknown cell (free-only)
7. (8-conn) diagonal shortcut beats the 4-conn detour
8. (8-conn) no corner-cutting between two occupied orthogonals
9. determinism: identical path across repeated calls
10. optimality: path cost equals the known shortest on a small grid
11. contiguity: every step is a legal adjacent move

## Files

- `src/swarm_mapping/planning/path_planner.py` — `PathPlanner`, `AStarPlanner`
- `src/swarm_mapping/planning/__init__.py` — export
- `tests/unit/test_planning/__init__.py`, `test_path_planner.py` — `sprint(2)`

## Done when

- Returns optimal, safe, deterministic paths on the test grids; unreachable →
  `None`; coverage counts toward `planning`'s ≥70%; ruff + mypy clean; merged
  into `sprint-2`.
