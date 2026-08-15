"""A* path planning over the occupancy grid.

Plans the shortest safe route of grid cells from a start cell to a goal cell
(typically a frontier region's representative cell). 8-connected movement with
an octile heuristic and integer edge costs. Only free cells are traversable —
occupied *and* unknown cells are blocked, so a planned path never crosses
unmapped space.

Reads `mapping` (`OccupancyGrid`) read-only; `planning` never imports
`coordination`.
"""

from __future__ import annotations

import heapq
from typing import Protocol

from swarm_mapping.mapping.grid import OccupancyGrid

# Integer edge costs keep A* in exact integer arithmetic — deterministic, with
# no floating-point comparisons. 14 ≈ 10 * sqrt(2) is the diagonal step.
_COST_ORTHOGONAL = 10
_COST_DIAGONAL = 14

# 8-connected neighbor offsets as (d_col, d_row).
_ORTHOGONAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

Cell = tuple[int, int]


class PathPlanner(Protocol):
    """Interface for planning a path between two grid cells.

    The only implementation is `AStarPlanner`. A sampling-based planner (e.g.
    RRT*) would be a second, and is only relevant once physics-based flight is
    in scope — see docs/progress.md for that comparison.
    """

    def plan(self, grid: OccupancyGrid, start: Cell, goal: Cell) -> list[Cell] | None:
        """Return a path of cells from start to goal, or None if unreachable."""
        ...


def path_cost(path: list[Cell]) -> int:
    """Return the total cost of a path in the planner's integer units.

    Lives here rather than in the caller so the 10/14 step costs have exactly
    one definition — a consumer scoring routes (e.g. `NearestFrontier`) ranks
    them on the same scale A* minimized.

    Args:
        path: Contiguous cells as returned by `PathPlanner.plan`.

    Returns:
        Summed step cost; 0 for an empty or single-cell path.
    """
    total = 0
    for (c0, r0), (c1, r1) in zip(path, path[1:], strict=False):
        diagonal = c0 != c1 and r0 != r1
        total += _COST_DIAGONAL if diagonal else _COST_ORTHOGONAL
    return total


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    """Walk parent pointers back to the start, returning start→goal order."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


class AStarPlanner:
    """A* path planner over the occupancy grid.

    8-connected movement with an octile heuristic and integer edge costs
    (10 orthogonal, 14 diagonal). Only free cells are traversable — occupied
    and unknown cells are blocked. Diagonal moves obey a no-corner-cutting rule:
    both shared orthogonal neighbors must be free, so the drone never clips a
    wall corner or squeezes through a diagonal gap.

    Args:
        free_threshold: Occupancy probability below which a cell is traversable.
    """

    def __init__(self, free_threshold: float = 0.4) -> None:
        self._free_threshold = free_threshold

    def plan(self, grid: OccupancyGrid, start: Cell, goal: Cell) -> list[Cell] | None:
        """Plan a shortest path from start to goal.

        Args:
            grid: The occupancy grid to plan over.
            start: The (col, row) start cell (the drone's current cell).
            goal: The (col, row) goal cell (e.g. a frontier's representative cell).

        Returns:
            The path as (col, row) cells from start to goal inclusive, or None
            if the goal is not reachable through free space.
        """
        prob = grid.probability()

        def is_free(col: int, row: int) -> bool:
            return grid.in_bounds(col, row) and bool(
                prob[row, col] < self._free_threshold
            )

        if not is_free(*start) or not is_free(*goal):
            return None
        if start == goal:
            return [start]

        goal_col, goal_row = goal

        def octile(col: int, row: int) -> int:
            dx = abs(col - goal_col)
            dy = abs(row - goal_row)
            return _COST_ORTHOGONAL * max(dx, dy) + (
                _COST_DIAGONAL - _COST_ORTHOGONAL
            ) * min(dx, dy)

        start_h = octile(*start)
        # Heap entries are (f, h, cell): ties in f break by h, then by cell, so
        # the expansion order — and thus the returned path — is deterministic.
        open_heap: list[tuple[int, int, Cell]] = [(start_h, start_h, start)]
        g_score: dict[Cell, int] = {start: 0}
        came_from: dict[Cell, Cell] = {}
        closed: set[Cell] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return _reconstruct(came_from, current)
            closed.add(current)

            col, row = current
            base_g = g_score[current]
            for moves, step, diagonal in (
                (_ORTHOGONAL, _COST_ORTHOGONAL, False),
                (_DIAGONAL, _COST_DIAGONAL, True),
            ):
                for d_col, d_row in moves:
                    n_col, n_row = col + d_col, row + d_row
                    if not is_free(n_col, n_row):
                        continue
                    if diagonal and not (
                        is_free(col + d_col, row) and is_free(col, row + d_row)
                    ):
                        continue  # no corner-cutting
                    neighbor = (n_col, n_row)
                    if neighbor in closed:
                        continue
                    tentative = base_g + step
                    if neighbor not in g_score or tentative < g_score[neighbor]:
                        g_score[neighbor] = tentative
                        came_from[neighbor] = current
                        n_h = octile(n_col, n_row)
                        heapq.heappush(open_heap, (tentative + n_h, n_h, neighbor))

        return None
