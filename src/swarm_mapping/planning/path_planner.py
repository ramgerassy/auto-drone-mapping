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
import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

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


def _dilate(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Expand a boolean mask by one cell in all 8 directions.

    Zero-padded rather than wrapped: an obstacle on the grid edge must not
    reappear on the opposite edge, which `np.roll` would do.

    Args:
        mask: Boolean array indexed [row, col].

    Returns:
        A new mask, true wherever `mask` was true within Chebyshev distance 1.
    """
    rows, cols = mask.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    out = mask.copy()
    for d_row in range(3):
        for d_col in range(3):
            out |= padded[d_row : d_row + rows, d_col : d_col + cols]
    return out


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

    The drone has a body, so a traversable cell is not enough: known-occupied
    cells are inflated by the clearance radius, and a cell within that radius is
    refused even though the cell itself is free. Inflation is square (Chebyshev)
    rather than radial, and that is *exact* here rather than conservative —
    `set_drone_position` resets the quaternion to identity on every teleport and
    no `mj_step` ever runs, so the drone never rotates. An axis-aligned box
    against an axis-aligned grid makes square dilation of the half-extent the
    true configuration-space obstacle.

    Args:
        free_threshold: Occupancy probability below which a cell is traversable.
        clearance_radius: The drone's half-extent plus any safety margin, in
            metres. **Required** — the whole reason this exists is that a
            body-size assumption went unstated, so it may not be left unstated
            here either. Pass exactly 0.0 to plan for a point robot.
        occupied_threshold: Probability above which a cell counts as a known
            obstacle worth inflating. Matches `mapping.frontier`'s default.

    Raises:
        ValueError: If clearance_radius is negative.
    """

    def __init__(
        self,
        free_threshold: float = 0.4,
        *,
        clearance_radius: float,
        occupied_threshold: float = 0.6,
    ) -> None:
        if clearance_radius < 0.0:
            msg = f"clearance_radius must be >= 0, got {clearance_radius}"
            raise ValueError(msg)

        self._free_threshold = free_threshold
        self._clearance_radius = clearance_radius
        self._occupied_threshold = occupied_threshold

    def _inflation_cells(self, resolution: float) -> int:
        """How many cells of Chebyshev dilation the clearance radius needs.

        An occupied cell at Chebyshev distance `k` has its near face at
        `(k - 0.5) * resolution`, so no overlap requires
        `(k - 0.5) * resolution >= clearance_radius`. The nearest legal
        distance is that `k`, and everything below it is blocked.

        Args:
            resolution: Metres per cell.

        Returns:
            The dilation radius in cells; 0 when the body fits within a cell.
        """
        return math.ceil(self._clearance_radius / resolution + 0.5) - 1

    def _blocked_by_clearance(
        self, grid: OccupancyGrid, prob: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Cells too close to a known obstacle for the body to occupy.

        Only *known-occupied* cells are inflated, never unknown ones. A frontier
        is by definition a free cell adjacent to unknown space, so inflating
        unknown would make every frontier unreachable and halt exploration on
        the first tick.

        Args:
            grid: The occupancy grid (read for its resolution).
            prob: The grid's occupancy probabilities, indexed [row, col].

        Returns:
            A boolean mask, true where the body would overlap an obstacle.
        """
        blocked: NDArray[np.bool_] = prob > self._occupied_threshold
        for _ in range(self._inflation_cells(grid.config.resolution)):
            blocked = _dilate(blocked)
        return blocked

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
        blocked = self._blocked_by_clearance(grid, prob)

        def is_free(col: int, row: int) -> bool:
            return (
                grid.in_bounds(col, row)
                and bool(prob[row, col] < self._free_threshold)
                and not bool(blocked[row, col])
            )

        # The start is exempt from the clearance test, and only the clearance
        # test. A drone that discovers a wall beside itself is suddenly inside
        # its own inflated zone; without this it gets no assignment, and if that
        # holds for every drone the mission reports completion over a
        # half-explored map. Exempting it is safe — it is already standing there.
        start_col, start_row = start
        start_traversable = grid.in_bounds(start_col, start_row) and bool(
            prob[start_row, start_col] < self._free_threshold
        )
        if not start_traversable or not is_free(*goal):
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
