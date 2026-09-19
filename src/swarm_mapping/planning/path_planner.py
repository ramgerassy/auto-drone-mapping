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

    def cost_lower_bound(self, start: Cell, goal: Cell) -> int:
        """Return a cost no real path from `start` to `goal` can undercut.

        Lets a consumer rank candidates before planning them and stop once the
        bound exceeds the best score found. Declared on the planner rather than
        computed by the caller because the bound has to match *this* planner's
        cost model — a caller assuming 8-connected octile costs would silently
        prune valid candidates from a planner that scores differently. Return 0
        to disable pruning; it is always a legal answer.
        """
        ...

    def clearance_mask(self, grid: OccupancyGrid) -> NDArray[np.bool_]:
        """Return cells the drone's body may not occupy, indexed [row, col].

        Exposed so a consumer holding an already-planned path can re-check it
        as the map changes, using the same definition of "legal cell" the
        planner used. Without it, `coordination` would have to re-implement the
        body model and the two would drift.
        """
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
        # `not isfinite` first: every comparison against NaN is False, so a
        # bare `< 0.0` would wave NaN through and fail later inside math.ceil,
        # mid-mission, with a message naming neither the parameter nor the
        # caller.
        if not math.isfinite(clearance_radius) or clearance_radius < 0.0:
            msg = (
                f"clearance_radius must be a finite value >= 0, got "
                f"{clearance_radius}. Expected the drone's half-extent plus a "
                "safety margin, in metres; pass exactly 0.0 for a point robot."
            )
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

    def cost_lower_bound(self, start: Cell, goal: Cell) -> int:
        """The octile distance — the same metric this planner's heuristic uses.

        Admissible by construction: an 8-connected path costs 10 per orthogonal
        step and 14 per diagonal, so no route can beat the octile distance.
        Clearance inflation only ever makes routes longer, never shorter, so
        the bound holds with it too.

        Args:
            start: Start cell.
            goal: Goal cell.

        Returns:
            The minimum possible path cost in the planner's integer units.
        """
        d_col = abs(start[0] - goal[0])
        d_row = abs(start[1] - goal[1])
        return _COST_ORTHOGONAL * max(d_col, d_row) + (
            _COST_DIAGONAL - _COST_ORTHOGONAL
        ) * min(d_col, d_row)

    def clearance_mask(self, grid: OccupancyGrid) -> NDArray[np.bool_]:
        """Cells the body may not occupy on this grid, indexed [row, col].

        Args:
            grid: The occupancy grid to evaluate.

        Returns:
            A boolean mask, true where the body would overlap a known-occupied
            cell.
        """
        return self._blocked_by_clearance(grid, grid.probability())

    def _check_radius_fits(self, grid: OccupancyGrid) -> None:
        """Reject a clearance that would refuse every route on this grid.

        Nothing else relates `clearance_radius` to `resolution`, so a config
        typo inflates every cell, no frontier is assignable, and the master
        reports a completed mission on tick 1 over an unexplored map. That is
        the hardest failure to read back from the output, so it fails here.

        Args:
            grid: The grid about to be planned over.

        Raises:
            ValueError: If the inflated footprint is wider than the grid.
        """
        radius = self._inflation_cells(grid.config.resolution)
        span = 2 * radius + 1
        config = grid.config
        if span > min(config.grid_width, config.grid_height):
            msg = (
                f"clearance_radius {self._clearance_radius} m inflates obstacles "
                f"by {radius} cells at resolution {config.resolution} m — a "
                f"{span}-cell footprint, wider than the {config.grid_width}x"
                f"{config.grid_height} grid. Every route would be refused and "
                "the mission would report completion over an unexplored map."
            )
            raise ValueError(msg)

    def _blocked_by_clearance(
        self, grid: OccupancyGrid, prob: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Cells too close to a known obstacle for the body to occupy.

        Only *known-occupied* cells are inflated, never unknown ones. A frontier
        is by definition a free cell adjacent to unknown space, so at `r >= 1`
        inflating unknown would make every frontier unreachable and end the
        mission on the first tick, over a map from a single scan. (At `r = 0`
        there is no inflation and the distinction does not arise.)

        Args:
            grid: The occupancy grid (read for its resolution).
            prob: The grid's occupancy probabilities, indexed [row, col].

        Returns:
            A boolean mask, true where the body would overlap a known-occupied
            *cell* — which is up to half a cell larger than the wall inside it.
        """
        radius = self._inflation_cells(grid.config.resolution)
        if radius <= 0:
            # An exact no-op. Returning `prob > occupied_threshold` here would
            # act as a second, stricter free test for any caller whose
            # free_threshold exceeds occupied_threshold, so `clearance_radius=0`
            # would not reproduce point-robot planning after all.
            return np.zeros(prob.shape, dtype=bool)

        blocked: NDArray[np.bool_] = prob > self._occupied_threshold
        for _ in range(radius):
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
        self._check_radius_fits(grid)
        prob = grid.probability()
        blocked = self._blocked_by_clearance(grid, prob)

        def is_open(col: int, row: int) -> bool:
            """Free and in bounds, ignoring clearance."""
            # in_bounds is load-bearing: numpy would wrap a negative index and
            # silently judge the start against the opposite edge of the map.
            return grid.in_bounds(col, row) and bool(
                prob[row, col] < self._free_threshold
            )

        def is_free(col: int, row: int) -> bool:
            return is_open(col, row) and not bool(blocked[row, col])

        # The start is exempt from the clearance test, and only that test. A
        # drone that discovers a wall beside itself is suddenly inside its own
        # inflated zone; without an escape it gets no assignment, and if that
        # holds for every drone the mission ends over a half-explored map.
        if not is_open(*start):
            return None
        if start == goal:
            # Before the goal's clearance test, so the exemption is not undone
            # by asking the same cell to satisfy it as a goal.
            return [start]
        if not is_free(*goal):
            return None

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
                # Exempting only the start cell is not enough: at r >= 2 every
                # neighbour of an embedded drone is inflated too, so it would
                # be stranded with no legal first step — and r = 2 is what
                # `resolution: 0.1` produces for a 0.20 m clearance. While the
                # search is still inside the zone it may move through it; once
                # it reaches open ground it may not re-enter. Escaping is
                # allowed, loitering is not.
                escaping = bool(blocked[row, col])
                passable = is_open if escaping else is_free

                for d_col, d_row in moves:
                    n_col, n_row = col + d_col, row + d_row
                    if not passable(n_col, n_row):
                        continue
                    if diagonal and not (
                        passable(col + d_col, row) and passable(col, row + d_row)
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
