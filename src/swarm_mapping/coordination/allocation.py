"""Global frontier allocation across the swarm.

`assign_all`'s default is greedy and sequential: drones are served in
descending id order and each takes the best frontier still unclaimed. That is
simple and it is what the higher-id-wins right-of-way rule already implies for
movement — but it allocates badly. The highest id gets first pick of the whole
map and drone 0 chooses from the leftovers, and nothing considers what the
combination costs. Measured on large_indoor: one drone covered the western half
while another never left the corridor junction and retrod 34% of the cells it
visited.

This module allocates across the whole swarm at once instead, so a cheap pairing
for one drone can be displaced when it is worth more to another.

**One Dijkstra flood per drone** gives the true path cost from that drone to
*every* reachable cell in a single pass, so the full drone x frontier cost
matrix costs one flood per drone rather than one A* per pair.

**On optimality — this is not the Hungarian algorithm.** It is Kuhn's
augmenting-path matching over edges visited in ascending cost, which maximises
the number of drones given work and strongly prefers cheap pairings, but does
*not* guarantee the minimum total. A true minimum-cost assignment would want
`scipy.optimize.linear_sum_assignment`, and scipy is not a dependency; adding
one to improve an allocation over at most 5 drones would be disproportionate.
The claim this module makes is "better than serving drones one at a time", and
the benchmark is what tests it.

`docs/progress.md` rejected a wavefront for Feature 2 on the grounds that its
compute-once reuse needs many-to-one goals while "we have each drone to a
different frontier". That reasoning inverts here. One drone against ~30
candidate frontiers is precisely one-to-many, which is the shape a flood is
best at.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.coordination.types import Cell
from swarm_mapping.mapping.grid import OccupancyGrid

# Matches AStarPlanner's integer edge costs, so a cost from this module is
# directly comparable with one from `path_cost`.
_COST_ORTHOGONAL = 10
_COST_DIAGONAL = 14

_UNREACHABLE = -1

_ORTHOGONAL = ((0, -1), (0, 1), (-1, 0), (1, 0))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def cost_field(
    grid: OccupancyGrid,
    start: Cell,
    blocked: NDArray[np.bool_],
    free_threshold: float = 0.4,
) -> NDArray[np.int64]:
    """True path cost from `start` to every cell, in the planner's units.

    A Dijkstra flood over the same 8-connected graph `AStarPlanner` searches,
    with the same 10/14 edge costs and the same no-corner-cutting rule, so a
    value here equals what A* would return for that goal.

    It reproduces the planner's **start-cell clearance exemption** as well, and
    must: a drone that discovers a wall beside itself is inside its own inflated
    zone, and refusing to seed there produced an all-unreachable field, so that
    drone was never given work again. Measured before the exemption was added:
    drones sat idle for 27-50% of a mission.

    Args:
        grid: The occupancy grid to flood.
        start: The drone's cell.
        blocked: The planner's clearance mask, indexed [row, col].
        free_threshold: Probability below which a cell is traversable.

    Returns:
        Costs indexed [row, col], with `_UNREACHABLE` where no route exists.
    """
    prob = grid.probability()
    width, height = grid.config.grid_width, grid.config.grid_height
    costs = np.full((height, width), _UNREACHABLE, dtype=np.int64)

    def passable(col: int, row: int) -> bool:
        return (
            0 <= col < width
            and 0 <= row < height
            and bool(prob[row, col] < free_threshold)
            and not bool(blocked[row, col])
        )

    # Exempt only the start, and only from clearance — exactly as
    # `AStarPlanner.plan` does.
    start_col, start_row = start
    seedable = (
        0 <= start_col < width
        and 0 <= start_row < height
        and bool(prob[start_row, start_col] < free_threshold)
    )
    if not seedable:
        return costs

    # (cost, col, row): the cell in the tuple keeps pops deterministic when two
    # cells are equidistant, exactly as the A* heap does.
    heap: list[tuple[int, int, int]] = [(0, start[0], start[1])]
    costs[start[1], start[0]] = 0

    while heap:
        cost, col, row = heapq.heappop(heap)
        if cost > costs[row, col]:
            continue
        for moves, step, diagonal in (
            (_ORTHOGONAL, _COST_ORTHOGONAL, False),
            (_DIAGONAL, _COST_DIAGONAL, True),
        ):
            for d_col, d_row in moves:
                n_col, n_row = col + d_col, row + d_row
                if not passable(n_col, n_row):
                    continue
                if diagonal and not (
                    passable(col + d_col, row) and passable(col, row + d_row)
                ):
                    continue  # no corner-cutting, as in AStarPlanner
                candidate = cost + step
                known = costs[n_row, n_col]
                if known == _UNREACHABLE or candidate < known:
                    costs[n_row, n_col] = candidate
                    heapq.heappush(heap, (candidate, n_col, n_row))

    return costs


def allocate(
    drone_ids: Sequence[int],
    targets: Sequence[Cell],
    costs: Mapping[int, NDArray[np.int64]],
) -> dict[int, Cell]:
    """Assign at most one target per drone, preferring cheap pairings.

    Args:
        drone_ids: Drones needing a target, in the caller's preferred order.
            Ties are broken toward earlier entries.
        targets: Candidate cells, deduplicated by the caller.
        costs: Each drone's cost field from `cost_field`.

    Returns:
        The chosen target per drone. A drone with nothing reachable is absent
        from the mapping rather than mapped to None, so the caller's "did this
        drone get work" test stays a simple membership check.
    """
    if not drone_ids or not targets:
        return {}

    matrix = np.full((len(drone_ids), len(targets)), _UNREACHABLE, dtype=np.int64)
    for row, drone_id in enumerate(drone_ids):
        field = costs[drone_id]
        for column, (col, cell_row) in enumerate(targets):
            matrix[row, column] = field[cell_row, col]

    return _min_cost_matching(drone_ids, targets, matrix)


def _min_cost_matching(
    drone_ids: Sequence[int],
    targets: Sequence[Cell],
    matrix: NDArray[np.int64],
) -> dict[int, Cell]:
    """Cost-preferring matching, by augmenting paths.

    Kuhn's algorithm with each drone's edges tried in ascending cost. Every
    drone with a reachable target gets one, and a drone already holding a target
    can be displaced onto its next-cheapest when that frees a better pairing —
    which is the thing sequential greedy cannot do.

    **Not optimal.** Preferring cheap edges is a heuristic, not a minimisation;
    see the module docstring. At 5 drones the gap to optimal is small and the
    benchmark measures whether it matters.

    Unreachable pairs are skipped rather than given a large finite cost: a big
    number would still be preferred over leaving a drone idle, which would
    assign a route that does not exist.

    Args:
        drone_ids: Drones needing a target.
        targets: Candidate cells.
        matrix: Costs indexed [drone, target], `_UNREACHABLE` where no route
            exists.

    Returns:
        The chosen target per drone; drones with no reachable target are absent.
    """
    n_drones, n_targets = matrix.shape
    target_of_drone: list[int] = [-1] * n_drones
    drone_of_target: list[int] = [-1] * n_targets

    def augment(drone: int, seen: list[bool]) -> bool:
        """Try to give `drone` a target, displacing others along the way."""
        # Ascending cost, then target index: deterministic, and it makes the
        # first attempt the drone's own preference.
        order = sorted(
            (int(matrix[drone, target]), target)
            for target in range(n_targets)
            if matrix[drone, target] != _UNREACHABLE
        )
        for _, target in order:
            if seen[target]:
                continue
            seen[target] = True
            holder = drone_of_target[target]
            if holder == -1 or augment(holder, seen):
                drone_of_target[target] = drone
                target_of_drone[drone] = target
                return True
        return False

    for drone in range(n_drones):
        augment(drone, [False] * n_targets)

    return {
        drone_ids[drone]: targets[target]
        for drone, target in enumerate(target_of_drone)
        if target != -1
    }
