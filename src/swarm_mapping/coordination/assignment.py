"""Frontier assignment across the swarm.

Pure logic: given the map, the detected frontiers and every drone's state,
decide which drone flies to which frontier. Kept free of the simulator so it is
testable without MuJoCo.

The claimed list is what makes this a *swarm* decision rather than N independent
ones: each drone's choice is appended before the next drone chooses, so no two
drones target the same frontier and Feature 3's spreading penalty has something
to push against.
"""

from __future__ import annotations

import logging
from collections.abc import Container, Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.coordination.allocation import allocate, cost_field
from swarm_mapping.coordination.types import Cell, DroneState
from swarm_mapping.mapping.frontier import FrontierRegion, frontier_cells
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.planning.frontier_strategy import FrontierStrategy
from swarm_mapping.planning.path_planner import PathPlanner

_LOGGER = logging.getLogger(__name__)


def _target_survives(
    target: Cell, live_frontiers: Container[Cell], tolerance: int
) -> bool:
    """Whether a frontier still exists at or near `target`.

    An exact-match test is too brittle to be the sole reason a committed
    assignment is dropped. A region's representative cell moves as the region
    changes shape, and wall-surface cells cross the 0.4/0.6 classification
    bands from tick to tick, so the target can leave the frontier set for a
    tick without anything having really changed. Measured under exact match:
    150 target changes per drone over 1498 ticks, 92% of them direct swaps,
    averaging ten ticks of commitment — and every swap throws away the travel
    already spent getting there.

    Args:
        target: The cell the drone is flying to.
        live_frontiers: Frontier cells this tick.
        tolerance: Chebyshev radius to search. 0 demands an exact match.

    Returns:
        True if a live frontier lies within `tolerance` of the target.
    """
    col, row = target
    for d_col in range(-tolerance, tolerance + 1):
        for d_row in range(-tolerance, tolerance + 1):
            if (col + d_col, row + d_row) in live_frontiers:
                return True
    return False


def next_cell(state: DroneState) -> Cell | None:
    """Return the next cell on a drone's path, or None if it has none.

    Args:
        state: The drone's current state.

    Returns:
        The cell one step along the path, or None when the drone is idle or has
        already reached the end of its path.
    """
    if state.assignment is None:
        return None
    step = state.path_index + 1
    if step >= len(state.assignment.path):
        return None
    return state.assignment.path[step]


def is_assignment_valid(
    state: DroneState,
    prob: NDArray[np.float64],
    free_threshold: float,
    max_wait_ticks: int,
    blocked: NDArray[np.bool_],
    live_frontiers: Container[Cell],
    target_tolerance_cells: int = 0,
) -> bool:
    """Check whether a drone should keep flying its current assignment.

    An assignment is dropped — forcing re-selection next pass — when the drone
    has arrived, when the next cell stopped being legal for the body, or when
    the drone has been blocked long enough to count as deadlocked.

    Paths are otherwise kept rather than re-planned every tick, so this test is
    the *only* thing standing between a committed path and a changed map. It
    must therefore use the same definition of a legal cell that the planner
    used, which is why `blocked` is a parameter rather than a second
    free-threshold test.

    **The clearance check is not redundant with the free check.** A* routes only
    through known-free cells, so a wall discovered later was *unknown* at plan
    time and is never itself on the path — but its inflation zone covers path
    cells that were free and stay free. Testing `prob` alone keeps such a path,
    and the drone flies the whole route with its body inside the wall. That is
    the normal exploration case, not an edge case: flying toward a frontier is
    exactly how walls beside the path get discovered.

    Args:
        state: The drone's current state.
        prob: The grid's occupancy probability array, indexed [row, col].
        free_threshold: Probability below which a cell is traversable.
        max_wait_ticks: Consecutive blocked ticks after which the drone gives up
            on this frontier and picks another (the deadlock escape).
        blocked: The planner's clearance mask, indexed [row, col].
        live_frontiers: Cells that are still frontiers this tick.
        target_tolerance_cells: How far a live frontier may be from the target
            before the target counts as gone.

    Returns:
        True if the assignment should be kept.
    """
    if state.assignment is None:
        return False
    if state.waited_ticks > max_wait_ticks:
        return False  # deadlock escape — try a different frontier
    if not _target_survives(
        state.assignment.region.cell, live_frontiers, target_tolerance_cells
    ):
        # The target stopped being a frontier while the drone was en route, so
        # there is nothing left to see there. Paths here run 60-100 steps, and
        # cells on a wall surface cross the 0.4/0.6 classification bands as they
        # accumulate mixed free and occupied evidence at grazing incidence — so
        # without this check a drone spends a hundred ticks travelling to a
        # frontier that evaporated after five, arrives at nothing, and picks up
        # another one that has since appeared. Measured on large_indoor: the
        # swarm plateaus at 97.6% coverage by tick 1000 and then burns 3000 more
        # ticks with `still_a_frontier=False` on almost every sample, never
        # terminating because some drone is always mid-journey and
        # `is_complete` needs every drone unassigned at once.
        _LOGGER.info(
            "assignment_dropped",
            extra={
                "reason": "frontier_gone",
                "drone_id": state.drone_id,
                "cell": state.assignment.region.cell,
            },
        )
        return False
    step = next_cell(state)
    if step is None:
        return False  # arrived
    col, row = step
    if bool(blocked[row, col]):
        # A near miss: the map changed under a committed path. Worth a line,
        # because the drone was one tick from flying its body into a wall.
        _LOGGER.info(
            "assignment_dropped",
            extra={
                "reason": "clearance",
                "drone_id": state.drone_id,
                "cell": step,
            },
        )
        return False
    return bool(prob[row, col] < free_threshold)


def assign_all(
    grid: OccupancyGrid,
    frontiers: Sequence[FrontierRegion],
    states: Mapping[int, DroneState],
    strategy: FrontierStrategy,
    planner: PathPlanner,
    max_wait_ticks: int,
    free_threshold: float = 0.4,
    target_tolerance_cells: int = 0,
    mode: str = "greedy",
) -> dict[int, DroneState]:
    """Assign a frontier to every drone that needs one.

    Runs in two passes. The first collects the drones whose existing assignment
    is still valid and marks their frontiers as claimed; only then does the
    second pass select for the rest. Doing it in one pass would let a drone pick
    a frontier another drone is already flying to, because that frontier would
    not yet be in the claimed list.

    Drones are processed in **descending id order**, matching the right-of-way
    rule, so the same seniority applies to target selection and to movement.

    Args:
        grid: The occupancy grid, read-only.
        frontiers: Regions from `Mapper.get_frontiers`.
        states: Current drone states, keyed by drone id.
        strategy: The frontier selection strategy.
        planner: The planner whose clearance model committed paths are
            re-checked against. Must be the one `strategy` plans with, or the
            two disagree about which cells the body may occupy.
        max_wait_ticks: Deadlock escape threshold.
        target_tolerance_cells: Passed to `is_assignment_valid`.
        mode: "greedy" serves drones one at a time in descending id order;
            "global" allocates across the whole swarm at once. See
            `coordination.allocation`.
        free_threshold: Probability below which a cell is traversable.

    Returns:
        New states, keyed by drone id. A drone with nothing reachable left gets
        `assignment=None` — the caller reads an all-None result as "the mission
        can make no further progress".
    """
    prob = grid.probability()
    blocked = planner.clearance_mask(grid)
    # Frontier *cells*, not regions: a region's representative cell can shift as
    # the region changes shape, which would drop assignments that are still
    # perfectly good.
    live = frozenset(frontier_cells(grid, free_threshold=free_threshold))
    ordered = sorted(states, reverse=True)

    claimed: list[FrontierRegion] = []
    keeping: set[int] = set()
    for drone_id in ordered:
        state = states[drone_id]
        if is_assignment_valid(
            state,
            prob,
            free_threshold,
            max_wait_ticks,
            blocked,
            live,
            target_tolerance_cells,
        ):
            keeping.add(drone_id)
            assert state.assignment is not None  # narrowed by is_assignment_valid
            claimed.append(state.assignment.region)

    needy = [drone_id for drone_id in ordered if drone_id not in keeping]
    if mode == "global" and needy:
        return _assign_globally(
            grid, frontiers, states, strategy, planner, keeping, needy, claimed, blocked
        )

    result: dict[int, DroneState] = {}
    for drone_id in ordered:
        state = states[drone_id]
        if drone_id in keeping:
            result[drone_id] = state
            continue

        assignment = strategy.select(grid, frontiers, state.cell, claimed)
        if assignment is not None:
            claimed.append(assignment.region)
        result[drone_id] = replace(
            state, assignment=assignment, path_index=0, waited_ticks=0
        )

    return result


def _assign_globally(
    grid: OccupancyGrid,
    frontiers: Sequence[FrontierRegion],
    states: Mapping[int, DroneState],
    strategy: FrontierStrategy,
    planner: PathPlanner,
    keeping: set[int],
    needy: Sequence[int],
    claimed: list[FrontierRegion],
    blocked: NDArray[np.bool_],
) -> dict[int, DroneState]:
    """Allocate targets to every unassigned drone at once.

    One Dijkstra flood per needy drone gives its cost to every candidate, then
    `allocate` pairs drones to regions across the swarm rather than serving
    them one at a time. The chosen region is handed back to the strategy so the
    actual route, and any scoring the strategy applies, stay its business —
    this decides *who goes where*, not *how*.

    Args:
        grid: The occupancy grid.
        frontiers: All detected regions.
        states: Current drone states.
        strategy: The frontier selection strategy.
        planner: The planner whose clearance model the flood respects.
        keeping: Drones holding a still-valid assignment.
        needy: Drones requiring a target, in descending id order.
        claimed: Regions already spoken for by `keeping` drones.
        blocked: The planner's clearance mask.

    Returns:
        New states for every drone.
    """
    taken = {region.cell for region in claimed}
    candidates = [region for region in frontiers if region.cell not in taken]

    result: dict[int, DroneState] = {drone_id: states[drone_id] for drone_id in keeping}

    if candidates:
        fields = {
            drone_id: cost_field(grid, states[drone_id].cell, blocked)
            for drone_id in needy
        }
        chosen = allocate(list(needy), [r.cell for r in candidates], fields)
        by_cell = {region.cell: region for region in candidates}
    else:
        chosen = {}
        by_cell = {}

    for drone_id in needy:
        state = states[drone_id]
        target = chosen.get(drone_id)
        assignment = None
        if target is not None:
            # Ask the strategy for exactly this region, so routing and scoring
            # stay in one place rather than being duplicated here.
            assignment = strategy.select(grid, [by_cell[target]], state.cell, claimed)
            if assignment is not None:
                claimed.append(assignment.region)
        result[drone_id] = replace(
            state, assignment=assignment, path_index=0, waited_ticks=0
        )

    return result
