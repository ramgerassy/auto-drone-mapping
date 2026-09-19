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
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.coordination.types import Cell, DroneState
from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.planning.frontier_strategy import FrontierStrategy
from swarm_mapping.planning.path_planner import PathPlanner

_LOGGER = logging.getLogger(__name__)


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

    Returns:
        True if the assignment should be kept.
    """
    if state.assignment is None:
        return False
    if state.waited_ticks > max_wait_ticks:
        return False  # deadlock escape — try a different frontier
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
        free_threshold: Probability below which a cell is traversable.

    Returns:
        New states, keyed by drone id. A drone with nothing reachable left gets
        `assignment=None` — the caller reads an all-None result as "the mission
        can make no further progress".
    """
    prob = grid.probability()
    blocked = planner.clearance_mask(grid)
    ordered = sorted(states, reverse=True)

    claimed: list[FrontierRegion] = []
    keeping: set[int] = set()
    for drone_id in ordered:
        state = states[drone_id]
        if is_assignment_valid(state, prob, free_threshold, max_wait_ticks, blocked):
            keeping.add(drone_id)
            assert state.assignment is not None  # narrowed by is_assignment_valid
            claimed.append(state.assignment.region)

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
