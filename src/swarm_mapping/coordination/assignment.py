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

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.coordination.types import Cell, DroneState
from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.planning.frontier_strategy import FrontierStrategy


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
) -> bool:
    """Check whether a drone should keep flying its current assignment.

    An assignment is dropped — forcing re-selection next pass — when the drone
    has arrived, when the next cell turned out to be occupied (something was
    discovered there since the path was planned), or when the drone has been
    blocked long enough to count as deadlocked.

    Paths are otherwise kept rather than re-planned every tick: A* routes only
    through known-free cells, so a path stays valid unless a cell it crosses
    stops being free, which is exactly the check below.

    Args:
        state: The drone's current state.
        prob: The grid's occupancy probability array, indexed [row, col].
        free_threshold: Probability below which a cell is traversable.
        max_wait_ticks: Consecutive blocked ticks after which the drone gives up
            on this frontier and picks another (the deadlock escape).

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
    return bool(prob[row, col] < free_threshold)


def assign_all(
    grid: OccupancyGrid,
    frontiers: Sequence[FrontierRegion],
    states: Mapping[int, DroneState],
    strategy: FrontierStrategy,
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
        max_wait_ticks: Deadlock escape threshold.
        free_threshold: Probability below which a cell is traversable.

    Returns:
        New states, keyed by drone id. A drone with nothing reachable left gets
        `assignment=None` — the caller reads an all-None result as "the mission
        can make no further progress".
    """
    prob = grid.probability()
    ordered = sorted(states, reverse=True)

    claimed: list[FrontierRegion] = []
    keeping: set[int] = set()
    for drone_id in ordered:
        state = states[drone_id]
        if is_assignment_valid(state, prob, free_threshold, max_wait_ticks):
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
