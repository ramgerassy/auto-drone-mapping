"""Data types for the coordination module."""

from __future__ import annotations

from dataclasses import dataclass

from swarm_mapping.planning.frontier_strategy import FrontierAssignment

Cell = tuple[int, int]


@dataclass(frozen=True)
class DroneState:
    """One drone's exploration state, as of the current tick.

    Frozen: the master replaces states rather than mutating them, so a state
    handed to `visualization` cannot be changed underneath it.

    Attributes:
        drone_id: Integer identifier for the drone.
        cell: The drone's current (col, row) grid cell.
        assignment: The frontier it is flying to, or None when idle.
        path_index: Index of `cell` within `assignment.path`. The next step is
            `path[path_index + 1]`; equality with the last index means arrived.
        waited_ticks: Consecutive ticks this drone has been blocked by another
            drone. Reset on any successful move; drives the deadlock escape.
    """

    drone_id: int
    cell: Cell
    assignment: FrontierAssignment | None
    path_index: int
    waited_ticks: int
