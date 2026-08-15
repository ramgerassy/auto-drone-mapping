"""Centralized mission orchestration.

`CentralizedMaster` drives the whole mission: it senses with every drone,
integrates the scans into one shared map, hands out frontiers, and steps each
drone one cell along its path — never letting two drones collide.

The tick runs in three complete phases rather than per-drone. If sensing and
moving interleaved, drone 1 would plan against a map already containing drone
5's newest scan while drone 5 planned against a staler one; separate phases give
every drone the same snapshot to reason about.

The two decisions with teeth — who gets which frontier, and who yields to whom —
live in `assignment.py` and `movement.py` as pure functions. This class is the
glue that talks to the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from swarm_mapping.coordination.assignment import assign_all, next_cell
from swarm_mapping.coordination.movement import resolve_moves
from swarm_mapping.coordination.types import Cell, DroneState
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.perception.protocols import Sensor
from swarm_mapping.planning.frontier_strategy import FrontierStrategy
from swarm_mapping.simulation.engine import SimulationEngine


class CentralizedMaster:
    """Single master orchestrating all drones over a synchronous tick loop.

    Args:
        engine: The simulation engine (drone poses and teleport commands).
        sensor: The sensor used to scan with each drone.
        mapper: The shared map every drone contributes to and plans against.
        strategy: How a drone picks its next frontier.
        altitude: Flight height in metres; drones stay in one horizontal plane.
        min_separation: Required centre-to-centre spacing between drones, in
            metres. Converted to cells internally — a cell is only
            `resolution` metres across, so "different cells" is not by itself
            enough separation for a drone body.
        max_wait_ticks: Consecutive blocked ticks after which a drone abandons
            its frontier and picks another, breaking head-on deadlocks.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        sensor: Sensor,
        mapper: Mapper,
        strategy: FrontierStrategy,
        altitude: float,
        min_separation: float,
        max_wait_ticks: int,
    ) -> None:
        self._engine = engine
        self._sensor = sensor
        self._mapper = mapper
        self._strategy = strategy
        self._altitude = altitude
        self._max_wait_ticks = max_wait_ticks

        grid = mapper.grid
        self._min_separation_cells = min_separation / grid.config.resolution

        # Descending id order: the higher id has right of way, and precedence
        # only holds if it also drives processing order.
        self._ordered_ids = sorted(engine.drone_ids, reverse=True)

        self._states: dict[int, DroneState] = {}
        for drone_id in self._ordered_ids:
            position = engine.get_pose(drone_id).position
            col, row = grid.world_to_grid(float(position[0]), float(position[1]))
            self._states[drone_id] = DroneState(
                drone_id=drone_id,
                cell=(col, row),
                assignment=None,
                path_index=0,
                waited_ticks=0,
            )

        self._complete = False
        self._tick_count = 0

    @property
    def is_complete(self) -> bool:
        """True once no drone could be assigned a reachable frontier."""
        return self._complete

    @property
    def drone_states(self) -> Mapping[int, DroneState]:
        """Read-only view of drone state, for visualization."""
        return self._states

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_count

    def tick(self) -> None:
        """Advance the mission by one tick: sense, assign, then move."""
        self._sense()
        self._assign()
        self._move()
        self._tick_count += 1

    def _sense(self) -> None:
        """Scan with every drone and fold the results into the shared map."""
        for drone_id in self._ordered_ids:
            self._mapper.integrate_scan(self._sensor.scan(drone_id))

    def _assign(self) -> None:
        """Give every drone that needs one a frontier to fly to."""
        self._states = assign_all(
            self._mapper.grid,
            self._mapper.get_frontiers(),
            self._states,
            self._strategy,
            self._max_wait_ticks,
        )
        # Nothing assignable to anyone means no drone can make progress —
        # either the map is fully explored or the rest is unreachable.
        self._complete = all(
            state.assignment is None for state in self._states.values()
        )

    def _move(self) -> None:
        """Step each drone one cell, yielding where two would collide."""
        current: dict[int, Cell] = {
            drone_id: state.cell for drone_id, state in self._states.items()
        }
        desired: dict[int, Cell | None] = {
            drone_id: next_cell(state) for drone_id, state in self._states.items()
        }
        final = resolve_moves(current, desired, self._min_separation_cells)

        for drone_id in self._ordered_ids:
            state = self._states[drone_id]
            cell = final[drone_id]
            if cell != state.cell:
                self._states[drone_id] = replace(
                    state,
                    cell=cell,
                    path_index=state.path_index + 1,
                    waited_ticks=0,
                )
                self._teleport(drone_id, cell)
            elif desired[drone_id] is not None:
                # Wanted to move but was blocked — count it toward the escape.
                self._states[drone_id] = replace(
                    state, waited_ticks=state.waited_ticks + 1
                )

    def _teleport(self, drone_id: int, cell: Cell) -> None:
        """Place a drone at a cell's centre, at the mission altitude."""
        world_x, world_y = self._mapper.grid.grid_to_world(*cell)
        self._engine.set_drone_position(
            drone_id, np.array([world_x, world_y, self._altitude], dtype=np.float64)
        )
