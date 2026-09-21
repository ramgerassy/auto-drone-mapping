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

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations

import numpy as np

from swarm_mapping.coordination.assignment import assign_all, next_cell
from swarm_mapping.coordination.movement import resolve_moves
from swarm_mapping.coordination.types import Cell, DroneHealth, DroneState
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.perception.protocols import Sensor
from swarm_mapping.planning.frontier_strategy import FrontierStrategy
from swarm_mapping.planning.path_planner import PathPlanner
from swarm_mapping.simulation.engine import DRONE_HALF_EXTENT, SimulationEngine

_LOGGER = logging.getLogger(__name__)

# `resolve_moves` is a RADIAL test, but the body is a square and
# `L_inf <= L_2` — so a radial threshold only clears two axis-aligned boxes at
# the circumscribed diameter. At dx = dy = 0.2125 the Euclidean distance is
# 0.3005, passing a 0.30 m threshold, while the Chebyshev distance is 0.2125
# and the boxes overlap by 0.0875 m. The floor is therefore the body diagonal,
# not the body width. (Break-even is 0.30/sqrt(2) = 0.21213; the widely-quoted
# 0.212 sits just *below* it and would be rejected.)
MIN_SEPARATION_FLOOR = 2.0 * DRONE_HALF_EXTENT * math.sqrt(2.0)

# Log-odds written under a wreck in the planning overlay: well past the 0.6
# band, so the planner treats it exactly like a confirmed obstacle.
_WRECK_LOG_ODDS = 2.0


class CentralizedMaster:
    """Single master orchestrating all drones over a synchronous tick loop.

    Args:
        engine: The simulation engine (drone poses and teleport commands).
        sensor: The sensor used to scan with each drone.
        mapper: The shared map every drone contributes to and plans against.
        strategy: How a drone picks its next frontier.
        planner: The planner `strategy` plans with. Held separately so a
            committed path can be re-checked against the same clearance model
            as the map changes — see `assignment.is_assignment_valid`.
        altitude: Flight height in metres; drones stay in one horizontal plane.
        min_separation: Required centre-to-centre spacing between drones, in
            metres. Converted to cells internally — a cell is only
            `resolution` metres across, so "different cells" is not by itself
            enough separation for a drone body.
        max_wait_ticks: Consecutive blocked ticks after which a drone abandons
            its frontier and picks another, breaking head-on deadlocks.
        no_progress_ticks: Consecutive ticks without a newly classified cell
            after which the mission stops, reported as blocked. "Every drone
            unassigned" is too strict a terminating condition on a large scene:
            wall-surface cells drift across the classification bands and keep
            emitting small frontier regions, a few transiently reachable, so a
            swarm with nothing left to find never stops on its own. 0 disables
            the check.
        return_to_base_ticks: Consecutive unassigned ticks after which an idle
            drone flies back to its start cell. 0 keeps it parked.
        assignment_mode: How frontiers are handed out — "greedy" or "global".
        target_tolerance_cells: How far a live frontier may drift from a
            drone's target before the target counts as gone.
        heartbeat_timeout_ticks: Consecutive missed heartbeats after which a
            drone is declared LOST. At least 1.
        stuck_timeout_ticks: Consecutive granted moves that did not happen
            after which a drone is declared STUCK. Waiting to yield never
            counts. At least 1.

    Raises:
        ValueError: If either failure-detection timeout is below 1, if the
            swarm is empty, if min_separation is not finite, is
            below the body diagonal, or spans under one cell, or if two drones
            start closer together than min_separation — in metres or, once
            snapped to cells, in cells.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        sensor: Sensor,
        mapper: Mapper,
        strategy: FrontierStrategy,
        planner: PathPlanner,
        altitude: float,
        min_separation: float,
        max_wait_ticks: int,
        no_progress_ticks: int = 0,
        return_to_base_ticks: int = 0,
        assignment_mode: str = "greedy",
        target_tolerance_cells: int = 0,
        heartbeat_timeout_ticks: int = 3,
        stuck_timeout_ticks: int = 3,
    ) -> None:
        self._engine = engine
        self._sensor = sensor
        self._mapper = mapper
        # Two references to what must be one object. If they diverge, the
        # strategy plans against one body model and committed paths are
        # re-checked against another — and the quiet direction is the harmful
        # one: a master holding a clearance of 0.0 gets an all-False mask and
        # never drops a path, silently restoring the bug the re-check exists to
        # prevent. Duck-typed rather than declared on the FrontierStrategy
        # Protocol, so a strategy that plans nothing is not forced to invent a
        # planner it does not have.
        strategy_planner = getattr(strategy, "planner", None)
        if strategy_planner is not None and strategy_planner is not planner:
            msg = (
                "planner must be the same object the strategy plans with; "
                "otherwise selection and path re-validation disagree about "
                "which cells the body may occupy, and a mismatch is silent."
            )
            raise ValueError(msg)

        self._strategy = strategy
        self._planner = planner
        self._altitude = altitude
        self._max_wait_ticks = max_wait_ticks
        self._no_progress_ticks = no_progress_ticks
        self._stalled_for = 0
        self._last_known = -1
        # Frontiers a drone flew all the way to and still could not
        # resolve. See `_assign`.
        self._exhausted: set[Cell] = set()
        self._return_ticks = return_to_base_ticks
        self._assignment_mode = assignment_mode
        self._target_tolerance = target_tolerance_cells
        self._idle_ticks: dict[int, int] = {}
        self._going_home: dict[int, list[Cell]] = {}
        for name, value in (
            ("heartbeat_timeout_ticks", heartbeat_timeout_ticks),
            ("stuck_timeout_ticks", stuck_timeout_ticks),
        ):
            if value < 1:
                msg = (
                    f"{name} must be at least 1, got {value}: failure "
                    "detection cannot be switched off"
                )
                raise ValueError(msg)
        self._heartbeat_timeout = heartbeat_timeout_ticks
        self._stuck_timeout = stuck_timeout_ticks
        # Failure evidence, per drone. Consecutive counts: one good heartbeat
        # or one realized move clears them.
        self._missed: dict[int, int] = {}
        self._unrealized: dict[int, int] = {}

        grid = mapper.grid
        # Cells a wreck's body overlaps — the same overlap rule AStarPlanner
        # uses for inflation. k = 1 at every resolution the scenarios use.
        self._wreck_radius = (
            math.ceil(DRONE_HALF_EXTENT / grid.config.resolution + 0.5) - 1
        )
        # Every guard below is a `<` comparison and every comparison against
        # NaN is False, so without this a NaN would pass all of them — and then
        # `threshold_sq` in resolve_moves is NaN, every `d2 < NaN` is False, and
        # no move is ever blocked. Collision avoidance would be silently off
        # while its own invariant test still passed.
        if not math.isfinite(min_separation):
            msg = f"min_separation must be a finite value, got {min_separation}"
            raise ValueError(msg)
        if min_separation < MIN_SEPARATION_FLOOR:
            msg = (
                f"min_separation {min_separation} m is below the body diagonal "
                f"{MIN_SEPARATION_FLOOR:.4f} m. The separation test is radial "
                "and the body is a square, so a smaller value permits diagonal "
                "overlap while appearing to pass."
            )
            raise ValueError(msg)

        self._min_separation_cells = min_separation / grid.config.resolution
        if self._min_separation_cells < 1.0:
            msg = (
                f"min_separation {min_separation} m is under one cell at "
                f"resolution {grid.config.resolution} m. Distinct cells are at "
                "least 1 apart, so a sub-cell radius degenerates to same-cell "
                "exclusion and stops separating distinct cells at all. A "
                "safety parameter that silently stops separating is worse than "
                "an absent one. It must span at least one cell."
            )
            raise ValueError(msg)

        # Descending id order: the higher id has right of way, and precedence
        # only holds if it also drives processing order.
        self._ordered_ids = sorted(engine.drone_ids, reverse=True)
        if not self._ordered_ids:
            # `all()` over no drones is True, so an empty swarm would report a
            # completed mission on tick 1 having mapped nothing. CLAUDE.md
            # scopes the system at 1-5 drones.
            msg = "no drones in the scene; the swarm must hold 1-5 drones"
            raise ValueError(msg)
        # Drones whose heartbeat arrived this tick, in descending id order.
        self._heard: list[int] = list(self._ordered_ids)

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

        # `resolve_moves` prevents new separation violations but never repairs
        # an existing one, so a pair that starts too close can never recover.
        # Checked in CELL space, not metres: `world_to_grid` floors, so each
        # drone can lose most of a cell when snapped, and both are teleported
        # to cell centres on the first move. A pair 0.43 m apart clears a
        # 0.4243 m metric threshold and still lands 0.25 m apart — two 0.30 m
        # bodies overlapping before the mission starts.
        for first, second in combinations(self._ordered_ids, 2):
            a_col, a_row = self._states[first].cell
            b_col, b_row = self._states[second].cell
            gap_cells = float(np.hypot(a_col - b_col, a_row - b_row))
            if gap_cells < self._min_separation_cells:
                gap_m = gap_cells * grid.config.resolution
                msg = (
                    f"drones {first} and {second} start {gap_cells:.2f} cells "
                    f"({gap_m:.3f} m) apart once snapped to the grid, closer "
                    f"than min_separation {min_separation} m "
                    f"({self._min_separation_cells:.2f} cells); they would "
                    "deadlock immediately."
                )
                raise ValueError(msg)

        # Where each drone started, and therefore where it returns to. Start
        # positions are validated clear of geometry at construction, so home is
        # always somewhere it can legally sit.
        self._home: dict[int, Cell] = {
            drone_id: state.cell for drone_id, state in self._states.items()
        }

        self._complete = False
        self._blocked = False
        self._unreachable_frontiers = 0
        self._tick_count = 0
        # `swarm_lost` is a one-shot event: a finished mission may still be
        # ticked, and must not report the loss again.
        self._swarm_lost_logged = False

    @property
    def is_complete(self) -> bool:
        """True once no drone could be assigned a reachable frontier.

        This is the *terminal* signal, not a success signal: it covers both
        "everything is mapped" and "nothing left is reachable". Check
        `is_blocked` to tell them apart before treating a finished run as a
        successful one.
        """
        return self._complete

    @property
    def is_blocked(self) -> bool:
        """True if the mission ended with frontiers still on the map.

        Reports that the swarm stopped with ground it could see but not reach —
        walled out by clearance, by an obstacle across the only route, or by a
        mis-set radius. Without it a run that mapped 58% of a room and a run
        that mapped all of it are indistinguishable from the outside.

        **Read it with `unreachable_frontiers` and coverage, not as pass/fail.**
        Measured on `small_indoor`: a complete single-drone run terminates with
        `is_blocked` True, 3 frontier regions left, and 98.1% coverage — every
        unmapped cell inside an obstacle's footprint or in the wall margin that
        `clearance_radius` refuses to enter. Inflation leaves wall-adjacent
        frontiers visible but unoccupiable, so a *successful* mission normally
        ends blocked. What separates that from a real failure is the magnitude:
        3 regions at 98% is residue, 200 regions at 58% is a wall.
        """
        return self._blocked

    @property
    def unreachable_frontiers(self) -> int:
        """Frontier regions still detected when the mission ended."""
        return self._unreachable_frontiers

    @property
    def drone_states(self) -> Mapping[int, DroneState]:
        """Read-only view of drone state, for visualization."""
        return self._states

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_count

    def tick(self) -> None:
        """Advance the mission one tick: observe, sense, assign, then move.

        Observing comes first so that a failure declared this tick releases its
        frontier before this tick's assignment pass hands frontiers out.
        """
        self._observe()
        self._sense()
        self._assign()
        self._move()
        self._tick_count += 1
        self._check_progress()

    def _check_progress(self) -> None:
        """Stop the mission once it has gone `no_progress_ticks` without a gain.

        Counts newly *classified* cells rather than visited ones: the mission's
        product is the map, so a tick that resolves nothing achieved nothing,
        whatever the drones did.
        """
        if self._no_progress_ticks <= 0:
            return
        prob = self._mapper.grid.probability()
        known = int(np.count_nonzero((prob < 0.4) | (prob > 0.6)))
        if known > self._last_known:
            self._last_known = known
            self._stalled_for = 0
            return
        self._stalled_for += 1
        if self._stalled_for >= self._no_progress_ticks:
            self._complete = True
            self._blocked = True
            self._unreachable_frontiers = len(self._mapper.get_frontiers())
            _LOGGER.warning(
                "mission_stalled",
                extra={
                    "tick": self._tick_count,
                    "no_progress_ticks": self._no_progress_ticks,
                    "unreachable_frontiers": self._unreachable_frontiers,
                },
            )

    def _observe(self) -> None:
        """Poll heartbeats, then declare any drone whose evidence crossed a timeout.

        The master learns of a failure only here, and only from what a ground
        station would see: silence, or commanded moves that did not happen. A
        LOST drone is no longer polled; a STUCK one still reports, and keeps
        being sensed.

        Until it is declared, a silent drone is still ACTIVE: within the
        ≤ `heartbeat_timeout_ticks` window, the assignment pass may hand it a
        new claim if its target evaporates. That claim is released on
        declaration like any other.
        """
        self._heard = []
        for drone_id in self._ordered_ids:
            health = self._states[drone_id].health
            if health is DroneHealth.LOST:
                continue
            if self._engine.heartbeat(drone_id):
                self._heard.append(drone_id)
                self._missed[drone_id] = 0
            elif health is DroneHealth.ACTIVE:
                self._missed[drone_id] = self._missed.get(drone_id, 0) + 1

        for drone_id in self._ordered_ids:
            if self._states[drone_id].health is not DroneHealth.ACTIVE:
                continue
            if self._missed.get(drone_id, 0) >= self._heartbeat_timeout:
                self._declare_failed(drone_id, DroneHealth.LOST)
            elif self._unrealized.get(drone_id, 0) >= self._stuck_timeout:
                self._declare_failed(drone_id, DroneHealth.STUCK)

    def _declare_failed(self, drone_id: int, health: DroneHealth) -> None:
        """Mark a drone failed for good and put its frontier back in the pool."""
        state = self._states[drone_id]
        released = state.assignment.region.cell if state.assignment else None
        self._states[drone_id] = replace(
            state, health=health, assignment=None, path_index=0, waited_ticks=0
        )
        self._going_home.pop(drone_id, None)
        self._idle_ticks.pop(drone_id, None)
        _LOGGER.warning(
            "drone_failed",
            extra={
                "drone_id": drone_id,
                "health": health.value,
                "tick": self._tick_count,
                "released": list(released) if released is not None else None,
            },
        )

    def _read_cell(self, drone_id: int) -> Cell:
        """Where the localizer says the drone is, snapped to the grid."""
        position = self._engine.get_pose(drone_id).position
        return self._mapper.grid.world_to_grid(float(position[0]), float(position[1]))

    def _sense(self) -> None:
        """Scan with every drone heard this tick and fold the results into the map.

        A drone that missed its heartbeat sent no telemetry, so there is no scan
        to integrate (F10-R3).
        """
        for drone_id in self._heard:
            self._mapper.integrate_scan(self._sensor.scan(drone_id))

    def _assign(self) -> None:
        """Give every drone that needs one a frontier to fly to."""
        frontiers = self._mapper.get_frontiers()

        # A frontier a drone reached without clearing cannot be cleared by
        # going there again. Wall-surface cells drift across the classification
        # bands and emit frontiers over space no scan can resolve, so without
        # this a drone shuttles between two such phantoms indefinitely —
        # visible in the viewer as a drone looping between the same two rooms.
        #
        # "Reached" is the right test rather than "targeted": arriving is what
        # proves the frontier is unresolvable from close range, which is the
        # only evidence available without ground truth.
        live = {region.cell for region in frontiers}
        for state in self._states.values():
            if (
                state.assignment is not None
                and next_cell(state) is None  # path exhausted: it arrived
                and state.assignment.region.cell in live
            ):
                self._exhausted.add(state.assignment.region.cell)
                _LOGGER.info(
                    "frontier_exhausted",
                    extra={
                        "drone_id": state.drone_id,
                        "cell": state.assignment.region.cell,
                    },
                )
        frontiers = [f for f in frontiers if f.cell not in self._exhausted]

        # Only ACTIVE drones are handed frontiers. Failed drones hold no
        # assignment, so they never reach the `_exhausted` loop above either.
        active = {
            d: s for d, s in self._states.items() if s.health is DroneHealth.ACTIVE
        }
        # The overlay also reaches `frontier_cells(grid)` inside `assign_all`,
        # the set a committed target must still be in to survive. That is
        # benign: frontier cells under a wreck's footprint are unreachable
        # anyway, so dropping them from the survival check loses nothing.
        assigned = assign_all(
            self._planning_grid(),
            frontiers,
            active,
            self._strategy,
            self._planner,
            self._max_wait_ticks,
            target_tolerance_cells=self._target_tolerance,
            mode=self._assignment_mode,
        )
        # Merge back into a copy of the full table: key order is descending id
        # and must stay that way, since `_move` and `resolve_moves` iterate it.
        merged = dict(self._states)
        merged.update(assigned)
        self._states = merged
        # Nothing assignable to anyone means no drone can make progress. The
        # frontier count is what separates the two reasons for that, and it is
        # already in hand — discarding it is what made a walled-out mission
        # indistinguishable from a finished one.
        self._complete = all(
            state.assignment is None for state in self._states.values()
        )
        self._update_idle_drones()
        self._unreachable_frontiers = len(frontiers) if self._complete else 0
        self._blocked = self._complete and bool(frontiers)
        if self._blocked:
            _LOGGER.warning(
                "mission_blocked",
                extra={
                    "tick": self._tick_count,
                    "unreachable_frontiers": self._unreachable_frontiers,
                },
            )
        # No ACTIVE drone left: `_complete` and `_blocked` already come out
        # right above, since every failed state holds `assignment=None`.
        if not active and not self._swarm_lost_logged:
            self._swarm_lost_logged = True
            _LOGGER.warning(
                "swarm_lost",
                extra={
                    "tick": self._tick_count,
                    "unreachable_frontiers": len(frontiers),
                },
            )

    def _planning_grid(self) -> OccupancyGrid:
        """The grid to plan on: the map, plus every wreck as an obstacle.

        A failed drone is a physical body the map deliberately does not
        contain — the teammate filter keeps drones out of it, and the map's
        accuracy KPI depends on that. So wrecks go into a copy used for
        planning only; nothing written here reaches the exported map.
        """
        grid = self._mapper.grid
        wrecks = [
            self._states[d].cell
            for d in self._ordered_ids
            if self._states[d].health is not DroneHealth.ACTIVE
        ]
        if not wrecks:
            return grid
        overlay = OccupancyGrid(grid.config)
        overlay.log_odds[:] = grid.log_odds
        k = self._wreck_radius
        for col, row in wrecks:
            overlay.log_odds[
                max(0, row - k) : row + k + 1, max(0, col - k) : col + k + 1
            ] = _WRECK_LOG_ODDS
        return overlay

    def _update_idle_drones(self) -> None:
        """Send a drone home once it has sat unassigned for long enough.

        Only drones with nothing to do are moved, so this never competes with
        exploration: a drone that picks up a frontier on any tick abandons the
        trip home immediately.
        """
        for drone_id, state in self._states.items():
            if state.health is not DroneHealth.ACTIVE:
                continue
            if state.assignment is not None:
                self._idle_ticks[drone_id] = 0
                self._going_home.pop(drone_id, None)
                continue

            idle = self._idle_ticks.get(drone_id, 0) + 1
            self._idle_ticks[drone_id] = idle
            home = self._home[drone_id]
            if (
                self._return_ticks <= 0
                or idle < self._return_ticks
                or drone_id in self._going_home
                or state.cell == home
            ):
                continue

            path = self._planner.plan(self._planning_grid(), state.cell, home)
            if path is not None:
                self._going_home[drone_id] = path[1:]
                _LOGGER.info(
                    "returning_to_base",
                    extra={"drone_id": drone_id, "idle_ticks": idle},
                )

    def _move(self) -> None:
        """Step each drone one cell, yielding where two would collide."""
        current: dict[int, Cell] = {
            drone_id: state.cell for drone_id, state in self._states.items()
        }
        # Only ACTIVE drones that were heard this tick get a desired cell.
        # Everyone else holds station but stays in `current`, so `resolve_moves`
        # keeps teammates clear of them as static bodies.
        heard = set(self._heard)  # membership only; never iterated
        desired: dict[int, Cell | None] = {
            drone_id: (
                next_cell(state)
                if state.health is DroneHealth.ACTIVE and drone_id in heard
                else None
            )
            for drone_id, state in self._states.items()
        }
        # An idle drone on its way home has no assignment, so `next_cell` gives
        # it nothing; its route lives here instead.
        for drone_id, route in self._going_home.items():
            if route and drone_id in heard:
                desired[drone_id] = route[0]
        final = resolve_moves(current, desired, self._min_separation_cells)

        for drone_id in self._ordered_ids:
            state = self._states[drone_id]
            cell = final[drone_id]
            if cell != state.cell:
                self._teleport(drone_id, cell)
                actual = self._read_cell(drone_id)
                if actual != cell:
                    # Granted and commanded, but the drone did not arrive. The
                    # state follows the localizer, the path does not advance,
                    # and `_observe` weighs the count next tick.
                    self._unrealized[drone_id] = self._unrealized.get(drone_id, 0) + 1
                    self._states[drone_id] = replace(state, cell=actual)
                    continue
                self._unrealized[drone_id] = 0
                self._states[drone_id] = replace(
                    state,
                    cell=cell,
                    path_index=state.path_index + 1,
                    waited_ticks=0,
                )
                homeward = self._going_home.get(drone_id)
                if homeward and homeward[0] == cell:
                    homeward.pop(0)
                    if not homeward:
                        self._going_home.pop(drone_id, None)
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
