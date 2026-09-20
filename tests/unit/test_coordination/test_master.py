"""Tests for the centralized master (coordination.master).

Uses a tiny inline MJCF room — never a MuJoCo mock, per CLAUDE.md. The scene
carries no drone bodies; SimulationEngine injects them.

Map accuracy is asserted since Feature 4b landed: with the perception teammate
filter in place, drones no longer range each other into the map. The assertion
is on the height layer — see TestMapAccuracy for why occupancy is not the
discriminating signal over a full mission.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_protocol_members

import numpy as np
import pytest

from swarm_mapping.coordination.master import (
    MIN_SEPARATION_FLOOR,
    CentralizedMaster,
)
from swarm_mapping.coordination.protocols import Coordinator
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.planning.frontier_strategy import NearestFrontier
from swarm_mapping.planning.path_planner import AStarPlanner
from swarm_mapping.simulation.engine import SimulationEngine

pytestmark = pytest.mark.sprint(2)  # Feature 4 — centralized master

# A 6m x 6m empty room. No drone bodies — the engine injects them.
ROOM_XML = """\
<mujoco model="coord_test_room">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <geom name="wall_east" type="box" pos="3 0 1" size="0.1 3 1"/>
    <geom name="wall_west" type="box" pos="-3 0 1" size="0.1 3 1"/>
    <geom name="wall_north" type="box" pos="0 3 1" size="3 0.1 1"/>
    <geom name="wall_south" type="box" pos="0 -3 1" size="3 0.1 1"/>
  </worldbody>
</mujoco>
"""

ALTITUDE = 1.0
MIN_SEPARATION = 0.5  # metres
MAX_WAIT = 5
RESOLUTION = 0.25
# Body half-extent 0.15 plus a 5 cm margin; at 0.25 m cells that is r = 1.
CLEARANCE = 0.20


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """Write the inline room to a file the engine can load."""
    path = tmp_path / "room.xml"
    path.write_text(ROOM_XML)
    return path


def build_master(
    scene: Path,
    starts: dict[int, tuple[float, float]],
    min_separation: float = MIN_SEPARATION,
    resolution: float = RESOLUTION,
) -> tuple[CentralizedMaster, SimulationEngine, Mapper]:
    """Wire an engine, sensor, mapper and master over the tiny room."""
    positions = {
        drone_id: np.array([x, y, ALTITUDE], dtype=np.float64)
        for drone_id, (x, y) in starts.items()
    }
    engine = SimulationEngine(scene, positions)
    sensor = Rangefinder(engine, num_rays=36, max_range=8.0)
    mapper = Mapper(
        MapConfig(
            resolution=resolution,
            origin_x=-3.0,
            origin_y=-3.0,
            grid_width=int(6 / resolution),
            grid_height=int(6 / resolution),
        )
    )
    # One planner instance: the master re-checks committed paths against the
    # same clearance model the strategy plans with.
    planner = AStarPlanner(clearance_radius=CLEARANCE)
    master = CentralizedMaster(
        engine=engine,
        sensor=sensor,
        mapper=mapper,
        strategy=NearestFrontier(planner),
        planner=planner,
        altitude=ALTITUDE,
        min_separation=min_separation,
        max_wait_ticks=MAX_WAIT,
    )
    return master, engine, mapper


def known_cells(mapper: Mapper) -> int:
    """Count cells that are no longer unknown."""
    prob = mapper.grid.probability()
    return int(np.sum((prob < 0.4) | (prob > 0.6)))


def run_mission(master: CentralizedMaster, max_ticks: int = 400) -> int:
    """Tick until the mission completes or the cap is hit; return tick count."""
    while not master.is_complete and master.tick_count < max_ticks:
        master.tick()
    return master.tick_count


class TestTick:
    """One tick of the loop."""

    @pytest.mark.sanity
    def test_tick_senses_and_maps(self, scene: Path) -> None:
        """A single tick turns unknown cells into known ones."""
        master, _, mapper = build_master(scene, {0: (0.0, 0.0)})

        assert known_cells(mapper) == 0
        master.tick()
        assert known_cells(mapper) > 0

    def test_tick_advances_the_counter(self, scene: Path) -> None:
        """Each tick counts once."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})

        master.tick()
        master.tick()

        assert master.tick_count == 2

    def test_drone_moves_along_its_path(self, scene: Path) -> None:
        """After a tick the drone has stepped toward a frontier."""
        master, engine, _ = build_master(scene, {0: (0.0, 0.0)})
        start = master.drone_states[0].cell

        master.tick()

        assert master.drone_states[0].cell != start
        # The engine was actually commanded, not just the bookkeeping updated.
        pose = engine.get_pose(0)
        assert pose.position[2] == pytest.approx(ALTITUDE)

    def test_states_expose_every_drone(self, scene: Path) -> None:
        """drone_states covers the whole swarm."""
        master, _, _ = build_master(scene, {0: (-1.0, 0.0), 1: (1.0, 0.0)})

        assert set(master.drone_states) == {0, 1}


class TestMission:
    """Running to completion."""

    def test_not_complete_before_running(self, scene: Path) -> None:
        """A fresh master has work to do."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})
        assert not master.is_complete

    def test_mission_completes(self, scene: Path) -> None:
        """An enclosed room is fully explored and the mission ends."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})

        run_mission(master)

        assert master.is_complete

    def test_mission_maps_the_room(self, scene: Path) -> None:
        """Completion means most of the room is known, not that it gave up."""
        master, _, mapper = build_master(scene, {0: (0.0, 0.0)})

        run_mission(master)

        total = mapper.grid.probability().size
        assert known_cells(mapper) > 0.8 * total

    def test_idle_drones_have_no_assignment_at_completion(self, scene: Path) -> None:
        """Completion is defined by nothing being assignable."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})

        run_mission(master)

        assert all(s.assignment is None for s in master.drone_states.values())


class TestSeparationInvariant:
    """The 'zero collisions' KPI, as an assertion."""

    def test_drones_never_come_closer_than_min_separation(self, scene: Path) -> None:
        """No tick ends with two drones inside the separation radius."""
        master, _, mapper = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0), 2: (0.0, 1.5)}
        )
        grid = mapper.grid

        while not master.is_complete and master.tick_count < 300:
            master.tick()
            positions = [
                grid.grid_to_world(*s.cell) for s in master.drone_states.values()
            ]
            for i, (ax, ay) in enumerate(positions):
                for bx, by in positions[i + 1 :]:
                    distance = float(np.hypot(ax - bx, ay - by))
                    assert distance >= MIN_SEPARATION, (
                        f"separation violated at tick {master.tick_count}: "
                        f"{distance:.3f} < {MIN_SEPARATION}"
                    )

    def test_multi_drone_mission_completes(self, scene: Path) -> None:
        """Three drones still finish — waiting must not deadlock the mission."""
        master, _, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0), 2: (0.0, 1.5)}
        )

        run_mission(master, max_ticks=300)

        assert master.is_complete


class TestMapAccuracy:
    """The swarm maps the room, not itself."""

    def test_height_layer_has_no_phantom_heights(self, scene: Path) -> None:
        """No floor cell in the empty interior carries a drone's altitude.

        This is the mission-level teammate-filter assertion, and deliberately
        the *height* layer rather than occupancy. Measured on this scene with
        the filter disabled: 47 poisoned height cells but **zero** phantom
        occupied cells — a false occupied reading needs only 2-3 later free
        observations to wash out, and over a 20-tick mission every drone cell
        gets them. `update_occupied` does `height = max(height, hit_z)`, which
        is monotonic, so the height layer has no such recovery path and a drone
        written in once stays for the whole mission.

        Single-scan occupancy — where the artifact is visible before it washes
        out — is covered in tests/integration/test_teammate_filter.py.
        """
        master, _, mapper = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0), 2: (0.0, 1.5)}
        )
        run_mission(master, max_ticks=300)

        grid = mapper.grid
        poisoned = []
        for col in range(grid.config.grid_width):
            for row in range(grid.config.grid_height):
                world_x, world_y = grid.grid_to_world(col, row)
                # The walls' inner faces are at +-2.9, so this is empty space.
                interior = abs(world_x) < 2.8 and abs(world_y) < 2.8
                if interior and grid.height[row, col] != -np.inf:
                    poisoned.append((col, row))

        assert poisoned == []


class TestSeparationGuards:
    """Feature 4c — a mis-set separation must fail loudly, not silently.

    `resolve_moves` is a radial test but the body is a square, and
    `L_inf <= L_2`, so a radial threshold only clears two axis-aligned boxes at
    the CIRCUMSCRIBED diameter. At dx = dy = 0.2125 the Euclidean distance is
    0.3005 - passing a 0.30 m threshold - while the Chebyshev distance is
    0.2125 and the boxes overlap by 0.0875 m. The floor is therefore
    0.30 * sqrt(2) ~= 0.4243, not 0.30. See MIN_SEPARATION_FLOOR for the
    derivation; do not restate it here.
    """

    def test_separation_below_the_body_diagonal_is_rejected(self, scene: Path) -> None:
        """Case 9: 0.35 m looks generous against a 0.30 m body and is not."""
        with pytest.raises(ValueError, match="body diagonal"):
            build_master(scene, {0: (-1.0, 0.0), 1: (1.0, 0.0)}, min_separation=0.35)

    def test_separation_at_the_floor_is_accepted(self, scene: Path) -> None:
        """Case 10: exactly at the floor is legal — the boxes touch, no overlap.

        Uses MIN_SEPARATION_FLOOR itself rather than a rounded literal, so a
        `<` silently becoming `<=` fails here. The rounded 0.4243 sat 3.6e-05
        above the true floor and left that mutation alive.
        """
        master, _, _ = build_master(
            scene, {0: (-1.0, 0.0), 1: (1.0, 0.0)}, min_separation=MIN_SEPARATION_FLOOR
        )

        assert set(master.drone_states) == {0, 1}

    def test_start_positions_closer_than_separation_are_rejected(
        self, scene: Path
    ) -> None:
        """Case 11: `resolve_moves` prevents new violations, never repairs one.

        Two drones spawned inside each other's disc find every target blocked,
        burn through `max_wait_ticks`, re-select, and stay frozen — a permanent
        deadlock that comes from config alone and looks like a hang.
        """
        with pytest.raises(ValueError, match="deadlock immediately"):
            build_master(scene, {0: (0.0, 0.0), 1: (0.3, 0.0)})

    def test_separation_below_one_cell_is_rejected(self, scene: Path) -> None:
        """Case 12: below a cell the radial check is dead code.

        `resolve_moves` already seeds reservations with every drone's current cell,
        so a sub-cell radius can never block anything that seeding does not.
        Silently dead safety parameters are worse than absent ones.
        """
        with pytest.raises(ValueError, match="at least one cell"):
            build_master(scene, {0: (-2.0, 0.0), 1: (2.0, 0.0)}, resolution=1.0)


# A room split by a divider with a doorway. Geometry is aligned to whole cells:
# divider_n spans y in [0.75, 3.0], divider_s spans y in [-3.0, -0.50], leaving
# a 1.25 m (5-cell) gap. The narrow variant leaves 0.75 m (3 cells).
def doorway_xml(gap_south: float, gap_north: float) -> str:
    """A 6x6 room split at x=0 by a divider with a gap of the given extent."""
    south_half = (gap_south + 3.0) / 2.0
    north_half = (3.0 - gap_north) / 2.0
    return f"""\
<mujoco model="doorway">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <geom name="wall_east" type="box" pos="3 0 1" size="0.1 3 1"/>
    <geom name="wall_west" type="box" pos="-3 0 1" size="0.1 3 1"/>
    <geom name="wall_north" type="box" pos="0 3 1" size="3 0.1 1"/>
    <geom name="wall_south" type="box" pos="0 -3 1" size="3 0.1 1"/>
    <geom name="divider_n" type="box" pos="0 {gap_north + north_half} 1"
          size="0.25 {north_half} 1"/>
    <geom name="divider_s" type="box" pos="0 {gap_south - south_half} 1"
          size="0.25 {south_half} 1"/>
  </worldbody>
</mujoco>
"""


class TestGuardBoundaries:
    """Each guard pinned at the exact value it accepts.

    Every guard is a `<`, and every test sat strictly inside its rejection
    region — so flipping any of the three to `<=` passed the whole suite.
    """

    def test_separation_of_exactly_one_cell_is_accepted(self, scene: Path) -> None:
        """The one-cell guard's boundary: 0.5 m at 0.5 m cells is 1.00 cells."""
        master, _, _ = build_master(
            scene, {0: (-1.0, 0.0), 1: (1.0, 0.0)}, min_separation=0.5, resolution=0.5
        )

        assert set(master.drone_states) == {0, 1}

    def test_start_positions_exactly_at_separation_are_accepted(
        self, scene: Path
    ) -> None:
        """Two cells apart at 0.25 m cells clears a 0.5 m requirement exactly."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0), 1: (0.5, 0.0)})

        assert set(master.drone_states) == {0, 1}

    def test_non_finite_separation_is_rejected(self, scene: Path) -> None:
        """NaN passes every `<` comparison, then disables collision avoidance.

        `threshold_sq` becomes NaN in `resolve_moves`, every `d2 < NaN` is
        False, and no move is ever blocked — while
        `test_drones_never_come_closer_than_min_separation` keeps passing.
        """
        with pytest.raises(ValueError, match="finite"):
            build_master(
                scene, {0: (-1.0, 0.0), 1: (1.0, 0.0)}, min_separation=float("nan")
            )

    def test_empty_swarm_is_rejected(self, scene: Path) -> None:
        """`all()` over no drones is True, so an empty swarm reported success.

        One tick, nothing mapped, `is_complete` True and no error anywhere.
        """
        with pytest.raises(ValueError, match="no drones"):
            build_master(scene, {})


class TestConstrainedPassage:
    """Feature 4c — clearance decides which gaps the body may fly through.

    The mission-level counterpart to the planner's corridor-width tests. A
    "body never overlaps a wall" assertion was written first and discarded: at
    `resolution` 0.25 a 0.30 m body overhangs its own cell by 0.025 m no matter
    where it sits, and an occupied *cell* is up to half a cell larger than the
    wall inside it, so such an assertion measures grid discretization rather
    than physical overlap. See docs/progress.md.
    """

    def run_doorway(
        self, tmp_path: Path, gap_south: float, gap_north: float
    ) -> tuple[bool, float, bool]:
        """Fly one drone from the west room; report crossed, known, blocked."""
        path = tmp_path / f"door_{gap_south}_{gap_north}.xml"
        path.write_text(doorway_xml(gap_south, gap_north))

        engine = SimulationEngine(path, {0: np.array([-2.0, 0.0, ALTITUDE])})
        mapper = Mapper(
            MapConfig(
                resolution=RESOLUTION,
                origin_x=-3.0,
                origin_y=-3.0,
                grid_width=24,
                grid_height=24,
            )
        )
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        master = CentralizedMaster(
            engine=engine,
            sensor=Rangefinder(engine, num_rays=72, max_range=8.0),
            mapper=mapper,
            strategy=NearestFrontier(planner),
            planner=planner,
            altitude=ALTITUDE,
            min_separation=MIN_SEPARATION,
            max_wait_ticks=MAX_WAIT,
        )

        crossed = False
        while not master.is_complete and master.tick_count < 200:
            master.tick()
            for state in master.drone_states.values():
                if mapper.grid.grid_to_world(*state.cell)[0] > 0.5:
                    crossed = True

        prob = mapper.grid.probability()
        known = float(np.sum((prob < 0.4) | (prob > 0.6))) / prob.size
        return crossed, known, master.is_blocked

    def test_a_wide_doorway_is_flown_through(self, tmp_path: Path) -> None:
        """Case 13 (liveness): inflation must not wall off reachable space.

        A 1.25 m gap is five cells, comfortably over the `2r + 1 = 3` the body
        needs. The drone crosses and maps both rooms — the check that clearance
        has not made the environment unexplorable, which is how "inflate
        unknown cells too" or an oversized radius would fail.
        """
        crossed, known, _ = self.run_doorway(tmp_path, gap_south=-0.5, gap_north=0.75)

        assert crossed
        assert known > 0.9
        # Deliberately NOT asserting `not blocked`. `is_blocked` fires on
        # successful runs too: clearance inflation leaves wall-adjacent
        # frontiers permanently visible but unoccupiable, so a fully explored
        # room still ends with frontiers outstanding. What separates a finished
        # mission from a walled-out one is magnitude, not the flag.

    def test_a_doorway_the_body_cannot_fit_is_refused(self, tmp_path: Path) -> None:
        """The width rule, pinned — and the tax Feature 5's MJCF must pay.

        A 0.75 m gap is three cells, which is `2r + 1` exactly and looks
        sufficient. It is not: the divider's face lands on a cell boundary and
        the hit point is attributed to the cell above it, so one row of the gap
        is mapped as occupied and only two remain. The usable width must clear
        `2r + 1` cells **plus a cell of discretization slop**.

        Refusing to cross is correct behaviour — before 4c the drone flew
        through with its body inside the jamb.
        """
        crossed, known, _ = self.run_doorway(tmp_path, gap_south=-0.25, gap_north=0.5)

        assert not crossed
        # Without this, the test passes HARDER as the radius grows: a clearance
        # so large that nothing moves at all also fails to cross.
        assert known > 0.5, "the drone should still have mapped its own room"


class TestDeterminism:
    """Same setup, same mission."""

    def test_identical_runs_produce_identical_trajectories(self, scene: Path) -> None:
        """Two independent runs agree tick for tick."""
        starts = {0: (-1.0, 0.0), 1: (1.0, 0.0)}

        trajectories = []
        for _ in range(2):
            master, _, _ = build_master(scene, starts)
            path: list[tuple[int, tuple[int, int]]] = []
            for _ in range(25):
                master.tick()
                for drone_id in sorted(master.drone_states):
                    path.append((drone_id, master.drone_states[drone_id].cell))
            trajectories.append(path)

        assert trajectories[0] == trajectories[1]


class TestProtocol:
    """CentralizedMaster satisfies the Coordinator protocol."""

    def test_master_is_a_coordinator(self, scene: Path) -> None:
        """CentralizedMaster is usable through the Coordinator interface."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})
        coordinator: Coordinator = master

        assert callable(coordinator.tick)
        assert coordinator.is_complete is False
        assert set(coordinator.drone_states) == {0}

    def test_coordinator_requires_outcome_not_just_termination(self) -> None:
        """Any coordinator must report *whether* it finished, not just *that*.

        Asserted against the Protocol's declared members rather than against a
        master instance: `CentralizedMaster` already has both properties, so
        reading them through a `Coordinator`-annotated name would pass whether
        or not the seam actually requires them. This is the assertion that
        fails if someone narrows the Protocol back.
        """
        members = get_protocol_members(Coordinator)

        assert "is_blocked" in members
        assert "unreachable_frontiers" in members

    def test_coordinator_outcome_is_readable_through_the_seam(
        self, scene: Path
    ) -> None:
        """The outcome properties answer through a Coordinator-typed name."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0)})
        coordinator: Coordinator = master

        assert coordinator.is_blocked is False
        assert coordinator.unreachable_frontiers == 0


class TestPlannerConsistency:
    """The master and the strategy must share one body model."""

    def test_a_different_planner_than_the_strategy_is_rejected(
        self, scene: Path
    ) -> None:
        """The silent direction is the dangerous one.

        A master holding `clearance_radius=0.0` gets an all-False clearance
        mask, never drops a committed path, and silently restores the bug the
        re-check exists to prevent — with every other test still green, since
        they build both references from one variable.
        """
        positions = {0: np.array([0.0, 0.0, ALTITUDE])}
        engine = SimulationEngine(scene, positions)
        mapper = Mapper(
            MapConfig(
                resolution=RESOLUTION,
                origin_x=-3.0,
                origin_y=-3.0,
                grid_width=24,
                grid_height=24,
            )
        )

        with pytest.raises(ValueError, match="same object the strategy plans with"):
            CentralizedMaster(
                engine=engine,
                sensor=Rangefinder(engine, num_rays=36, max_range=8.0),
                mapper=mapper,
                strategy=NearestFrontier(AStarPlanner(clearance_radius=CLEARANCE)),
                planner=AStarPlanner(clearance_radius=0.0),  # a different object
                altitude=ALTITUDE,
                min_separation=MIN_SEPARATION,
                max_wait_ticks=MAX_WAIT,
            )
