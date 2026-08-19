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

import numpy as np
import pytest

from swarm_mapping.coordination.master import CentralizedMaster
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


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """Write the inline room to a file the engine can load."""
    path = tmp_path / "room.xml"
    path.write_text(ROOM_XML)
    return path


def build_master(
    scene: Path, starts: dict[int, tuple[float, float]]
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
            resolution=RESOLUTION,
            origin_x=-3.0,
            origin_y=-3.0,
            grid_width=24,
            grid_height=24,
        )
    )
    master = CentralizedMaster(
        engine=engine,
        sensor=sensor,
        mapper=mapper,
        strategy=NearestFrontier(AStarPlanner()),
        altitude=ALTITUDE,
        min_separation=MIN_SEPARATION,
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
