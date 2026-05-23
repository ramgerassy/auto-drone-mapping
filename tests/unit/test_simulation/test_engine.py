"""Tests for SimulationEngine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.simulation.engine import SimulationEngine

SCENE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "swarm_mapping"
    / "simulation"
    / "assets"
    / "small_indoor.xml"
)


@pytest.fixture
def engine() -> SimulationEngine:
    """Return a SimulationEngine loaded with the small indoor scene."""
    return SimulationEngine(SCENE_PATH, {0: "drone_0"})


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_loads_scene(self, engine: SimulationEngine) -> None:
        """Engine loads the scene without error."""
        assert engine.model is not None
        assert engine.data is not None

    def test_initial_drone_pose(self, engine: SimulationEngine) -> None:
        """Drone starts at the position defined in MJCF."""
        pose = engine.get_pose(0)
        np.testing.assert_allclose(pose.position, [0.0, 0.0, 1.0])

    def test_timestep(self, engine: SimulationEngine) -> None:
        """Timestep matches the MJCF option."""
        assert engine.timestep == pytest.approx(0.01)

    def test_step_advances_time(self, engine: SimulationEngine) -> None:
        """Stepping advances simulation time by one timestep."""
        t0 = engine.time
        engine.step()
        t1 = engine.time
        assert t1 == pytest.approx(t0 + engine.timestep)

    def test_reset_restores_initial_state(
        self, engine: SimulationEngine
    ) -> None:
        """Reset returns drone to start position and zeros time."""
        engine.set_drone_position(0, np.array([5.0, 5.0, 2.0]))
        engine.step()
        engine.step()

        engine.reset()

        pose = engine.get_pose(0)
        np.testing.assert_allclose(pose.position, [0.0, 0.0, 1.0])
        assert engine.time == pytest.approx(0.0)

    def test_set_drone_position(self, engine: SimulationEngine) -> None:
        """Teleporting a drone updates its pose."""
        target = np.array([3.0, -2.0, 1.5])
        engine.set_drone_position(0, target)

        pose = engine.get_pose(0)
        np.testing.assert_allclose(pose.position, target)

    def test_set_drone_position_unknown_id(
        self, engine: SimulationEngine
    ) -> None:
        """Setting position for unknown drone_id raises KeyError."""
        with pytest.raises(KeyError, match="Unknown drone_id"):
            engine.set_drone_position(99, np.array([0.0, 0.0, 1.0]))

    def test_cast_rays_from_drone(
        self, engine: SimulationEngine
    ) -> None:
        """Rays from drone hit walls at expected distances."""
        directions = np.array([
            [1.0, 0.0, 0.0],   # east wall at ~10m
            [-1.0, 0.0, 0.0],  # west wall at ~10m
            [0.0, 1.0, 0.0],   # north wall at ~10m
            [0.0, -1.0, 0.0],  # south wall at ~10m
        ])

        results = engine.cast_rays(0, directions)

        assert len(results) == 4
        for hit in results:
            assert hit is not None
            # Drone is at center, walls at ±10m with 0.1m thickness
            assert 8.0 < hit.distance < 11.0

    def test_determinism(self, engine: SimulationEngine) -> None:
        """Same initial state produces identical results after N steps."""
        n_steps = 50

        # First run
        engine.reset()
        for _ in range(n_steps):
            engine.step()
        pose_a = engine.get_pose(0)

        # Second run
        engine.reset()
        for _ in range(n_steps):
            engine.step()
        pose_b = engine.get_pose(0)

        np.testing.assert_array_equal(pose_a.position, pose_b.position)
        np.testing.assert_array_equal(pose_a.quaternion, pose_b.quaternion)
