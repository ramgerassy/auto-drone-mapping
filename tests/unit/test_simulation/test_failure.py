"""Tests for drone failure in SimulationEngine.

A failed drone ignores motion commands in both modes; the modes differ only in
what the drone still *reports*. That difference is the whole reason the
coordinator needs two detectors in Feature 10, so it is what these pin down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.types import FailureMode

pytestmark = pytest.mark.sprint(3)

SCENE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "swarm_mapping"
    / "simulation"
    / "assets"
    / "small_indoor.xml"
)
START = {0: (0.0, 0.0, 1.0), 1: (2.0, 0.0, 1.0)}


@pytest.fixture
def engine() -> SimulationEngine:
    """Two healthy drones in the small indoor room."""
    return SimulationEngine(
        SCENE_PATH, {d: np.array(p, dtype=np.float64) for d, p in START.items()}
    )


def command(
    engine: SimulationEngine, drone_id: int, target: tuple[float, float, float]
) -> bool:
    """Command a move and report whether the drone actually got there."""
    goal = np.array(target, dtype=np.float64)
    engine.set_drone_position(drone_id, goal)
    return bool(np.allclose(engine.get_pose(drone_id).position, goal))


class TestHealthyDrone:
    """The baseline both failure modes are measured against."""

    def test_reports_a_heartbeat(self, engine: SimulationEngine) -> None:
        """A healthy drone's heartbeat is True."""
        assert engine.heartbeat(0)

    def test_moves_when_commanded(self, engine: SimulationEngine) -> None:
        """A healthy drone reaches the commanded position."""
        assert command(engine, 0, (0.5, 0.0, 1.0))


class TestStuckDrone:
    """Motor fault: still talking, no longer moving."""

    def test_keeps_reporting(self, engine: SimulationEngine) -> None:
        """A stuck drone still reports a heartbeat."""
        engine.fail_drone(0, FailureMode.STUCK)
        assert engine.heartbeat(0)

    def test_ignores_motion_commands(self, engine: SimulationEngine) -> None:
        """A stuck drone does not move when commanded."""
        engine.fail_drone(0, FailureMode.STUCK)
        assert not command(engine, 0, (0.5, 0.0, 1.0))
        assert np.allclose(engine.get_pose(0).position, START[0])

    def test_its_sensor_still_works(self, engine: SimulationEngine) -> None:
        """A stuck drone still scans — its reports stay useful to the map."""
        engine.fail_drone(0, FailureMode.STUCK)
        hits = engine.cast_rays(0, np.array([[1.0, 0.0, 0.0]]))
        assert hits[0] is not None


class TestSilentDrone:
    """Crash or comms loss: nothing comes back."""

    def test_stops_reporting(self, engine: SimulationEngine) -> None:
        """A silently failed drone's heartbeat is False."""
        engine.fail_drone(0, FailureMode.SILENT)
        assert not engine.heartbeat(0)

    def test_ignores_motion_commands(self, engine: SimulationEngine) -> None:
        """A silently failed drone does not move when commanded."""
        engine.fail_drone(0, FailureMode.SILENT)
        assert not command(engine, 0, (0.5, 0.0, 1.0))


class TestIsolation:
    """Failing one drone must not affect another."""

    def test_failing_one_drone_leaves_the_other_untouched(
        self, engine: SimulationEngine
    ) -> None:
        """A healthy drone keeps working after its teammate fails."""
        engine.fail_drone(0, FailureMode.SILENT)
        assert engine.heartbeat(1)
        assert command(engine, 1, (2.5, 0.0, 1.0))


class TestMisuse:
    """Failure API misuse is rejected, not silently accepted."""

    def test_an_unknown_drone_is_rejected(self, engine: SimulationEngine) -> None:
        """An unrecognized drone_id raises KeyError."""
        with pytest.raises(KeyError):
            engine.fail_drone(7, FailureMode.SILENT)
        with pytest.raises(KeyError):
            engine.heartbeat(7)

    def test_a_drone_fails_once(self, engine: SimulationEngine) -> None:
        """Failure is permanent (sprint plan, Locked decision 2)."""
        engine.fail_drone(0, FailureMode.STUCK)
        with pytest.raises(ValueError, match="fails once"):
            engine.fail_drone(0, FailureMode.SILENT)
