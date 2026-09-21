"""Tests for FailureInjector — applying a failure schedule tick by tick."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.failure import FailureInjector, ScheduledFailure
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


@pytest.fixture
def engine() -> SimulationEngine:
    """Two healthy drones in the small indoor room."""
    return SimulationEngine(
        SCENE_PATH,
        {0: np.array([0.0, 0.0, 1.0]), 1: np.array([2.0, 0.0, 1.0])},
    )


def silent(drone_id: int, tick: int) -> ScheduledFailure:
    """Build a silent-mode ScheduledFailure for the given drone and tick."""
    return ScheduledFailure(drone_id=drone_id, tick=tick, mode=FailureMode.SILENT)


class TestApply:
    """FailureInjector.apply fires each scheduled failure exactly once."""

    def test_nothing_happens_before_the_scheduled_tick(
        self, engine: SimulationEngine
    ) -> None:
        """Nothing fires ahead of its scheduled tick."""
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(99) == []
        assert engine.heartbeat(0)

    def test_the_drone_fails_on_its_tick(self, engine: SimulationEngine) -> None:
        """The failure fires on its own scheduled tick."""
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(100) == [silent(0, 100)]
        assert not engine.heartbeat(0)

    def test_each_failure_is_applied_once(self, engine: SimulationEngine) -> None:
        """A failure already applied does not fire again."""
        injector = FailureInjector(engine, [silent(0, 100)])
        injector.apply(100)
        assert injector.apply(101) == []  # and no "fails once" ValueError

    def test_a_missed_tick_still_fires(self, engine: SimulationEngine) -> None:
        """A caller that skips ticks must not silently skip a failure."""
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(150) == [silent(0, 100)]

    def test_same_tick_failures_apply_in_drone_order(
        self, engine: SimulationEngine
    ) -> None:
        """Multiple failures due on the same tick apply in drone_id order."""
        injector = FailureInjector(engine, [silent(1, 50), silent(0, 50)])
        assert [f.drone_id for f in injector.apply(50)] == [0, 1]


class TestValidation:
    """The injector rejects a schedule naming a drone not in this run."""

    def test_a_drone_not_in_this_run_is_rejected(
        self, engine: SimulationEngine
    ) -> None:
        """Decision D6: a --drones override must not drop a failure silently."""
        with pytest.raises(ValueError, match="drone 4"):
            FailureInjector(engine, [silent(4, 10)])
