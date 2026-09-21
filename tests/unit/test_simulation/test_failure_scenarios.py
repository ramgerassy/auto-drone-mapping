"""The shipped failure scenarios load, and fail the drone they say they fail."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings

pytestmark = pytest.mark.sprint(3)

SCENARIOS = Path(__file__).resolve().parents[3] / "scenarios"


@pytest.mark.parametrize(
    ("name", "mode"), [("failure_injection", "silent"), ("failure_stuck", "stuck")]
)
def test_the_scenario_schedules_one_mid_mission_failure(name: str, mode: str) -> None:
    """F11-R2: drone 1 fails at tick 300, in the mode the scenario name says."""
    config = load_config(SCENARIOS / name / "config.yaml")
    assert config.failures == (FailureSettings(drone_id=1, tick=300, mode=mode),)


@pytest.mark.parametrize("name", ["failure_injection", "failure_stuck"])
def test_the_scenario_is_large_indoor_plus_a_failure(name: str) -> None:
    """CLAUDE.md: 'same as large indoor, with one drone scripted to fail'."""
    large = load_config(SCENARIOS / "large_indoor" / "config.yaml")
    failing = load_config(SCENARIOS / name / "config.yaml")
    # Every field but `failures` matches, including `sensor` — a per-field
    # comparison missed it before (this scenario never varies sensor
    # geometry, so the field-by-field version silently never caught a drift).
    assert replace(failing, failures=()) == large
