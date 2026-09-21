"""Tests for `drones.cruise_speed` and the tick duration derived from it (D1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.config.loader import load_config
from tests.unit.test_config.test_schema import load_broken, valid_config, write_config

pytestmark = pytest.mark.sprint(3)

SCENARIOS = Path(__file__).resolve().parents[3] / "scenarios"


def test_tick_seconds_is_one_cell_at_cruise_speed(tmp_path: Path) -> None:
    """0.2 m cells at 1.0 m/s cruise speed take 0.2 s to cross."""
    config = load_config(write_config(tmp_path, valid_config()))  # 0.2 m cells, 1.0 m/s
    assert config.tick_seconds == pytest.approx(0.2)


@pytest.mark.parametrize("value", [0, -1.0])
def test_a_non_positive_cruise_speed_is_rejected(tmp_path: Path, value: float) -> None:
    """A cruise speed of 0 or below would make `tick_seconds` undefined or negative."""
    excinfo = load_broken(
        tmp_path, lambda c: c["drones"].__setitem__("cruise_speed", value)
    )
    assert "drones.cruise_speed" in str(excinfo.value)


def test_a_missing_cruise_speed_is_rejected(tmp_path: Path) -> None:
    """No default: a silently assumed speed would misreport every latency KPI."""
    excinfo = load_broken(tmp_path, lambda c: c["drones"].pop("cruise_speed"))
    assert "drones.cruise_speed" in str(excinfo.value)


@pytest.mark.parametrize(
    ("scenario", "seconds"), [("small_indoor", 0.1), ("large_indoor", 0.2)]
)
def test_shipped_tick_durations(scenario: str, seconds: float) -> None:
    """The tick durations the KPI conversion in the sprint plan relies on."""
    config = load_config(SCENARIOS / scenario / "config.yaml")
    assert config.tick_seconds == pytest.approx(seconds)
