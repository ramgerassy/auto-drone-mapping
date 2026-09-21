"""Recovery from a mid-mission failure, end to end, fast enough for every push.

`small_indoor` with three drones finishes in 197 healthy ticks, so a failure
at tick 40 is mid-mission. Driven through `Mission.tick()`, the one place the
schedule meets the loop: the master finds the failure from symptoms.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import pytest

from swarm_mapping.cli import Mission, build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings
from swarm_mapping.coordination.types import DroneHealth

pytestmark = pytest.mark.sprint(3)

SMALL_INDOOR = (
    Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
)
FAIL_AT = 40
EXPECTED = {"silent": DroneHealth.LOST, "stuck": DroneHealth.STUCK}


def fly(mode: str) -> tuple[Mission, list[logging.LogRecord]]:
    """Run small_indoor, 3 drones, drone 2 failing at FAIL_AT.

    Returns:
        The finished mission and its log records.
    """
    config = replace(
        load_config(SMALL_INDOOR),
        failures=(FailureSettings(drone_id=2, tick=FAIL_AT, mode=mode),),
    )
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    try:
        mission = build_mission(config, drones=3)
        separation = config.coordination.min_separation / config.map.resolution
        while (
            not mission.master.is_complete
            and mission.master.tick_count < config.coordination.max_ticks
        ):
            mission.tick()
            wreck = mission.master.drone_states[2]
            if wreck.health is not DroneHealth.ACTIVE:
                for drone_id in (0, 1):
                    col, row = mission.master.drone_states[drone_id].cell
                    assert (
                        math.hypot(col - wreck.cell[0], row - wreck.cell[1])
                        >= separation
                    )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
    return mission, records


def event_tick(records: list[logging.LogRecord], name: str) -> int:
    """Return the `tick` field of the single log record with this message."""
    (record,) = [r for r in records if r.getMessage() == name]
    return int(record.tick)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["silent", "stuck"])
def test_the_swarm_detects_the_failure_and_still_maps_the_room(mode: str) -> None:
    """Detection lands under the 2s KPI and the surviving swarm still covers 95%."""
    mission, records = fly(mode)
    config = mission.config

    assert mission.master.drone_states[2].health is EXPECTED[mode]
    latency = event_tick(records, "drone_failed") - event_tick(
        records, "failure_injected"
    )
    assert latency * config.tick_seconds < 2.0, f"{latency} ticks"

    assert mission.master.tick_count < config.coordination.max_ticks
    assert coverage_fraction(mission.mapper.grid) >= 0.95


def test_a_failure_run_is_deterministic() -> None:
    """Same config, same seed: identical tick count and identical event tick."""
    first, first_events = fly("silent")
    second, second_events = fly("silent")
    assert first.master.tick_count == second.master.tick_count
    assert event_tick(first_events, "drone_failed") == event_tick(
        second_events, "drone_failed"
    )
