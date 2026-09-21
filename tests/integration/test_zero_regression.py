"""Zero-regression gate for Sprint 3 (docs/sprint-3-plan.md).

Reading poses back from the localizer must change nothing for a healthy
swarm. Baseline measured at 02a9a13, before Feature 10. Exact integers on
purpose: the decisions are discrete, and a one-tick drift is a behaviour
change. Map hashes are checked locally, not here (F10-R7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.cli import build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config

pytestmark = pytest.mark.sprint(3)

SMALL_INDOOR = (
    Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
)
BASELINE = {1: (498, 0.9799), 2: (254, 0.9822), 3: (197, 0.9831)}


@pytest.mark.parametrize("drones", sorted(BASELINE))
def test_a_healthy_run_is_unchanged(drones: int) -> None:
    """Same ticks and coverage as the pre-Feature-10 baseline."""
    config = load_config(SMALL_INDOOR)
    mission = build_mission(config, drones)
    while (
        not mission.master.is_complete
        and mission.master.tick_count < config.coordination.max_ticks
    ):
        mission.tick()
    ticks, coverage = BASELINE[drones]
    assert mission.master.tick_count == ticks
    assert round(coverage_fraction(mission.mapper.grid), 4) == coverage
