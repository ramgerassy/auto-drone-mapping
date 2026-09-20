"""Acceptance: the 1 -> 3 drone scaling KPI.

Sprint 2 commits to ">= 1.5x speedup from 1 to 3 drones (small indoor)".
Speedup is measured in **ticks to reach 95% coverage** (decision D3), not in
wall-clock seconds: per-tick compute rises with drone count — three drones scan
three times and plan three times — so a wall-clock metric would penalise the
swarm for the very parallelism it is meant to reward, and would be noisy on CI
besides. Ticks are deterministic, so this number is reproducible.

Both arms come from one config with a `--drones` override taking the first N
start positions (decision D4), so the comparison differs in swarm size and
nothing else. Two config files could drift and make the KPI lie rather than
fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.cli import build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config

pytestmark = [pytest.mark.sprint(2), pytest.mark.acceptance]

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "scenarios" / "small_indoor" / "config.yaml"

TARGET_COVERAGE = 0.95
REQUIRED_SPEEDUP = 1.5


def ticks_to_target(drones: int) -> tuple[int, float]:
    """Ticks for `drones` drones to reach TARGET_COVERAGE on small_indoor.

    Args:
        drones: Swarm size, taking the first N configured start positions.

    Returns:
        (ticks, final_coverage). Ticks is the cap if the target is never met,
        which the caller surfaces rather than silently comparing against.
    """
    config = load_config(SMALL)
    mission = build_mission(config, drones=drones)
    cap = config.coordination.max_ticks

    while mission.master.tick_count < cap:
        mission.master.tick()
        if coverage_fraction(mission.mapper.grid) >= TARGET_COVERAGE:
            break
        if mission.master.is_complete:
            break

    return mission.master.tick_count, coverage_fraction(mission.mapper.grid)


class TestScalingKpi:
    """More drones must explore materially faster."""

    def test_three_drones_are_at_least_1_5x_faster_than_one(self) -> None:
        """The Tier-2 scaling KPI, reported with its measured value.

        A bare assert would say only "red"; the message carries both arms so a
        near-miss is legible and a regression can be told from a tuning change.
        """
        solo_ticks, solo_coverage = ticks_to_target(1)
        swarm_ticks, swarm_coverage = ticks_to_target(3)

        assert solo_coverage >= TARGET_COVERAGE, (
            f"1 drone never reached {TARGET_COVERAGE:.0%} "
            f"(stopped at {solo_coverage:.2%} after {solo_ticks} ticks) — "
            "the baseline arm is invalid, so the ratio is meaningless"
        )
        assert swarm_coverage >= TARGET_COVERAGE, (
            f"3 drones never reached {TARGET_COVERAGE:.0%} "
            f"(stopped at {swarm_coverage:.2%} after {swarm_ticks} ticks)"
        )

        speedup = solo_ticks / swarm_ticks
        assert speedup >= REQUIRED_SPEEDUP, (
            f"speedup {speedup:.2f}x is under the {REQUIRED_SPEEDUP}x KPI: "
            f"1 drone took {solo_ticks} ticks, 3 drones took {swarm_ticks}"
        )
