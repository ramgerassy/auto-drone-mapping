"""Unit tests for CLI helpers.

Only the pure logic is tested here — swarm sizing and coverage arithmetic.
The wiring that needs MuJoCo lives in `tests/integration/test_e2e.py`, and the
live viewer needs a display, so it is exercised manually rather than in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.cli import (
    MissionResult,
    coverage_fraction,
    select_start_positions,
)
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig

pytestmark = pytest.mark.sprint(2)  # tests introduced in Sprint 2

POSITIONS = (
    (0.0, 0.0, 1.0),
    (0.0, -1.2, 1.0),
    (0.0, 1.2, 1.0),
)


class TestSelectStartPositions:
    """`--drones N` takes the first N configured positions, or fails."""

    def test_none_keeps_every_configured_position(self) -> None:
        """Without the flag, the config's swarm is used unchanged."""
        assert select_start_positions(POSITIONS, None) == POSITIONS

    @pytest.mark.sanity
    def test_takes_the_first_n_in_order(self) -> None:
        """A 1-drone run is the 3-drone run's first drone, not a new one.

        This is what makes the scaling KPI a fair comparison: both arms share
        every other parameter and the smaller arm is a prefix of the larger.
        """
        assert select_start_positions(POSITIONS, 1) == (POSITIONS[0],)
        assert select_start_positions(POSITIONS, 2) == POSITIONS[:2]

    def test_exact_count_is_a_no_op(self) -> None:
        """Asking for exactly what is configured changes nothing."""
        assert select_start_positions(POSITIONS, 3) == POSITIONS

    def test_more_drones_than_positions_is_rejected(self) -> None:
        """The CLI never invents a start position.

        A fabricated position has to be collision-free, inside the grid, and
        min_separation clear of its neighbours. A flag guessing at that would
        be a safety parameter set by accident.
        """
        with pytest.raises(ValueError) as excinfo:
            select_start_positions(POSITIONS, 4)

        message = str(excinfo.value)
        assert "4" in message
        assert "3" in message

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_count_is_rejected(self, count: int) -> None:
        """An empty swarm reports instant success having mapped nothing."""
        with pytest.raises(ValueError, match="at least 1"):
            select_start_positions(POSITIONS, count)


class TestMissionSucceeded:
    """What the CLI's exit code is allowed to fail on.

    Pinned as its own test because the obvious rule — fail on `blocked` — is
    wrong, and wrong in a way that only shows up on a *successful* run.
    Clearance inflation leaves wall-adjacent frontiers that can be seen but
    not occupied, so `small_indoor` terminates blocked at 98.1% coverage.
    Failing on it would exit non-zero on the happy path.
    """

    def result(self, *, blocked: bool, tick_capped: bool) -> MissionResult:
        """Build a MissionResult varying only the two outcome flags."""
        return MissionResult(
            ticks=100,
            coverage=0.98,
            blocked=blocked,
            unreachable_frontiers=3 if blocked else 0,
            tick_capped=tick_capped,
            npz_path=Path("map.npz"),
            png_path=Path("map.png"),
        )

    def test_blocked_alone_is_not_a_failure(self) -> None:
        """Residual unreachable frontiers are normal, not a failed mission."""
        assert self.result(blocked=True, tick_capped=False).succeeded

    def test_hitting_the_tick_cap_is_a_failure(self) -> None:
        """A mission that never converged did not succeed."""
        assert not self.result(blocked=False, tick_capped=True).succeeded

    def test_a_clean_run_succeeds(self) -> None:
        """Nothing wrong, nothing reported."""
        assert self.result(blocked=False, tick_capped=False).succeeded


class TestCoverageFraction:
    """Coverage is the fraction of cells classified either free or occupied."""

    def test_untouched_grid_has_zero_coverage(self) -> None:
        """Every cell starts at p=0.5 — unknown, and so uncovered."""
        grid = OccupancyGrid(MapConfig(grid_width=10, grid_height=10))

        assert coverage_fraction(grid) == 0.0

    def test_confident_cells_count_as_covered(self) -> None:
        """Cells pushed clear of the unknown band count, whichever way."""
        grid = OccupancyGrid(MapConfig(grid_width=10, grid_height=10))
        grid.log_odds[0, :5] = 5.0  # confidently occupied
        grid.log_odds[1, :5] = -5.0  # confidently free

        assert coverage_fraction(grid) == pytest.approx(10 / 100)

    def test_cells_inside_the_unknown_band_do_not_count(self) -> None:
        """A cell nudged but not resolved is still unknown, not covered.

        The band is (0.4, 0.6); a cell at p=0.55 has been seen but not
        decided, and counting it would inflate the KPI with cells the map
        cannot actually classify.
        """
        grid = OccupancyGrid(MapConfig(grid_width=10, grid_height=10))
        grid.log_odds[0, :] = np.log(0.55 / 0.45)

        assert coverage_fraction(grid) == 0.0
