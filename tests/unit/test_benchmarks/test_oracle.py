"""Tests for the ground-truth oracle strategy (benchmarks.oracle).

Benchmark code, but tested like shipped code, and for a specific reason: this
sprint produced three conclusions that were wrong because the *instrument* was
wrong, not the system. An oracle that silently over-reports gain would produce
a confident "NBV is worth building" that nothing downstream could catch.

Pure-logic on hand-built OccupancyGrids — no MuJoCo. Cells are (col, row); the
log_odds array is [row, col]. Resolution 1.0 with origin (0, 0) puts cell
(c, r) at world centre (c + 0.5, r + 0.5).
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.oracle import OracleInfoGainFrontier, TruthVisibility

from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.frontier_strategy import FrontierStrategy, NearestFrontier
from swarm_mapping.planning.path_planner import AStarPlanner

pytestmark = pytest.mark.sprint(2)  # Sprint 2.5 — pricing the NBV family

FREE, OCC, UNK = -2.0, 2.0, 0.0

Cell = tuple[int, int]


def make_grid(width: int = 21, height: int = 21, fill: float = UNK) -> OccupancyGrid:
    """A width x height grid of 1 m cells at origin (0, 0)."""
    grid = OccupancyGrid(
        MapConfig(
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            grid_width=width,
            grid_height=height,
        )
    )
    grid.log_odds[:] = fill
    return grid


def region(col: int, row: int, size: int = 3) -> FrontierRegion:
    """A single-cell-representative frontier region centred on that cell."""
    return FrontierRegion(centroid=(col + 0.5, row + 0.5), cell=(col, row), size=size)


def unknown_mask(grid: OccupancyGrid) -> np.ndarray:
    """Flattened unknown mask, as `OracleInfoGainFrontier` computes it."""
    prob = grid.probability()
    return ((prob >= 0.4) & (prob <= 0.6)).ravel()


class TestTruthVisibility:
    """What the oracle can see, and what it correctly cannot."""

    def test_an_open_room_is_visible_out_to_the_full_range(self) -> None:
        """Every in-range cell is seen — i.e. the ray fan leaves no gaps.

        This is the failure mode that would quietly inflate the bound: too few
        rays and distant cells fall between them, so a far frontier scores less
        gain than it deserves and the oracle under-selects it. Asserting the
        disc is *complete* catches that directly, where a spot check would not.
        """
        truth = make_grid(fill=FREE)
        vis = TruthVisibility(truth, max_range=8.0)

        seen = set(vis.visible_from((10, 10)).tolist())
        missed = [
            (col, row)
            for row in range(21)
            for col in range(21)
            if (col - 10) ** 2 + (row - 10) ** 2 <= 7**2
            and (row * 21 + col) not in seen
        ]

        assert missed == []

    def test_a_wall_hides_what_is_behind_it(self) -> None:
        """The wall face is observed; the cells beyond it are not."""
        truth = make_grid(fill=FREE)
        truth.log_odds[:, 12] = OCC  # full-height wall at col 12
        vis = TruthVisibility(truth, max_range=8.0)

        seen = set(vis.visible_from((10, 10)).tolist())

        assert (10 * 21 + 12) in seen, "the wall face itself is observable"
        assert (10 * 21 + 13) not in seen, "the cell behind the wall is not"
        assert (10 * 21 + 15) not in seen

    def test_visibility_stops_at_the_sensor_range(self) -> None:
        """Range is a hard clip — an oracle is better informed, not unbounded."""
        truth = make_grid(fill=FREE)
        vis = TruthVisibility(truth, max_range=4.0)

        seen = set(vis.visible_from((10, 10)).tolist())

        assert (10 * 21 + 13) in seen, "3 cells away, inside a 4 m range"
        assert (10 * 21 + 18) not in seen, "8 cells away, well outside it"

    def test_gain_counts_only_the_cells_still_unknown(self) -> None:
        """Re-observing known cells is worth nothing, which is the whole point."""
        truth = make_grid(fill=FREE)
        vis = TruthVisibility(truth, max_range=5.0)
        belief = make_grid(fill=UNK)

        all_unknown = vis.gain((10, 10), unknown_mask(belief))
        belief.log_odds[8:13, 8:13] = FREE  # a 5x5 patch already mapped
        partly_known = vis.gain((10, 10), unknown_mask(belief))

        assert partly_known == all_unknown - 25


class TestOracleSelection:
    """That the oracle chooses on gain, and still flies legal routes."""

    @staticmethod
    def _two_rooms() -> tuple[OccupancyGrid, OccupancyGrid]:
        """A belief/truth pair: a near closet and a far open hall.

        The drone sits at (10, 10) on a mapped east-west corridor. West at
        col 4 is a frontier opening a 3x3 closet; east at col 17 is a frontier
        opening the rest of the map. Nearest-frontier takes the closet.
        """
        truth = make_grid(fill=FREE)
        truth.log_odds[0:8, 0:8] = OCC
        truth.log_odds[13:21, 0:8] = OCC
        truth.log_odds[8:13, 0:3] = OCC  # closet is 3 cells deep, rows 8-12
        belief = make_grid(fill=UNK)
        belief.log_odds[10, 4:18] = FREE  # the mapped corridor only
        return belief, truth

    def test_it_crosses_the_map_for_the_bigger_reveal(self) -> None:
        """With distance free, the oracle takes the hall over the near closet."""
        belief, truth = self._two_rooms()
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(truth, max_range=8.0)
        near, far = region(4, 10), region(17, 10)

        baseline = NearestFrontier(planner).select(belief, [near, far], (10, 10))
        oracle = OracleInfoGainFrontier(planner, vis, decay=0.0).select(
            belief, [near, far], (10, 10)
        )

        assert baseline is not None and oracle is not None
        assert baseline.region.cell == (4, 10), "nearest frontier takes the closet"
        assert oracle.region.cell == (17, 10), "the oracle pays the distance"

    def test_a_steep_decay_reproduces_nearest_frontier(self) -> None:
        """The swept family contains the baseline at one end.

        Without this the sweep could not be read as a bound *over the family*:
        if no setting of `decay` recovered nearest-frontier behaviour, a poor
        result would be ambiguous between "NBV does not help" and "this
        particular weighting is bad".
        """
        belief, truth = self._two_rooms()
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(truth, max_range=8.0)
        near, far = region(4, 10), region(17, 10)

        baseline = NearestFrontier(planner).select(belief, [near, far], (10, 10))
        oracle = OracleInfoGainFrontier(planner, vis, decay=5.0).select(
            belief, [near, far], (10, 10)
        )

        assert baseline is not None and oracle is not None
        assert oracle.region.cell == baseline.region.cell

    def test_the_route_never_leaves_known_free_space(self) -> None:
        """The oracle scores with truth but flies on the belief map.

        An oracle that also routed through unmapped cells would be measuring a
        system the project does not ship — the planner deliberately refuses to
        fly through unknown space.
        """
        belief, truth = self._two_rooms()
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(truth, max_range=8.0)

        oracle = OracleInfoGainFrontier(planner, vis, decay=0.0).select(
            belief, [region(4, 10), region(17, 10)], (10, 10)
        )

        assert oracle is not None
        probability = belief.probability()
        assert all(probability[row, col] < 0.4 for col, row in oracle.path)

    def test_a_claimed_frontier_is_never_selected(self) -> None:
        """However much gain it promises — the coordinator owns exclusivity."""
        belief, truth = self._two_rooms()
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(truth, max_range=8.0)
        near, far = region(4, 10), region(17, 10)

        oracle = OracleInfoGainFrontier(planner, vis, decay=0.0).select(
            belief, [near, far], (10, 10), claimed=[far]
        )

        assert oracle is not None
        assert oracle.region.cell == (4, 10)

    def test_a_frontier_walled_off_in_the_belief_map_is_skipped(self) -> None:
        """Unreachable now means unreachable, gain notwithstanding."""
        belief, truth = self._two_rooms()
        belief.log_odds[10, 13] = OCC  # sever the corridor east of the drone
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(truth, max_range=8.0)

        oracle = OracleInfoGainFrontier(planner, vis, decay=0.0).select(
            belief, [region(4, 10), region(17, 10)], (10, 10)
        )

        assert oracle is not None
        assert oracle.region.cell == (4, 10)

    def test_it_satisfies_the_frontier_strategy_seam(self) -> None:
        """Structural: the coordinator takes it with no knowledge of the swap."""
        planner = AStarPlanner(clearance_radius=0.0)
        vis = TruthVisibility(make_grid(fill=FREE), max_range=4.0)
        strategy: FrontierStrategy = OracleInfoGainFrontier(planner, vis, decay=0.1)

        assert strategy.select(make_grid(fill=UNK), [], (1, 1)) is None
