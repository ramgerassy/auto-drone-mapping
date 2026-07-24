"""Tests for frontier detection (mapping.frontier).

Pure-logic on hand-built OccupancyGrids — no MuJoCo. Cell classes are set
directly via log_odds: -2 => free (p<0.4), +2 => occupied (p>0.6), 0 => unknown.
"""

from __future__ import annotations

import pytest

from swarm_mapping.mapping.frontier import (
    FrontierRegion,
    detect_frontiers,
    frontier_cells,
)
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig

pytestmark = pytest.mark.sprint(2)  # Feature 1 — frontier detection

FREE, OCC, UNK = -2.0, 2.0, 0.0


def make_grid(width: int = 10, height: int = 10) -> OccupancyGrid:
    """A width x height grid, 1 m cells, origin (0, 0), all-unknown."""
    return OccupancyGrid(
        MapConfig(
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            grid_width=width,
            grid_height=height,
        )
    )


class TestFrontierCells:
    """The cell-level predicate: free cell with >=1 unknown 4-neighbor."""

    def test_all_unknown_has_no_frontier_cells(self) -> None:
        """An untouched (all-unknown) grid has no free cells, so no frontiers."""
        assert frontier_cells(make_grid()) == []

    def test_all_free_has_no_frontier_cells(self) -> None:
        """All-free grid: no unknown neighbors, and the grid edge is not unknown."""
        g = make_grid()
        g.log_odds[:] = FREE
        assert frontier_cells(g) == []

    @pytest.mark.sanity
    def test_free_cell_next_to_unknown_is_frontier(self) -> None:
        """Free cells bordering unknown space are reported as frontier cells."""
        g = make_grid()
        g.log_odds[5, 4] = FREE  # (col=4, row=5)
        g.log_odds[5, 5] = FREE  # (col=5, row=5)
        assert set(frontier_cells(g)) == {(4, 5), (5, 5)}

    def test_free_cell_walled_by_occupied_is_not_frontier(self) -> None:
        """A free cell whose 4-neighbors are occupied has no unknown side."""
        g = make_grid()
        g.log_odds[5, 5] = FREE
        for c, r in [(4, 5), (6, 5), (5, 4), (5, 6)]:  # 4-neighbors occupied
            g.log_odds[r, c] = OCC
        assert frontier_cells(g) == []

    def test_interior_free_cell_excluded(self) -> None:
        """A free cell surrounded by free (no unknown neighbor) is not a frontier."""
        g = make_grid()
        for r in range(2, 5):
            for c in range(2, 5):
                g.log_odds[r, c] = FREE
        cells = frontier_cells(g)
        assert (3, 3) not in cells  # center of the 3x3 free block
        assert len(cells) == 8

    def test_grid_edge_and_4_connectivity(self) -> None:
        """Only 4-neighbors of unknown count; grid-edge corners are excluded."""
        g = make_grid(3, 3)
        g.log_odds[:] = FREE
        g.log_odds[1, 1] = UNK
        assert frontier_cells(g) == [(1, 0), (0, 1), (2, 1), (1, 2)]  # row-major


class TestDetectFrontiers:
    """Clustering frontier cells into regions."""

    def test_no_frontiers_returns_empty(self) -> None:
        """No frontier cells yields no regions."""
        assert detect_frontiers(make_grid()) == []

    @pytest.mark.sanity
    def test_single_region_centroid_and_size(self) -> None:
        """Adjacent frontier cells form one region with the right centroid/size."""
        g = make_grid()
        g.log_odds[5, 4] = FREE
        g.log_odds[5, 5] = FREE
        regions = detect_frontiers(g)
        assert len(regions) == 1
        assert regions[0].size == 2
        assert regions[0].centroid == pytest.approx((5.0, 5.5))
        assert regions[0].cell in {(4, 5), (5, 5)}

    def test_two_separated_regions_ordered_deterministically(self) -> None:
        """Two disjoint clusters give two regions in a stable (row, col) order."""
        g = make_grid()
        g.log_odds[1, 1] = FREE
        g.log_odds[1, 2] = FREE  # region A near row 1
        g.log_odds[7, 7] = FREE
        g.log_odds[7, 8] = FREE  # region B near row 7
        regions = detect_frontiers(g)
        assert len(regions) == 2
        assert regions[0].cell[1] < regions[1].cell[1]  # row of A < row of B

    def test_min_region_size_drops_small(self) -> None:
        """A lone frontier cell is dropped by the default min_region_size=2."""
        g = make_grid()
        g.log_odds[5, 5] = FREE
        assert detect_frontiers(g) == []
        small = detect_frontiers(g, min_region_size=1)
        assert len(small) == 1
        assert small[0].size == 1

    def test_representative_cell_is_free_and_near_centroid(self) -> None:
        """The region's representative cell is a free frontier cell near centroid."""
        g = make_grid()
        for c in (2, 3, 4):  # a 3-cell horizontal free line at row 5
            g.log_odds[5, c] = FREE
        region = detect_frontiers(g)[0]
        assert region.centroid == pytest.approx((3.5, 5.5))
        assert region.cell == (3, 5)
        assert region.cell in frontier_cells(g)

    def test_deterministic_across_calls(self) -> None:
        """Repeated detection on the same grid returns identical regions."""
        g = make_grid()
        for r in range(2, 5):
            for c in range(2, 5):
                g.log_odds[r, c] = FREE
        assert detect_frontiers(g) == detect_frontiers(g)

    def test_returns_frontier_region_instances(self) -> None:
        """Detection returns FrontierRegion dataclass instances."""
        g = make_grid()
        g.log_odds[5, 4] = FREE
        g.log_odds[5, 5] = FREE
        assert all(isinstance(r, FrontierRegion) for r in detect_frontiers(g))
