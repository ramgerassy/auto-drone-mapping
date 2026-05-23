"""Tests for OccupancyGrid."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig


@pytest.fixture
def config() -> MapConfig:
    """A small 10x10 grid with 1m resolution, origin at (0, 0)."""
    return MapConfig(
        resolution=1.0, origin_x=0.0, origin_y=0.0, grid_width=10, grid_height=10
    )


@pytest.fixture
def grid(config: MapConfig) -> OccupancyGrid:
    """An OccupancyGrid built from the test config."""
    return OccupancyGrid(config)


class TestConstruction:
    """Tests for grid initialization."""

    def test_log_odds_shape(self, grid: OccupancyGrid, config: MapConfig) -> None:
        """Log-odds array has shape (height, width)."""
        assert grid.log_odds.shape == (config.grid_height, config.grid_width)

    def test_log_odds_initialized_to_zero(self, grid: OccupancyGrid) -> None:
        """All log-odds cells start at zero (unknown)."""
        assert np.all(grid.log_odds == 0.0)

    def test_height_initialized_to_neg_inf(self, grid: OccupancyGrid) -> None:
        """All height cells start at -inf (no observation)."""
        assert np.all(np.isneginf(grid.height))


class TestCoordinateTransforms:
    """Tests for world_to_grid and grid_to_world."""

    def test_world_to_grid_origin(self, grid: OccupancyGrid) -> None:
        """World origin maps to grid cell (0, 0)."""
        col, row = grid.world_to_grid(0.5, 0.5)
        assert (col, row) == (0, 0)

    def test_world_to_grid_known_point(self, grid: OccupancyGrid) -> None:
        """Known world point maps to expected cell."""
        col, row = grid.world_to_grid(3.5, 7.2)
        assert (col, row) == (3, 7)

    def test_grid_to_world_cell_center(self, grid: OccupancyGrid) -> None:
        """Grid cell (0, 0) center is at (0.5, 0.5) with 1m resolution."""
        wx, wy = grid.grid_to_world(0, 0)
        assert wx == pytest.approx(0.5)
        assert wy == pytest.approx(0.5)

    def test_round_trip(self, grid: OccupancyGrid) -> None:
        """world_to_grid then grid_to_world returns a point within the same cell."""
        wx, wy = 4.7, 2.3
        col, row = grid.world_to_grid(wx, wy)
        wx2, wy2 = grid.grid_to_world(col, row)
        # Cell center should be within resolution of original point
        assert abs(wx2 - wx) < grid.config.resolution
        assert abs(wy2 - wy) < grid.config.resolution


class TestInBounds:
    """Tests for bounds checking."""

    def test_inside(self, grid: OccupancyGrid) -> None:
        """Cell (5, 5) is within a 10x10 grid."""
        assert grid.in_bounds(5, 5)

    def test_edge(self, grid: OccupancyGrid) -> None:
        """Cell (9, 9) is the last valid cell."""
        assert grid.in_bounds(9, 9)
        assert grid.in_bounds(0, 0)

    def test_outside(self, grid: OccupancyGrid) -> None:
        """Cells outside the grid return False."""
        assert not grid.in_bounds(10, 5)
        assert not grid.in_bounds(5, 10)
        assert not grid.in_bounds(-1, 0)


class TestUpdates:
    """Tests for cell update methods."""

    def test_update_occupied_increases_log_odds(self, grid: OccupancyGrid) -> None:
        """Occupied update increases log-odds."""
        grid.update_occupied(3, 3, 1.0)
        assert grid.log_odds[3, 3] > 0.0

    def test_update_free_decreases_log_odds(self, grid: OccupancyGrid) -> None:
        """Free update decreases log-odds."""
        grid.update_free(3, 3)
        assert grid.log_odds[3, 3] < 0.0

    def test_update_occupied_tracks_max_height(self, grid: OccupancyGrid) -> None:
        """Height tracks the maximum observed z."""
        grid.update_occupied(3, 3, 2.5)
        assert grid.height[3, 3] == pytest.approx(2.5)
        grid.update_occupied(3, 3, 3.0)
        assert grid.height[3, 3] == pytest.approx(3.0)
        grid.update_occupied(3, 3, 1.0)
        assert grid.height[3, 3] == pytest.approx(3.0)  # not decreased

    def test_clamping_occupied(self, grid: OccupancyGrid) -> None:
        """Repeated occupied updates clamp at log_odds_max."""
        for _ in range(100):
            grid.update_occupied(0, 0, 1.0)
        assert grid.log_odds[0, 0] == pytest.approx(grid.config.log_odds_max)

    def test_clamping_free(self, grid: OccupancyGrid) -> None:
        """Repeated free updates clamp at log_odds_min."""
        for _ in range(100):
            grid.update_free(0, 0)
        assert grid.log_odds[0, 0] == pytest.approx(grid.config.log_odds_min)


class TestProbability:
    """Tests for log-odds to probability conversion."""

    def test_zero_log_odds_is_half(self, grid: OccupancyGrid) -> None:
        """Zero log-odds gives 0.5 probability."""
        probs = grid.probability()
        np.testing.assert_allclose(probs, 0.5)

    def test_positive_log_odds_above_half(self, grid: OccupancyGrid) -> None:
        """Positive log-odds gives probability > 0.5."""
        grid.update_occupied(0, 0, 1.0)
        probs = grid.probability()
        assert probs[0, 0] > 0.5

    def test_negative_log_odds_below_half(self, grid: OccupancyGrid) -> None:
        """Negative log-odds gives probability < 0.5."""
        grid.update_free(0, 0)
        probs = grid.probability()
        assert probs[0, 0] < 0.5
