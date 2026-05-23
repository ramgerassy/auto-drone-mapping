"""Tests for map export functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.mapping.export import save_npz, save_png
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig

TEST_CONFIG = MapConfig(
    resolution=0.5,
    origin_x=-2.0,
    origin_y=-2.0,
    grid_width=8,
    grid_height=8,
)


@pytest.fixture
def grid() -> OccupancyGrid:
    """A small grid with some occupied and free cells."""
    g = OccupancyGrid(TEST_CONFIG)
    # Mark some cells as occupied and free
    g.update_occupied(2, 2, 1.5)
    g.update_occupied(2, 2, 1.5)
    g.update_free(5, 5)
    g.update_free(5, 5)
    return g


class TestSaveNpz:
    """Tests for save_npz."""

    def test_file_created(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """NPZ file is created at the given path."""
        out = tmp_path / "map.npz"
        save_npz(grid, out)
        assert out.exists()

    def test_contains_expected_keys(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """NPZ file contains all expected arrays."""
        out = tmp_path / "map.npz"
        save_npz(grid, out)

        data = np.load(str(out))
        expected_keys = {"log_odds", "probability", "height", "resolution", "origin"}
        assert set(data.files) == expected_keys

    def test_shapes_match_grid(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """Saved arrays have the same shape as the grid."""
        out = tmp_path / "map.npz"
        save_npz(grid, out)

        data = np.load(str(out))
        assert data["log_odds"].shape == (
            TEST_CONFIG.grid_height,
            TEST_CONFIG.grid_width,
        )
        assert data["probability"].shape == (
            TEST_CONFIG.grid_height,
            TEST_CONFIG.grid_width,
        )
        assert data["height"].shape == (TEST_CONFIG.grid_height, TEST_CONFIG.grid_width)

    def test_round_trip_values(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """Save then load preserves array values."""
        out = tmp_path / "map.npz"
        save_npz(grid, out)

        data = np.load(str(out))
        np.testing.assert_array_equal(data["log_odds"], grid.log_odds)
        np.testing.assert_array_equal(data["probability"], grid.probability())
        assert data["resolution"] == pytest.approx(TEST_CONFIG.resolution)
        np.testing.assert_array_equal(
            data["origin"], [TEST_CONFIG.origin_x, TEST_CONFIG.origin_y]
        )


class TestSavePng:
    """Tests for save_png."""

    def test_file_created(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """PNG file is created at the given path."""
        out = tmp_path / "map.png"
        save_png(grid, out)
        assert out.exists()

    def test_valid_png(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """File starts with PNG magic bytes."""
        out = tmp_path / "map.png"
        save_png(grid, out)

        magic = out.read_bytes()[:8]
        assert magic == b"\x89PNG\r\n\x1a\n"

    def test_image_dimensions(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """Image dimensions match grid size."""
        from PIL import Image

        out = tmp_path / "map.png"
        save_png(grid, out)

        img = Image.open(str(out))
        # PIL Image.size is (width, height)
        assert img.size == (TEST_CONFIG.grid_width, TEST_CONFIG.grid_height)
        assert img.mode == "RGB"

    def test_empty_grid_all_blue(self, tmp_path: Path) -> None:
        """An empty grid (all unknown) produces an all-blue image."""
        from PIL import Image

        g = OccupancyGrid(TEST_CONFIG)
        out = tmp_path / "empty.png"
        save_png(g, out)

        img = Image.open(str(out))
        pixels = np.array(img)
        # Unknown = blue (70, 130, 180)
        assert np.all(pixels[:, :, 0] == 70)
        assert np.all(pixels[:, :, 1] == 130)
        assert np.all(pixels[:, :, 2] == 180)

    def test_free_cells_are_white(self, grid: OccupancyGrid, tmp_path: Path) -> None:
        """Free cells appear as white in the image."""
        from PIL import Image

        out = tmp_path / "map.png"
        save_png(grid, out)

        img = Image.open(str(out))
        pixels = np.array(img)
        # Grid has free cells at (5,5) — flipped in image
        flipped_row = TEST_CONFIG.grid_height - 1 - 5
        assert np.all(pixels[flipped_row, 5] == [255, 255, 255])

    def test_occupied_cells_use_height_shading(self, tmp_path: Path) -> None:
        """Occupied cells are shaded by height — taller is darker."""
        from PIL import Image

        g = OccupancyGrid(TEST_CONFIG)
        # Two occupied cells at different heights
        for _ in range(10):
            g.update_occupied(2, 2, 1.0)  # short
            g.update_occupied(4, 4, 3.0)  # tall

        out = tmp_path / "height.png"
        save_png(g, out, max_height=3.0)

        img = Image.open(str(out))
        pixels = np.array(img)
        # Taller cell should be darker (lower intensity)
        short_row = TEST_CONFIG.grid_height - 1 - 2
        tall_row = TEST_CONFIG.grid_height - 1 - 4
        short_intensity = pixels[short_row, 2, 0]
        tall_intensity = pixels[tall_row, 4, 0]
        assert tall_intensity < short_intensity

    def test_short_obstacle_not_black_with_high_ceiling(self, tmp_path: Path) -> None:
        """A 0.3m obstacle with max_height=3.0 should be light gray, not black."""
        from PIL import Image

        g = OccupancyGrid(TEST_CONFIG)
        for _ in range(10):
            g.update_occupied(3, 3, 0.3)

        out = tmp_path / "short.png"
        save_png(g, out, max_height=3.0)

        img = Image.open(str(out))
        pixels = np.array(img)
        flipped_row = TEST_CONFIG.grid_height - 1 - 3
        intensity = pixels[flipped_row, 3, 0]
        # 0.3 / 3.0 = 10% of ceiling → intensity ~180 (light gray)
        assert intensity > 150
