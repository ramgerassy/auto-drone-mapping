"""2.5D occupancy grid with log-odds updates."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.mapping.types import MapConfig


class OccupancyGrid:
    """2D occupancy grid with per-cell height tracking.

    Each cell stores a log-odds occupancy value and a height value.
    Log-odds of zero means unknown (p=0.5). Positive values indicate
    occupied, negative values indicate free. Height tracks the maximum
    observed z-coordinate from sensor hits.

    Array convention: internal arrays use ``[row, col]`` indexing
    (numpy row-major). The public API uses ``(col, row)`` which
    maps to ``(x, y)`` in world coordinates.

    Args:
        config: Grid configuration (resolution, origin, size, params).
    """

    def __init__(self, config: MapConfig) -> None:
        self._config = config
        self._log_odds = np.zeros(
            (config.grid_height, config.grid_width), dtype=np.float64
        )
        self._height = np.full(
            (config.grid_height, config.grid_width),
            -np.inf,
            dtype=np.float64,
        )

    @property
    def config(self) -> MapConfig:
        """The grid configuration."""
        return self._config

    @property
    def log_odds(self) -> NDArray[np.float64]:
        """The log-odds occupancy array, shape (height, width)."""
        return self._log_odds

    @property
    def height(self) -> NDArray[np.float64]:
        """The height array, shape (height, width)."""
        return self._height

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates to grid cell indices.

        Args:
            wx: World x-coordinate in meters.
            wy: World y-coordinate in meters.

        Returns:
            Tuple of (col, row) grid indices.
        """
        col = int(math.floor((wx - self._config.origin_x) / self._config.resolution))
        row = int(math.floor((wy - self._config.origin_y) / self._config.resolution))
        return col, row

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center).

        Args:
            col: Column index (x direction).
            row: Row index (y direction).

        Returns:
            Tuple of (wx, wy) world coordinates at cell center.
        """
        wx = self._config.origin_x + (col + 0.5) * self._config.resolution
        wy = self._config.origin_y + (row + 0.5) * self._config.resolution
        return wx, wy

    def in_bounds(self, col: int, row: int) -> bool:
        """Check if grid indices are within bounds.

        Args:
            col: Column index.
            row: Row index.

        Returns:
            True if the cell is within the grid.
        """
        return (
            0 <= col < self._config.grid_width and 0 <= row < self._config.grid_height
        )

    def update_occupied(self, col: int, row: int, hit_z: float) -> None:
        """Mark a cell as occupied and update its height.

        Adds the occupied log-odds increment, updates the height to
        the maximum of the current and observed z, and clamps.

        Args:
            col: Column index.
            row: Row index.
            hit_z: The z-coordinate of the observation.
        """
        self._log_odds[row, col] = np.clip(
            self._log_odds[row, col] + self._config.log_odds_occ,
            self._config.log_odds_min,
            self._config.log_odds_max,
        )
        self._height[row, col] = max(self._height[row, col], hit_z)

    def update_free(self, col: int, row: int) -> None:
        """Mark a cell as free.

        Adds the free log-odds increment and clamps.

        Args:
            col: Column index.
            row: Row index.
        """
        self._log_odds[row, col] = np.clip(
            self._log_odds[row, col] + self._config.log_odds_free,
            self._config.log_odds_min,
            self._config.log_odds_max,
        )

    def probability(self) -> NDArray[np.float64]:
        """Convert the log-odds grid to a probability grid.

        Returns:
            Array of occupancy probabilities in [0, 1],
            shape (height, width). 0.5 means unknown.
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
