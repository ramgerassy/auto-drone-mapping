"""Data types for the mapping module."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MapConfig:
    """Configuration for the occupancy grid.

    Attributes:
        resolution: Meters per grid cell.
        origin_x: World x-coordinate of grid cell (0, 0).
        origin_y: World y-coordinate of grid cell (0, 0).
        grid_width: Number of cells in x direction.
        grid_height: Number of cells in y direction.
        log_odds_occ: Log-odds increment for occupied observations.
        log_odds_free: Log-odds increment for free observations.
        log_odds_min: Lower clamp bound for log-odds values.
        log_odds_max: Upper clamp bound for log-odds values.
    """

    resolution: float = 0.1
    origin_x: float = 0.0
    origin_y: float = 0.0
    grid_width: int = 200
    grid_height: int = 200
    log_odds_occ: float = math.log(0.7 / 0.3)  # ~0.847
    log_odds_free: float = math.log(0.4 / 0.6)  # ~-0.405
    log_odds_min: float = -5.0
    log_odds_max: float = 5.0
