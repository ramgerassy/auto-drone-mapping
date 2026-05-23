"""Export occupancy grid as .npz data and .png image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from swarm_mapping.mapping.grid import OccupancyGrid


def save_npz(grid: OccupancyGrid, path: str | Path) -> None:
    """Save grid data as a compressed .npz file.

    Contents:
        log_odds: float64 array, shape (H, W).
        probability: float64 array, shape (H, W), derived from log_odds.
        height: float64 array, shape (H, W).
        resolution: scalar float.
        origin: (origin_x, origin_y) array.

    Args:
        grid: The occupancy grid to save.
        path: Output file path (should end in .npz).
    """
    np.savez_compressed(
        str(path),
        log_odds=grid.log_odds,
        probability=grid.probability(),
        height=grid.height,
        resolution=np.array(grid.config.resolution),
        origin=np.array([grid.config.origin_x, grid.config.origin_y]),
    )


def save_png(grid: OccupancyGrid, path: str | Path) -> None:
    """Save a grayscale occupancy image for human inspection.

    Color mapping:
        - Free (p < 0.4): white (255)
        - Occupied (p > 0.6): black (0)
        - Unknown (0.4 <= p <= 0.6): gray (128)

    The image origin is bottom-left (row 0 = bottom of image),
    matching the world coordinate convention.

    Args:
        grid: The occupancy grid to save.
        path: Output file path (should end in .png).
    """
    probs = grid.probability()

    # Build grayscale image
    pixels = np.full(probs.shape, 128, dtype=np.uint8)
    pixels[probs < 0.4] = 255  # free = white
    pixels[probs > 0.6] = 0  # occupied = black

    # Flip vertically so row 0 (y=0) is at bottom of image
    pixels = np.flipud(pixels)

    img = Image.fromarray(pixels, mode="L")
    img.save(str(path))
