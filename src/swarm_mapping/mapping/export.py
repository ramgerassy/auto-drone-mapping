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
    """Save an RGB occupancy image with height-based shading.

    Color mapping:
        - Unknown (0.4 <= p <= 0.6): blue (70, 130, 180)
        - Free (p < 0.4): white (255, 255, 255)
        - Occupied (p > 0.6): grayscale intensity derived from
          height — darker = taller. Uses the max observed height
          in the grid as the reference ceiling.

    The image origin is bottom-left (row 0 = bottom of image),
    matching the world coordinate convention.

    Args:
        grid: The occupancy grid to save.
        path: Output file path (should end in .png).
    """
    probs = grid.probability()
    heights = grid.height

    h, w = probs.shape
    pixels = np.zeros((h, w, 3), dtype=np.uint8)

    # Unknown cells: blue
    unknown = (probs >= 0.4) & (probs <= 0.6)
    pixels[unknown] = [70, 130, 180]

    # Free cells: white
    free = probs < 0.4
    pixels[free] = [255, 255, 255]

    # Occupied cells: height-based grayscale (dark = tall, light = short)
    occupied = probs > 0.6

    # Find max observed height for normalization, avoiding -inf cells
    finite_heights = heights[np.isfinite(heights)]
    max_height = float(finite_heights.max()) if finite_heights.size > 0 else 1.0
    max_height = max(max_height, 0.01)  # avoid division by zero

    # Normalize heights to [0, 1], then map to intensity [200, 0]
    # (200 = short/light gray, 0 = tallest/black)
    occ_heights = np.clip(heights[occupied], 0.0, max_height)
    intensity = 200 - (occ_heights / max_height * 200).astype(np.uint8)
    pixels[occupied, 0] = intensity
    pixels[occupied, 1] = intensity
    pixels[occupied, 2] = intensity

    # Flip vertically so row 0 (y=0) is at bottom of image
    pixels = np.flipud(pixels)

    img = Image.fromarray(pixels, mode="RGB")
    img.save(str(path))
