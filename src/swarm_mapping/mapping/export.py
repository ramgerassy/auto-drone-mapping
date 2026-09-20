"""Export occupancy grid as .npz data and .png image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
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


def save_png(grid: OccupancyGrid, path: str | Path, max_height: float = 3.0) -> None:
    """Save an RGB occupancy image with height-based shading.

    Color mapping:
        - Unknown (0.4 <= p <= 0.6): blue (70, 130, 180)
        - Free (p < 0.4): white (255, 255, 255)
        - Occupied (p > 0.6): grayscale intensity derived from
          height — darker = taller, relative to ``max_height``.

    The image origin is bottom-left (row 0 = bottom of image),
    matching the world coordinate convention.

    Args:
        grid: The occupancy grid to save.
        path: Output file path (should end in .png).
        max_height: Reference ceiling height in meters. Obstacles
            at this height render as black; shorter obstacles are
            lighter. Default 3.0 (typical indoor ceiling).
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
    ref_height = max(max_height, 0.01)  # avoid division by zero

    # Normalize heights to [0, 1], then map to intensity [200, 0]
    # (200 = short/light gray, 0 = at or above max_height/black)
    occ_heights = np.clip(heights[occupied], 0.0, ref_height)
    intensity = 200 - (occ_heights / ref_height * 200).astype(np.uint8)
    pixels[occupied, 0] = intensity
    pixels[occupied, 1] = intensity
    pixels[occupied, 2] = intensity

    # Flip vertically so row 0 (y=0) is at bottom of image
    pixels = np.flipud(pixels)

    img = Image.fromarray(pixels, mode="RGB")
    img.save(str(path))


def save_visit_heatmap(
    visits: NDArray[np.int_],
    grid: OccupancyGrid,
    path: str | Path,
) -> None:
    """Save a per-cell visit-count image for one drone, over the map.

    A diagnostic, not a deliverable: it answers "where did this drone actually
    spend its time, and how often did it retread the same cell", which is the
    question a coverage percentage cannot. Heavy repeat visits show up as hot
    spots, and their position relative to the walls is what makes the picture
    worth having — so the occupancy map is drawn underneath rather than the
    counts alone.

    Colours:
        - occupied cells: dark grey, for orientation
        - unknown: pale blue, matching `save_png`
        - visited 0 times: white
        - visited 1+ times: yellow through red, on a **log** ramp

    The ramp is logarithmic because visit counts are heavy-tailed — a drone
    parked at its base accrues hundreds of visits to one cell while a corridor
    it swept once has a single visit, and a linear ramp would render everything
    except the parking spot as the same white.

    The image origin is bottom-left (row 0 = bottom), matching `save_png` and
    the world coordinate convention.

    Args:
        visits: Per-cell visit counts, shape (height, width), indexed
            [row, col].
        grid: The occupancy grid, drawn underneath for context.
        path: Destination PNG path.
    """
    prob = grid.probability()
    height, width = visits.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    rgb[...] = (255, 255, 255)  # free and unvisited
    rgb[(prob >= 0.4) & (prob <= 0.6)] = (214, 228, 240)  # unknown
    rgb[prob > 0.6] = (70, 70, 78)  # occupied

    seen = visits > 0
    if np.any(seen):
        # log1p keeps a single visit distinguishable from a hundred without
        # the hundred flattening everything else.
        weight = np.log1p(visits.astype(np.float64))
        weight = weight / weight.max()
        hot = np.zeros((height, width, 3), dtype=np.uint8)
        hot[..., 0] = 255  # red channel is full across the ramp
        hot[..., 1] = (255 * (1.0 - weight)).astype(np.uint8)  # yellow -> red
        rgb[seen] = hot[seen]

    Image.fromarray(np.flipud(rgb), mode="RGB").save(str(path))
