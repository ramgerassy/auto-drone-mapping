"""Frontier detection on the occupancy grid.

A *frontier* is the boundary between known-free and unknown space — the
natural target for exploration. This module finds free cells that border
unknown cells, groups them into connected regions, and returns each region's
world-coordinate centroid, a representative free cell (a valid path-planning
goal), and its size.

Detection lives in ``mapping`` (not ``planning``) so the dependency direction
stays one-way: ``planning`` reads these regions, ``mapping`` never imports
``planning``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from swarm_mapping.mapping.grid import OccupancyGrid

# 8-connectivity offsets as (d_col, d_row), used to cluster frontier cells.
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class FrontierRegion:
    """A connected cluster of frontier cells.

    Attributes:
        centroid: World (x, y) of the region's mean cell center — used for
            display and the coordinator's spatial spreading penalty.
        cell: A representative (col, row) frontier cell nearest the centroid.
            Guaranteed free (hence traversable), so it is a valid A* goal even
            when the centroid itself falls on an unknown or occupied cell.
        size: Number of frontier cells in the region.
    """

    centroid: tuple[float, float]
    cell: tuple[int, int]
    size: int


def frontier_cells(
    grid: OccupancyGrid,
    *,
    free_threshold: float = 0.4,
    occ_threshold: float = 0.6,
) -> list[tuple[int, int]]:
    """Return the frontier cells of a grid.

    A frontier cell is a *free* cell (occupancy probability below
    ``free_threshold``) with at least one 4-connected neighbor that is
    *unknown* (probability between the thresholds). Out-of-bounds neighbors are
    not treated as unknown, so cells on the grid edge are not frontiers merely
    for touching the boundary.

    Args:
        grid: The occupancy grid to analyze.
        free_threshold: Probability below which a cell counts as free.
        occ_threshold: Probability above which a cell counts as occupied;
            cells between the thresholds are unknown.

    Returns:
        Frontier cells as ``(col, row)`` tuples, sorted row-major.
    """
    prob = grid.probability()
    free = prob < free_threshold
    unknown = (prob >= free_threshold) & (prob <= occ_threshold)

    # For each cell, does any in-bounds 4-neighbor hold unknown space? Slicing
    # drops off-grid neighbors, so the grid edge never counts as unknown.
    neighbor_unknown = np.zeros_like(unknown)
    neighbor_unknown[1:, :] |= unknown[:-1, :]  # neighbor above
    neighbor_unknown[:-1, :] |= unknown[1:, :]  # neighbor below
    neighbor_unknown[:, 1:] |= unknown[:, :-1]  # neighbor left
    neighbor_unknown[:, :-1] |= unknown[:, 1:]  # neighbor right

    frontier = free & neighbor_unknown
    # argwhere yields [row, col] pairs in row-major order.
    return [(int(col), int(row)) for row, col in np.argwhere(frontier)]


def detect_frontiers(
    grid: OccupancyGrid,
    *,
    free_threshold: float = 0.4,
    occ_threshold: float = 0.6,
    min_region_size: int = 2,
) -> list[FrontierRegion]:
    """Detect and cluster frontier regions in a grid.

    Finds frontier cells (see :func:`frontier_cells`), groups them into
    8-connected regions, drops regions smaller than ``min_region_size``, and
    returns them sorted deterministically by representative cell ``(row, col)``.

    Args:
        grid: The occupancy grid to analyze.
        free_threshold: Probability below which a cell counts as free.
        occ_threshold: Probability above which a cell counts as occupied.
        min_region_size: Regions with fewer cells than this are discarded as
            noise.

    Returns:
        Frontier regions, sorted by representative cell ``(row, col)``.
    """
    cells = frontier_cells(
        grid, free_threshold=free_threshold, occ_threshold=occ_threshold
    )
    cell_set = set(cells)
    visited: set[tuple[int, int]] = set()
    regions: list[FrontierRegion] = []

    # Iterate cells in row-major order (already sorted) for a stable clustering.
    for start in cells:
        if start in visited:
            continue

        cluster: list[tuple[int, int]] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            col, row = queue.popleft()
            cluster.append((col, row))
            for d_col, d_row in _NEIGHBORS_8:
                neighbor = (col + d_col, row + d_row)
                if neighbor in cell_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(cluster) < min_region_size:
            continue

        world = [grid.grid_to_world(col, row) for col, row in cluster]
        centroid_x = sum(wx for wx, _ in world) / len(world)
        centroid_y = sum(wy for _, wy in world) / len(world)

        # Representative = frontier cell nearest the centroid; ties broken by
        # (row, col) for determinism. Reuse the world coords computed above
        # rather than re-projecting every cell a second time.
        def rank(
            item: tuple[tuple[int, int], tuple[float, float]],
            cx: float = centroid_x,
            cy: float = centroid_y,
        ) -> tuple[float, int, int]:
            (col, row), (wx, wy) = item
            return ((wx - cx) ** 2 + (wy - cy) ** 2, row, col)

        (rep_col, rep_row), _ = min(zip(cluster, world, strict=True), key=rank)
        regions.append(
            FrontierRegion(
                centroid=(centroid_x, centroid_y),
                cell=(rep_col, rep_row),
                size=len(cluster),
            )
        )

    regions.sort(key=lambda region: (region.cell[1], region.cell[0]))
    return regions
