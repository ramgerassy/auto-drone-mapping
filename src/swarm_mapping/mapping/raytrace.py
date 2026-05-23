"""2D ray tracing through a grid using Bresenham's line algorithm."""

from __future__ import annotations


def bresenham_2d(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Trace a line between two grid cells and return all cells visited.

    Uses Bresenham's line algorithm (integer arithmetic only) to
    determine the ordered sequence of grid cells that a line from
    (x0, y0) to (x1, y1) passes through.

    Args:
        x0: Start column.
        y0: Start row.
        x1: End column.
        y1: End row.

    Returns:
        Ordered list of (col, row) tuples from start to end, inclusive.
    """
    cells: list[tuple[int, int]] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0

    while True:
        cells.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return cells
