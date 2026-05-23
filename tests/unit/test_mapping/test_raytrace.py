"""Tests for Bresenham 2D ray tracing."""

from __future__ import annotations

from swarm_mapping.mapping.raytrace import bresenham_2d


class TestBresenham2D:
    """Tests for bresenham_2d."""

    def test_horizontal_line(self) -> None:
        """Horizontal line visits all cells along x."""
        cells = bresenham_2d(0, 0, 5, 0)
        assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

    def test_vertical_line(self) -> None:
        """Vertical line visits all cells along y."""
        cells = bresenham_2d(0, 0, 0, 4)
        assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_diagonal_line(self) -> None:
        """45-degree diagonal visits cells along both axes."""
        cells = bresenham_2d(0, 0, 3, 3)
        assert len(cells) == 4
        assert cells[0] == (0, 0)
        assert cells[-1] == (3, 3)

    def test_same_cell(self) -> None:
        """Start equals end returns a single cell."""
        cells = bresenham_2d(2, 3, 2, 3)
        assert cells == [(2, 3)]

    def test_negative_coordinates(self) -> None:
        """Works with negative grid coordinates."""
        cells = bresenham_2d(-2, -1, 1, 2)
        assert cells[0] == (-2, -1)
        assert cells[-1] == (1, 2)
        # All cells should be unique and ordered
        assert len(cells) == len(set(cells))

    def test_reversed_direction(self) -> None:
        """Line traced in reverse visits same cells in reverse order."""
        forward = bresenham_2d(0, 0, 5, 3)
        backward = bresenham_2d(5, 3, 0, 0)
        assert forward == list(reversed(backward))

    def test_steep_line(self) -> None:
        """Steep line (dy > dx) visits correct cells."""
        cells = bresenham_2d(0, 0, 1, 4)
        assert cells[0] == (0, 0)
        assert cells[-1] == (1, 4)
        assert len(cells) == 5
