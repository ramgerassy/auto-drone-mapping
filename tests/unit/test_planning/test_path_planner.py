"""Tests for the A* path planner (planning.path_planner).

Pure-logic on hand-built OccupancyGrids — no MuJoCo. Cell classes are set
directly via log_odds: -2 => free (traversable), +2 => occupied (blocked),
0 => unknown (blocked — the drone never plans through unmapped space).
Movement is 8-connected with an octile heuristic and no corner-cutting.

Cells are (col, row); the log_odds array is [row, col].
"""

from __future__ import annotations

import pytest

from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.path_planner import AStarPlanner, PathPlanner

pytestmark = pytest.mark.sprint(2)  # Feature 2 — A* path planner

FREE, OCC, UNK = -2.0, 2.0, 0.0


def make_grid(width: int = 8, height: int = 8) -> OccupancyGrid:
    """A width x height grid, 1 m cells, origin (0, 0), all-unknown (blocked)."""
    return OccupancyGrid(
        MapConfig(
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            grid_width=width,
            grid_height=height,
        )
    )


def fill_free(grid: OccupancyGrid) -> None:
    """Mark every cell free (traversable)."""
    grid.log_odds[:] = FREE


def assert_contiguous(path: list[tuple[int, int]]) -> None:
    """Every consecutive pair is 8-adjacent (a legal single move)."""
    for (c0, r0), (c1, r1) in zip(path, path[1:], strict=False):
        assert max(abs(c1 - c0), abs(r1 - r0)) == 1


def all_free(grid: OccupancyGrid, path: list[tuple[int, int]]) -> bool:
    """True if every cell in the path is free (traversable)."""
    prob = grid.probability()
    return all(prob[row, col] < 0.4 for col, row in path)


@pytest.fixture
def planner() -> AStarPlanner:
    """A default A* planner (8-connected, octile, no corner-cutting)."""
    return AStarPlanner()


class TestAStarPlanner:
    """A* pathfinding on the occupancy grid."""

    @pytest.mark.sanity
    def test_straight_path_on_free_row(self, planner: AStarPlanner) -> None:
        """A clear horizontal corridor yields the direct contiguous path."""
        g = make_grid()
        for c in range(8):
            g.log_odds[3, c] = FREE  # free row at row=3
        assert planner.plan(g, (0, 3), (5, 3)) == [(c, 3) for c in range(6)]

    def test_diagonal_shortcut_on_open_ground(self, planner: AStarPlanner) -> None:
        """On open ground the drone cuts diagonally (8-connectivity)."""
        g = make_grid()
        fill_free(g)
        assert planner.plan(g, (0, 0), (3, 3)) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_optimal_corner_to_corner_is_pure_diagonal(
        self, planner: AStarPlanner
    ) -> None:
        """Open-ground corner-to-corner is the unique 7-step diagonal."""
        g = make_grid()
        fill_free(g)
        assert planner.plan(g, (0, 0), (7, 7)) == [(i, i) for i in range(8)]

    def test_detour_around_wall(self, planner: AStarPlanner) -> None:
        """A wall with a gap forces a detour; the path stays on free cells."""
        g = make_grid()
        fill_free(g)
        for r in range(6):  # wall at col 4, rows 0..5; gap at rows 6-7
            g.log_odds[r, 4] = OCC
        path = planner.plan(g, (2, 2), (6, 2))
        assert path is not None
        assert path[0] == (2, 2)
        assert path[-1] == (6, 2)
        assert all_free(g, path)
        assert_contiguous(path)

    def test_unreachable_returns_none(self, planner: AStarPlanner) -> None:
        """A full wall separating start and goal makes the goal unreachable."""
        g = make_grid()
        fill_free(g)
        for r in range(8):  # full wall at col 4
            g.log_odds[r, 4] = OCC
        assert planner.plan(g, (2, 2), (6, 6)) is None

    def test_start_equals_goal(self, planner: AStarPlanner) -> None:
        """Start == goal returns a single-cell path."""
        g = make_grid()
        g.log_odds[3, 3] = FREE
        assert planner.plan(g, (3, 3), (3, 3)) == [(3, 3)]

    def test_goal_occupied_returns_none(self, planner: AStarPlanner) -> None:
        """A goal on an occupied cell is unreachable."""
        g = make_grid()
        fill_free(g)
        g.log_odds[5, 5] = OCC
        assert planner.plan(g, (0, 0), (5, 5)) is None

    def test_path_only_traverses_free_cells(self, planner: AStarPlanner) -> None:
        """Unknown cells are blocked; the path uses only free cells."""
        g = make_grid()
        for c in range(6):  # an L-shaped free corridor; rest stays unknown
            g.log_odds[1, c] = FREE
        for r in range(1, 6):
            g.log_odds[r, 5] = FREE
        path = planner.plan(g, (0, 1), (5, 5))
        assert path is not None
        assert all_free(g, path)
        assert_contiguous(path)

    def test_no_corner_cutting_between_two_walls(self, planner: AStarPlanner) -> None:
        """A diagonal squeezing between two occupied orthogonals is disallowed."""
        g = make_grid()
        g.log_odds[0, 0] = FREE  # (0,0)
        g.log_odds[1, 1] = FREE  # (1,1)
        g.log_odds[0, 1] = OCC  # (1,0) — shared orthogonal
        g.log_odds[1, 0] = OCC  # (0,1) — shared orthogonal
        assert planner.plan(g, (0, 0), (1, 1)) is None

    def test_no_corner_cutting_single_wall_goes_around(
        self, planner: AStarPlanner
    ) -> None:
        """One blocked shared orthogonal: the diagonal is refused, route around."""
        g = make_grid()
        g.log_odds[0, 0] = FREE  # (0,0)
        g.log_odds[1, 1] = FREE  # (1,1)
        g.log_odds[0, 1] = OCC  # (1,0) blocked
        g.log_odds[1, 0] = FREE  # (0,1) free
        assert planner.plan(g, (0, 0), (1, 1)) == [(0, 0), (0, 1), (1, 1)]

    def test_deterministic_across_calls(self, planner: AStarPlanner) -> None:
        """Same grid + endpoints -> identical path every call."""
        g = make_grid()
        fill_free(g)
        g.log_odds[3, 3] = OCC
        assert planner.plan(g, (0, 0), (7, 7)) == planner.plan(g, (0, 0), (7, 7))


class TestProtocol:
    """AStarPlanner satisfies the PathPlanner protocol."""

    def test_astar_is_a_pathplanner(self) -> None:
        """AStarPlanner is usable through the PathPlanner interface."""
        p: PathPlanner = AStarPlanner()
        assert callable(p.plan)
