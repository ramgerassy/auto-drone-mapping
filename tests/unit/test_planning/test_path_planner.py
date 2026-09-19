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
    return AStarPlanner(clearance_radius=0.0)


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
        p: PathPlanner = AStarPlanner(clearance_radius=0.0)
        assert callable(p.plan)


class TestObstacleClearance:
    """Feature 4c — the drone has a body, so a path must leave room for it.

    `clearance_radius` is the drone's half-extent plus any safety margin, in
    metres. An occupied cell at Chebyshev distance `k` has its near face at
    `(k - 0.5) * resolution`, so no overlap requires
    `(k - 0.5) * res >= clearance_radius`; the planner blocks everything nearer.

    These grids use 1 m cells, so a 1.0 m clearance gives an inflation radius of
    exactly one cell — `k_min = ceil(1.0/1.0 + 0.5) = 2`, so `r = 1`.
    """

    ONE_CELL = 1.0  # clearance_radius giving r = 1 on a 1 m grid

    def blocked_grid(self) -> OccupancyGrid:
        """An 8x8 free grid with a single obstacle at (4, 4)."""
        grid = make_grid()
        fill_free(grid)
        grid.log_odds[4, 4] = OCC
        return grid

    @staticmethod
    def hugs(path: list[tuple[int, int]], obstacle: tuple[int, int]) -> bool:
        """True if any path cell is Chebyshev-adjacent to the obstacle."""
        o_col, o_row = obstacle
        return any(max(abs(col - o_col), abs(row - o_row)) <= 1 for col, row in path)

    @pytest.mark.sanity
    def test_zero_clearance_still_hugs_the_obstacle(self) -> None:
        """Case 1: the control — with clearance off, nothing changes.

        The pre-4c planner routes straight past the obstacle's face. This pins
        that `clearance_radius` is what moves the path in the next test, rather
        than some unrelated change to the search.
        """
        grid = self.blocked_grid()
        path = AStarPlanner(clearance_radius=0.0).plan(grid, (1, 4), (7, 4))

        assert path is not None
        assert self.hugs(path, (4, 4))

    def test_clearance_pushes_the_path_off_the_obstacle(self) -> None:
        """Case 2: with a one-cell clearance the path keeps its distance."""
        grid = self.blocked_grid()
        path = AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, (1, 4), (7, 4))

        assert path is not None
        assert not self.hugs(path, (4, 4))
        assert_contiguous(path)

    def corridor_grid(self, free_rows: range) -> OccupancyGrid:
        """Two open rooms joined by a corridor of the given free rows.

        Rooms occupy cols 0-2 and 8-10 and are fully free; the wall spans
        cols 3-7 with only `free_rows` left open.
        """
        grid = make_grid(width=11, height=7)
        fill_free(grid)
        for col in range(3, 8):
            for row in range(7):
                if row not in free_rows:
                    grid.log_odds[row, col] = OCC
        return grid

    def test_corridor_narrower_than_the_body_is_impassable(self) -> None:
        """Case 3: a 2-cell corridor needs 2r+1 = 3, so the route is refused.

        Both free rows touch a wall, so both are inflated away. Returning None
        is correct — the drone genuinely does not fit — and is what makes the
        planner stop teleporting bodies through walls.
        """
        grid = self.corridor_grid(range(3, 5))  # rows 3-4, two cells tall

        assert AStarPlanner(clearance_radius=0.0).plan(grid, (1, 3), (9, 3)) is not None
        assert (
            AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, (1, 3), (9, 3))
            is None
        )

    def test_corridor_of_exactly_two_r_plus_one_still_passes(self) -> None:
        """Case 4: a 3-cell corridor passes, down its centre line only.

        Pins the radius from ABOVE. Without this, `r` could grow and quietly
        make legal corridors impassable — the failure direction that would look
        like "the map is just unexplorable" rather than like a bug.
        """
        grid = self.corridor_grid(range(2, 5))  # rows 2-4, three cells tall
        path = AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, (1, 3), (9, 3))

        assert path is not None
        corridor_rows = {row for col, row in path if 3 <= col <= 7}
        assert corridor_rows == {3}  # the centre line, the only legal one

    def test_unknown_cells_are_not_inflated(self) -> None:
        """Case 5: inflating unknown would halt exploration on tick 1.

        A frontier is by definition a free cell adjacent to unknown space. If
        unknown were inflated too, every frontier would be unreachable and the
        mission would complete immediately over an empty map.
        """
        grid = make_grid()
        fill_free(grid)
        for row in range(8):  # a band of unmapped space
            grid.log_odds[row, 6] = UNK
        goal = (5, 4)  # free, but directly against the unknown band

        path = AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, (1, 4), goal)

        assert path is not None
        assert path[-1] == goal

    def test_start_inside_the_inflated_zone_can_still_escape(self) -> None:
        """Case 6: the liveness escape hatch.

        A drone that discovers a wall beside itself is suddenly inside its own
        inflated zone. Without this exemption `plan()` returns None, the drone
        gets no assignment, and if that holds for every drone the mission
        reports completion over a half-unknown map. Exempting the start is safe
        — the drone is already standing there.
        """
        grid = make_grid()
        fill_free(grid)
        grid.log_odds[1, 1] = OCC
        start = (1, 2)  # free, but Chebyshev-adjacent to the obstacle

        path = AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, start, (6, 6))

        assert path is not None
        assert path[0] == start
        assert not self.hugs(path[1:], (1, 1))  # it leaves and does not return

    def test_goal_inside_the_inflated_zone_is_unreachable(self) -> None:
        """Case 7: the goal is not exempt — the drone would not fit there.

        Correct rather than unfortunate: coverage counts cells *mapped*, not
        cells *visited*, and the sensor maps that frontier from a cell the
        drone can legally occupy.
        """
        grid = make_grid()
        fill_free(grid)
        grid.log_odds[1, 1] = OCC

        assert (
            AStarPlanner(clearance_radius=self.ONE_CELL).plan(grid, (6, 6), (1, 2))
            is None
        )

    def test_inflation_is_deterministic(self) -> None:
        """Case 8: same grid and radius, same path."""
        grid = self.blocked_grid()
        planner = AStarPlanner(clearance_radius=self.ONE_CELL)

        assert planner.plan(grid, (1, 4), (7, 4)) == planner.plan(grid, (1, 4), (7, 4))
