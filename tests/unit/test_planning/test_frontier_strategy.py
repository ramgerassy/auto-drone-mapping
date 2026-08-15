"""Tests for the frontier selection strategy (planning.frontier_strategy).

Pure-logic on hand-built OccupancyGrids — no MuJoCo. Cell classes are set
directly via log_odds: -2 => free, +2 => occupied, 0 => unknown.

Scoring is tested against a StubPlanner so selection logic is exercised
independently of A*; the wall-awareness tests use the real AStarPlanner
because the whole point there is that walls reorder the candidates.

Cells are (col, row); the log_odds array is [row, col]. With resolution 1.0
and origin (0, 0), cell (c, r) has world centre (c + 0.5, r + 0.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.frontier_strategy import (
    FrontierAssignment,
    FrontierStrategy,
    NearestFrontier,
)
from swarm_mapping.planning.path_planner import AStarPlanner

pytestmark = pytest.mark.sprint(2)  # Feature 3 — frontier strategy

FREE, OCC, UNK = -2.0, 2.0, 0.0

Cell = tuple[int, int]


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


def region(col: int, row: int, size: int = 3) -> FrontierRegion:
    """A single-cell-representative frontier region centred on that cell."""
    return FrontierRegion(centroid=(col + 0.5, row + 0.5), cell=(col, row), size=size)


def straight(start: Cell, goal_col: int) -> list[Cell]:
    """A horizontal path from start rightwards to goal_col (orthogonal steps)."""
    col, row = start
    return [(c, row) for c in range(col, goal_col + 1)]


class StubPlanner:
    """A PathPlanner returning canned paths, so scoring is tested in isolation.

    Maps a goal cell to the path to return; an absent goal (or a None value)
    means unreachable.
    """

    def __init__(self, paths: dict[Cell, list[Cell] | None]) -> None:
        self._paths = paths
        self.calls: list[Cell] = []

    def plan(self, grid: OccupancyGrid, start: Cell, goal: Cell) -> list[Cell] | None:
        """Return the canned path for `goal`, or None if unreachable."""
        self.calls.append(goal)
        return self._paths.get(goal)


@pytest.fixture
def free_grid() -> OccupancyGrid:
    """An 8x8 all-free grid."""
    grid = make_grid()
    fill_free(grid)
    return grid


class TestSelectionCore:
    """Choosing the cheapest reachable frontier."""

    @pytest.mark.sanity
    def test_single_reachable_frontier_is_selected(
        self, free_grid: OccupancyGrid
    ) -> None:
        """One reachable frontier is returned, with its path and cost."""
        target = region(3, 0)
        planner = StubPlanner({(3, 0): straight((0, 0), 3)})
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [target], (0, 0))

        assert result is not None
        assert result.region == target
        assert result.path == [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert result.cost == 30  # three orthogonal steps

    def test_cheaper_frontier_wins(self, free_grid: OccupancyGrid) -> None:
        """Of two reachable frontiers, the cheaper-to-reach one is chosen."""
        near, far = region(2, 0), region(6, 0)
        planner = StubPlanner(
            {(2, 0): straight((0, 0), 2), (6, 0): straight((0, 0), 6)}
        )
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [far, near], (0, 0))

        assert result is not None
        assert result.region == near
        assert result.cost == 20

    def test_no_frontiers_returns_none(self, free_grid: OccupancyGrid) -> None:
        """An empty frontier list means nothing left to explore."""
        strategy = NearestFrontier(StubPlanner({}))
        assert strategy.select(free_grid, [], (0, 0)) is None

    def test_all_unreachable_returns_none(self, free_grid: OccupancyGrid) -> None:
        """Every frontier walled off -> no assignment."""
        planner = StubPlanner({(2, 0): None, (6, 0): None})
        strategy = NearestFrontier(planner)
        assert strategy.select(free_grid, [region(2, 0), region(6, 0)], (0, 0)) is None

    def test_unreachable_skipped_in_favour_of_reachable(
        self, free_grid: OccupancyGrid
    ) -> None:
        """A nearer-but-unreachable frontier loses to a farther reachable one."""
        blocked, open_target = region(1, 0), region(6, 0)
        planner = StubPlanner({(1, 0): None, (6, 0): straight((0, 0), 6)})
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [blocked, open_target], (0, 0))

        assert result is not None
        assert result.region == open_target

    def test_path_spans_start_to_region_cell(self, free_grid: OccupancyGrid) -> None:
        """The returned path starts at the drone and ends on the frontier cell."""
        target = region(4, 0)
        planner = StubPlanner({(4, 0): straight((0, 0), 4)})
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [target], (0, 0))

        assert result is not None
        assert result.path[0] == (0, 0)
        assert result.path[-1] == target.cell

    def test_cost_counts_diagonal_steps_as_fourteen(
        self, free_grid: OccupancyGrid
    ) -> None:
        """Cost uses the planner's 10/14 integer units, not step count."""
        target = region(2, 2)
        planner = StubPlanner({(2, 2): [(0, 0), (1, 1), (2, 2)]})
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [target], (0, 0))

        assert result is not None
        assert result.cost == 28  # two diagonal steps


class TestWallAwareness:
    """Real A*: walls, not straight-line distance, decide the winner."""

    def test_euclidean_near_frontier_behind_wall_loses(self) -> None:
        """A frontier 3 m away behind a wall loses to one 4 m down open space.

        This is the test that would fail under straight-line scoring — it is
        the reason the strategy scores by true path cost (Decision D1).
        """
        grid = make_grid()
        fill_free(grid)
        for r in range(7):  # wall at col 2, rows 0..6; gap at row 7
            grid.log_odds[r, 2] = OCC

        start = (0, 3)
        behind_wall = region(3, 3)  # centroid 3.0 m from the drone
        down_corridor = region(0, 7)  # centroid 4.0 m from the drone
        strategy = NearestFrontier(AStarPlanner())

        result = strategy.select(grid, [down_corridor, behind_wall], start)

        assert result is not None
        assert result.region == down_corridor

    def test_walled_off_frontier_is_never_assigned(self) -> None:
        """A fully enclosed frontier is unreachable and skipped."""
        grid = make_grid()
        fill_free(grid)
        for r in range(8):  # full wall at col 4 splits the grid
            grid.log_odds[r, 4] = OCC

        strategy = NearestFrontier(AStarPlanner())
        result = strategy.select(grid, [region(6, 6)], (0, 0))

        assert result is None


class TestSpreadingPenalty:
    """Hard exclusion of claimed regions plus a soft proximity penalty."""

    def test_claimed_region_is_never_reselected(self, free_grid: OccupancyGrid) -> None:
        """A claimed frontier is excluded even when it is much cheaper."""
        taken, other = region(1, 0), region(6, 0)
        planner = StubPlanner(
            {(1, 0): straight((0, 0), 1), (6, 0): straight((0, 0), 6)}
        )
        strategy = NearestFrontier(planner)

        result = strategy.select(free_grid, [taken, other], (0, 0), claimed=[taken])

        assert result is not None
        assert result.region == other

    def test_candidate_near_claimed_loses_to_equal_cost_far_candidate(
        self, free_grid: OccupancyGrid
    ) -> None:
        """Equal cost: the one crowding a claimed frontier is penalized away."""
        taken = region(1, 1)
        crowding, spread_out = region(2, 1), region(7, 7)
        planner = StubPlanner(
            {(2, 1): straight((0, 1), 2), (7, 7): straight((5, 7), 7)}
        )
        strategy = NearestFrontier(planner, spread_radius=2.0, spread_penalty=50)

        result = strategy.select(
            free_grid, [crowding, spread_out], (0, 1), claimed=[taken]
        )

        assert result is not None
        assert result.region == spread_out

    def test_penalized_candidate_still_selected_when_only_option(
        self, free_grid: OccupancyGrid
    ) -> None:
        """The penalty never strands a frontier — last one left is still taken."""
        taken, crowding = region(1, 1), region(2, 1)
        planner = StubPlanner({(2, 1): straight((0, 1), 2)})
        strategy = NearestFrontier(planner, spread_radius=2.0, spread_penalty=50)

        result = strategy.select(free_grid, [crowding], (0, 1), claimed=[taken])

        assert result is not None
        assert result.region == crowding
        assert result.cost == 20  # reported cost is the true path cost, unpenalized

    def test_zero_radius_disables_proximity_penalty(
        self, free_grid: OccupancyGrid
    ) -> None:
        """spread_radius=0 leaves only hard exclusion; the near candidate wins."""
        taken = region(1, 1)
        crowding, spread_out = region(2, 1), region(7, 7)
        planner = StubPlanner(
            {(2, 1): straight((0, 1), 2), (7, 7): straight((5, 7), 7)}
        )
        strategy = NearestFrontier(planner)  # defaults: radius 0, penalty 0

        result = strategy.select(
            free_grid, [crowding, spread_out], (0, 1), claimed=[taken]
        )

        assert result is not None
        assert result.region == crowding

    def test_default_claimed_is_empty(self, free_grid: OccupancyGrid) -> None:
        """Omitting `claimed` behaves as no claims at all."""
        target = region(2, 1)
        planner = StubPlanner({(2, 1): straight((0, 1), 2)})
        strategy = NearestFrontier(planner, spread_radius=2.0, spread_penalty=50)

        assert strategy.select(free_grid, [target], (0, 1)) is not None


class TestDeterminismAndContract:
    """Determinism is a hard requirement; the strategy is stateless."""

    def test_equal_cost_tie_broken_by_row_then_col(
        self, free_grid: OccupancyGrid
    ) -> None:
        """Equal-cost candidates tie-break on (row, col), not list order."""
        lower_row, higher_row = region(3, 1), region(5, 2)
        planner = StubPlanner(
            {(3, 1): straight((1, 1), 3), (5, 2): straight((3, 2), 5)}
        )
        strategy = NearestFrontier(planner)

        # Passed out of sorted order on purpose: first-in-list must not win.
        result = strategy.select(free_grid, [higher_row, lower_row], (1, 1))

        assert result is not None
        assert result.region == lower_row

    def test_repeated_calls_are_identical(self, free_grid: OccupancyGrid) -> None:
        """Same inputs -> same assignment, every call."""
        frontiers = [region(2, 0), region(6, 0)]
        planner = StubPlanner(
            {(2, 0): straight((0, 0), 2), (6, 0): straight((0, 0), 6)}
        )
        strategy = NearestFrontier(planner)

        first = strategy.select(free_grid, frontiers, (0, 0))
        second = strategy.select(free_grid, frontiers, (0, 0))

        assert first == second

    def test_select_does_not_mutate_inputs(self, free_grid: OccupancyGrid) -> None:
        """The strategy is stateless: no input is modified."""
        frontiers = [region(2, 0), region(6, 0)]
        claimed = [region(6, 0)]
        frontiers_before = list(frontiers)
        claimed_before = list(claimed)
        grid_before = free_grid.log_odds.copy()
        planner = StubPlanner(
            {(2, 0): straight((0, 0), 2), (6, 0): straight((0, 0), 6)}
        )

        NearestFrontier(planner).select(free_grid, frontiers, (0, 0), claimed=claimed)

        assert frontiers == frontiers_before
        assert claimed == claimed_before
        assert np.array_equal(free_grid.log_odds, grid_before)


class TestProtocol:
    """NearestFrontier satisfies the FrontierStrategy protocol."""

    def test_nearest_frontier_is_a_frontier_strategy(self) -> None:
        """NearestFrontier is usable through the FrontierStrategy interface."""
        s: FrontierStrategy = NearestFrontier(AStarPlanner())
        assert callable(s.select)

    def test_assignment_is_frozen(self, free_grid: OccupancyGrid) -> None:
        """FrontierAssignment is immutable — the coordinator cannot edit it."""
        target = region(2, 0)
        planner = StubPlanner({(2, 0): straight((0, 0), 2)})
        result = NearestFrontier(planner).select(free_grid, [target], (0, 0))

        assert isinstance(result, FrontierAssignment)
        with pytest.raises(AttributeError):
            result.cost = 0  # type: ignore[misc]

    def test_frontiers_accepts_any_sequence(self, free_grid: OccupancyGrid) -> None:
        """The signature takes a Sequence, so a tuple works as well as a list."""
        frontiers: Sequence[FrontierRegion] = (region(2, 0),)
        planner = StubPlanner({(2, 0): straight((0, 0), 2)})

        assert NearestFrontier(planner).select(free_grid, frontiers, (0, 0)) is not None
