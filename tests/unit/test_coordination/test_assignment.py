"""Tests for frontier assignment across the swarm (coordination.assignment).

Pure logic on hand-built OccupancyGrids — no MuJoCo. Cell classes are set
directly via log_odds: -2 => free, +2 => occupied, 0 => unknown.

Cells are (col, row); the log_odds array is [row, col].
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_mapping.coordination.assignment import (
    assign_all,
    is_assignment_valid,
    next_cell,
)
from swarm_mapping.coordination.types import Cell, DroneState
from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.frontier_strategy import (
    FrontierAssignment,
    NearestFrontier,
)
from swarm_mapping.planning.path_planner import AStarPlanner

pytestmark = pytest.mark.sprint(2)  # Feature 4 — centralized master

FREE, OCC = -2.0, 2.0
MAX_WAIT = 3


def make_grid(width: int = 12, height: int = 12) -> OccupancyGrid:
    """A width x height all-free grid, 1 m cells, origin (0, 0)."""
    grid = OccupancyGrid(
        MapConfig(
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            grid_width=width,
            grid_height=height,
        )
    )
    grid.log_odds[:] = FREE
    return grid


def region(col: int, row: int, size: int = 3) -> FrontierRegion:
    """A single-cell-representative frontier region centred on that cell."""
    return FrontierRegion(centroid=(col + 0.5, row + 0.5), cell=(col, row), size=size)


def straight(start: Cell, goal_col: int) -> list[Cell]:
    """A horizontal path from start rightwards to goal_col."""
    col, row = start
    return [(c, row) for c in range(col, goal_col + 1)]


def state(
    drone_id: int,
    cell: Cell,
    assignment: FrontierAssignment | None = None,
    path_index: int = 0,
    waited_ticks: int = 0,
) -> DroneState:
    """Build a DroneState with sensible defaults."""
    return DroneState(
        drone_id=drone_id,
        cell=cell,
        assignment=assignment,
        path_index=path_index,
        waited_ticks=waited_ticks,
    )


def assignment_to(col: int, row: int, start: Cell) -> FrontierAssignment:
    """An assignment flying from `start` rightwards to (col, row)."""
    path = straight(start, col)
    return FrontierAssignment(
        region=region(col, row), path=path, cost=10 * (len(path) - 1)
    )


PLANNER = AStarPlanner(clearance_radius=0.0)


@pytest.fixture
def strategy() -> NearestFrontier:
    """A NearestFrontier over a real A* planner."""
    return NearestFrontier(PLANNER)


def clear(grid: OccupancyGrid) -> NDArray[np.bool_]:
    """A clearance mask with nothing blocked (these tests use a point robot)."""
    return PLANNER.clearance_mask(grid)


class TestNextCell:
    """Reading the next step off a drone's path."""

    def test_none_when_idle(self) -> None:
        """An unassigned drone has no next cell."""
        assert next_cell(state(0, (0, 0))) is None

    def test_none_when_arrived(self) -> None:
        """At the last path index there is no further step."""
        a = assignment_to(3, 0, (0, 0))  # path length 4
        assert next_cell(state(0, (3, 0), a, path_index=3)) is None

    def test_returns_following_cell(self) -> None:
        """Mid-path, the next cell is one index along."""
        a = assignment_to(3, 0, (0, 0))
        assert next_cell(state(0, (1, 0), a, path_index=1)) == (2, 0)


class TestAssignmentValidity:
    """When a drone keeps flying its current target."""

    def test_valid_mid_path(self) -> None:
        """A normal in-progress assignment is kept."""
        grid = make_grid()
        a = assignment_to(4, 0, (0, 0))
        assert is_assignment_valid(
            state(0, (1, 0), a, 1), grid.probability(), 0.4, MAX_WAIT, clear(grid)
        )

    def test_invalid_when_arrived(self) -> None:
        """Reaching the end of the path releases the assignment."""
        grid = make_grid()
        a = assignment_to(4, 0, (0, 0))
        assert not is_assignment_valid(
            state(0, (4, 0), a, 4), grid.probability(), 0.4, MAX_WAIT, clear(grid)
        )

    def test_invalid_when_next_cell_became_occupied(self) -> None:
        """A newly discovered obstacle on the path forces a re-plan."""
        grid = make_grid()
        grid.log_odds[0, 2] = OCC  # cell (2, 0) now blocked
        a = assignment_to(4, 0, (0, 0))
        assert not is_assignment_valid(
            state(0, (1, 0), a, 1), grid.probability(), 0.4, MAX_WAIT, clear(grid)
        )

    def test_invalid_when_waited_too_long(self) -> None:
        """Exceeding the wait threshold triggers the deadlock escape."""
        grid = make_grid()
        a = assignment_to(4, 0, (0, 0))
        assert not is_assignment_valid(
            state(0, (1, 0), a, 1, waited_ticks=MAX_WAIT + 1),
            grid.probability(),
            0.4,
            MAX_WAIT,
            clear(grid),
        )


class TestAssignAll:
    """Handing frontiers out across the swarm."""

    @pytest.mark.sanity
    def test_idle_drone_receives_a_frontier(self, strategy: NearestFrontier) -> None:
        """An unassigned drone is given a reachable frontier and a path."""
        grid = make_grid()
        states = {0: state(0, (0, 0))}

        result = assign_all(grid, [region(5, 0)], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].assignment is not None
        assert result[0].assignment.region.cell == (5, 0)
        assert result[0].path_index == 0

    def test_two_drones_get_different_frontiers(
        self, strategy: NearestFrontier
    ) -> None:
        """The claimed list stops both drones targeting the same region."""
        grid = make_grid()
        states = {0: state(0, (0, 0)), 1: state(1, (0, 5))}

        result = assign_all(
            grid, [region(6, 0), region(6, 5)], states, strategy, PLANNER, MAX_WAIT
        )

        first = result[0].assignment
        second = result[1].assignment
        assert first is not None
        assert second is not None
        assert first.region.cell != second.region.cell

    def test_existing_assignment_is_kept(self, strategy: NearestFrontier) -> None:
        """An in-progress path survives the next pass — no re-planning."""
        grid = make_grid()
        a = assignment_to(6, 0, (0, 0))
        states = {0: state(0, (2, 0), a, path_index=2)}

        result = assign_all(grid, [region(6, 0)], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].assignment is a
        assert result[0].path_index == 2  # progress preserved

    def test_held_assignment_is_claimed_before_others_choose(
        self, strategy: NearestFrontier
    ) -> None:
        """A frontier already being flown to is not handed to a second drone.

        Drone 9 holds a path to (6, 0). Drone 0 sits right next to (6, 0) and
        would otherwise pick it as the cheapest — the two-pass claim is what
        prevents that.
        """
        grid = make_grid()
        held = assignment_to(6, 0, (0, 0))
        states = {
            9: state(9, (2, 0), held, path_index=2),
            0: state(0, (5, 0)),
        }

        result = assign_all(
            grid, [region(6, 0), region(0, 9)], states, strategy, PLANNER, MAX_WAIT
        )

        assert result[9].assignment is held
        assert result[0].assignment is not None
        assert result[0].assignment.region.cell != (6, 0)

    def test_reassigned_after_arriving(self, strategy: NearestFrontier) -> None:
        """A drone that arrived is given a fresh target."""
        grid = make_grid()
        arrived = assignment_to(3, 0, (0, 0))
        states = {0: state(0, (3, 0), arrived, path_index=3)}

        result = assign_all(grid, [region(9, 9)], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].assignment is not None
        assert result[0].assignment.region.cell == (9, 9)

    def test_no_frontiers_leaves_drone_idle(self, strategy: NearestFrontier) -> None:
        """Nothing left to explore is not an error."""
        grid = make_grid()
        states = {0: state(0, (0, 0))}

        result = assign_all(grid, [], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].assignment is None

    def test_unreachable_frontier_leaves_drone_idle(
        self, strategy: NearestFrontier
    ) -> None:
        """A walled-off frontier is not assigned."""
        grid = make_grid()
        for row in range(12):  # full wall at col 6
            grid.log_odds[row, 6] = OCC
        states = {0: state(0, (0, 0))}

        result = assign_all(grid, [region(9, 0)], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].assignment is None


class TestDeadlockEscape:
    """A drone blocked too long gives up on its frontier."""

    def test_assignment_dropped_after_max_wait(self, strategy: NearestFrontier) -> None:
        """Past the threshold, the held assignment is not kept."""
        grid = make_grid()
        stuck = assignment_to(6, 0, (0, 0))
        states = {0: state(0, (1, 0), stuck, 1, waited_ticks=MAX_WAIT + 1)}

        result = assign_all(grid, [region(6, 0)], states, strategy, PLANNER, MAX_WAIT)

        assert result[0].waited_ticks == 0  # counter reset on re-selection

    def test_reselects_a_different_frontier_when_available(
        self, strategy: NearestFrontier
    ) -> None:
        """After giving up, another drone's claim pushes it to a new target."""
        grid = make_grid()
        stuck = assignment_to(6, 0, (0, 0))
        # Drone 1 holds (6, 0); drone 0 has waited too long and must move on.
        states = {
            1: state(1, (5, 0), assignment_to(6, 0, (5, 0)), path_index=0),
            0: state(0, (1, 0), stuck, 1, waited_ticks=MAX_WAIT + 1),
        }

        result = assign_all(
            grid, [region(6, 0), region(1, 9)], states, strategy, PLANNER, MAX_WAIT
        )

        assert result[0].assignment is not None
        assert result[0].assignment.region.cell == (1, 9)


class TestDeterminism:
    """Assignment is reproducible."""

    def test_repeated_calls_identical(self, strategy: NearestFrontier) -> None:
        """Same grid, frontiers and states produce the same assignments."""
        grid = make_grid()
        states = {0: state(0, (0, 0)), 1: state(1, (0, 5))}
        frontiers = [region(6, 0), region(6, 5)]

        first = assign_all(grid, frontiers, states, strategy, PLANNER, MAX_WAIT)
        second = assign_all(grid, frontiers, states, strategy, PLANNER, MAX_WAIT)

        assert first == second

    def test_input_states_not_mutated(self, strategy: NearestFrontier) -> None:
        """States are replaced, never edited in place."""
        grid = make_grid()
        original = state(0, (0, 0))
        states = {0: original}

        assign_all(grid, [region(5, 0)], states, strategy, PLANNER, MAX_WAIT)

        assert states[0] is original
        assert original.assignment is None


class TestCommittedPathsAreRechecked:
    """A committed path must be re-checked against the planner's body model.

    Paths are deliberately kept rather than re-planned each tick, so this test
    is the only thing between a committed route and a changed map. It used to
    check `prob` alone, which cannot see clearance — and that is not an edge
    case: A* routes only through known-free cells, so a wall discovered later
    was *unknown* when the path was planned and is never itself on it. Only its
    inflation zone touches the path, over cells that were free and stay free.
    """

    def wall_scenario(self) -> tuple[OccupancyGrid, AStarPlanner, list[Cell]]:
        """Plan along row 1 while row 0 is still unmapped, then reveal a wall."""
        grid = OccupancyGrid(
            MapConfig(
                resolution=0.1,
                origin_x=0.0,
                origin_y=0.0,
                grid_width=12,
                grid_height=6,
            )
        )
        grid.log_odds[:] = FREE
        grid.log_odds[0, :] = 0.0  # unknown: no wall to inflate yet

        planner = AStarPlanner(clearance_radius=0.20)  # r = 2 at 0.1 m cells
        path = planner.plan(grid, (1, 1), (7, 1))
        assert path is not None  # the case is what we think it is

        grid.log_odds[0, :] = OCC  # the scan reveals a wall
        return grid, planner, path

    def held(self, path: list[Cell]) -> DroneState:
        """A drone one step into the given path."""
        return state(
            0,
            (1, 1),
            FrontierAssignment(
                region=region(7, 1), path=path, cost=10 * (len(path) - 1)
            ),
            path_index=0,
        )

    @pytest.mark.sanity
    def test_path_is_dropped_when_a_wall_appears_beside_it(self) -> None:
        """The whole point: the drone re-plans instead of flying into the jamb."""
        grid, planner, path = self.wall_scenario()

        keep = is_assignment_valid(
            self.held(path),
            grid.probability(),
            0.4,
            MAX_WAIT,
            planner.clearance_mask(grid),
        )

        assert not keep

    def test_the_free_check_alone_would_have_kept_it(self) -> None:
        """Pins WHY the clearance mask is a separate parameter.

        Every cell on the path is still free after the wall appears, so the
        pre-existing `prob < free_threshold` test cannot catch this. Without
        this assertion someone could delete the mask argument as redundant.
        """
        grid, planner, path = self.wall_scenario()
        prob = grid.probability()

        next_step = path[1]
        assert prob[next_step[1], next_step[0]] < 0.4  # still free
        assert planner.clearance_mask(grid)[next_step[1], next_step[0]]  # but illegal

    def test_replanning_now_agrees_it_is_illegal(self) -> None:
        """The planner and the keep-alive test must not disagree."""
        grid, planner, _ = self.wall_scenario()

        assert planner.plan(grid, (1, 1), (7, 1)) is None

    def test_a_path_clear_of_the_wall_is_kept(self) -> None:
        """The check rejects only what it must — no blanket re-planning."""
        grid = make_grid()
        planner = AStarPlanner(clearance_radius=0.0)
        a = assignment_to(6, 0, (0, 0))

        assert is_assignment_valid(
            state(0, (1, 0), a, 1),
            grid.probability(),
            0.4,
            MAX_WAIT,
            planner.clearance_mask(grid),
        )
