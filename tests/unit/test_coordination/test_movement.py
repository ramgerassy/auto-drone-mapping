"""Tests for collision-free movement resolution (coordination.movement).

Pure logic — no MuJoCo, no map. Cells are (col, row). Separation is expressed
in cells here; the master converts metres to cells before calling in.

Right of way goes to the higher drone id.
"""

from __future__ import annotations

import pytest

from swarm_mapping.coordination.movement import resolve_moves
from swarm_mapping.coordination.types import Cell

pytestmark = pytest.mark.sprint(2)  # Feature 4 — centralized master

# One cell of separation: a drone may not enter a cell another drone holds,
# but adjacent cells are fine.
ONE_CELL = 1.0


class TestNoConflict:
    """Drones with room to move all advance."""

    @pytest.mark.sanity
    def test_all_advance_when_far_apart(self) -> None:
        """Well-separated drones each take their step."""
        current: dict[int, Cell] = {0: (0, 0), 1: (10, 10)}
        desired: dict[int, Cell | None] = {0: (1, 0), 1: (11, 10)}

        assert resolve_moves(current, desired, ONE_CELL) == {0: (1, 0), 1: (11, 10)}

    def test_drone_without_target_holds(self) -> None:
        """A drone with nowhere to go stays where it is."""
        current: dict[int, Cell] = {0: (0, 0), 1: (5, 5)}
        desired: dict[int, Cell | None] = {0: None, 1: (6, 5)}

        assert resolve_moves(current, desired, ONE_CELL) == {0: (0, 0), 1: (6, 5)}

    def test_target_equal_to_current_is_a_hold(self) -> None:
        """A target identical to the current cell is not treated as a move."""
        current: dict[int, Cell] = {0: (3, 3)}
        desired: dict[int, Cell | None] = {0: (3, 3)}

        assert resolve_moves(current, desired, ONE_CELL) == {0: (3, 3)}


class TestRightOfWay:
    """Conflicts resolve in favour of the higher drone id."""

    def test_same_target_higher_id_wins(self) -> None:
        """Two drones want one cell: the higher id takes it, the lower holds."""
        current: dict[int, Cell] = {1: (0, 0), 3: (2, 0)}
        desired: dict[int, Cell | None] = {1: (1, 0), 3: (1, 0)}

        result = resolve_moves(current, desired, ONE_CELL)

        assert result[3] == (1, 0)  # advanced
        assert result[1] == (0, 0)  # held

    def test_precedence_is_by_id_not_input_order(self) -> None:
        """Insertion order must not decide who yields."""
        # Lower id inserted first here, higher id first in the mirror case;
        # both must give the cell to id 3.
        low_first = resolve_moves(
            {1: (0, 0), 3: (2, 0)}, {1: (1, 0), 3: (1, 0)}, ONE_CELL
        )
        high_first = resolve_moves(
            {3: (2, 0), 1: (0, 0)}, {3: (1, 0), 1: (1, 0)}, ONE_CELL
        )

        assert low_first == high_first
        assert low_first[3] == (1, 0)

    def test_blocked_drone_holds_its_current_cell(self) -> None:
        """A drone that yields is reserved at its current cell, not dropped."""
        current: dict[int, Cell] = {1: (0, 0), 2: (2, 0)}
        desired: dict[int, Cell | None] = {1: (1, 0), 2: (1, 0)}

        result = resolve_moves(current, desired, ONE_CELL)

        assert set(result) == {1, 2}
        assert result[1] == (0, 0)


class TestSwapConflict:
    """A pair must never pass through each other."""

    def test_head_on_swap_is_blocked(self) -> None:
        """Both targets look free, but the swap is refused by both drones.

        Seeding reservations with current cells is what blocks this — the
        unprocessed drone still occupies the cell the other one wants.
        """
        current: dict[int, Cell] = {1: (0, 0), 2: (1, 0)}
        desired: dict[int, Cell | None] = {1: (1, 0), 2: (0, 0)}

        result = resolve_moves(current, desired, ONE_CELL)

        assert result == {1: (0, 0), 2: (1, 0)}  # neither moved

    def test_swap_blocked_regardless_of_id_order(self) -> None:
        """The higher id does not get to swap through the lower one either."""
        result = resolve_moves({5: (4, 4), 2: (5, 4)}, {5: (5, 4), 2: (4, 4)}, ONE_CELL)

        assert result == {5: (4, 4), 2: (5, 4)}


class TestSeparationDistance:
    """Separation is a distance, not merely cell inequality."""

    def test_distinct_but_too_close_is_blocked(self) -> None:
        """With a 3-cell separation, an adjacent free cell is still refused."""
        current: dict[int, Cell] = {1: (0, 0), 2: (3, 0)}
        desired: dict[int, Cell | None] = {1: (1, 0), 2: (3, 0)}

        result = resolve_moves(current, desired, 3.0)

        assert result[1] == (0, 0)  # (1,0) is only 2 cells from drone 2

    def test_diagonal_separation_uses_euclidean_distance(self) -> None:
        """Separation is radial: a diagonal neighbour is ~1.41 cells away."""
        current: dict[int, Cell] = {1: (0, 0), 2: (2, 2)}
        desired: dict[int, Cell | None] = {1: (1, 1), 2: (2, 2)}

        # sqrt(2) ~= 1.414: allowed at 1.4, refused at 1.5.
        assert resolve_moves(current, desired, 1.4)[1] == (1, 1)
        assert resolve_moves(current, desired, 1.5)[1] == (0, 0)


class TestDeterminism:
    """Same inputs, same outcome."""

    def test_repeated_calls_identical(self) -> None:
        """Resolution is a pure function of its inputs."""
        current: dict[int, Cell] = {1: (0, 0), 2: (1, 0), 3: (2, 0)}
        desired: dict[int, Cell | None] = {1: (1, 0), 2: (2, 0), 3: (3, 0)}

        first = resolve_moves(current, desired, ONE_CELL)
        second = resolve_moves(current, desired, ONE_CELL)

        assert first == second

    def test_inputs_are_not_mutated(self) -> None:
        """Resolution does not write back into its arguments."""
        current: dict[int, Cell] = {1: (0, 0), 2: (5, 5)}
        desired: dict[int, Cell | None] = {1: (1, 0), 2: (6, 5)}

        resolve_moves(current, desired, ONE_CELL)

        assert current == {1: (0, 0), 2: (5, 5)}
        assert desired == {1: (1, 0), 2: (6, 5)}
