"""Tests for per-drone path recording (visualization.path_log).

Pure data structures — no MuJoCo, no grid. The point of the module is that two
views (the route as flown, and per-cell visit ticks) come from one recording,
so the tests check they stay consistent with each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from swarm_mapping.visualization.path_log import (
    DroneTrack,
    PathLog,
    iter_segments,
    save_path_log,
)

pytestmark = pytest.mark.sprint(2)  # path diagnostics


@dataclass
class FakeState:
    """Just enough of DroneState for the log, which reads only `.cell`."""

    cell: tuple[int, int]


def track(*cells: tuple[int, int]) -> DroneTrack:
    """A track that occupied `cells` on consecutive ticks from 0."""
    return DroneTrack(drone_id=0, cells=list(cells))


class TestRoute:
    """The path as flown."""

    @pytest.mark.sanity
    def test_route_follows_the_cells_in_order(self) -> None:
        """The route is the sequence, not a set."""
        assert track((0, 0), (1, 0), (2, 0)).route() == [(0, 0), (1, 0), (2, 0)]

    def test_holding_station_collapses_to_one_entry(self) -> None:
        """Twenty ticks in one cell is one place, not twenty.

        Dwell time belongs to `cells`; the route answers where the drone went.
        """
        assert track((5, 5), (5, 5), (5, 5), (6, 5)).route() == [(5, 5), (6, 5)]

    def test_a_revisit_appears_twice_in_the_route(self) -> None:
        """Collapsing repeats must not collapse a genuine return."""
        assert track((0, 0), (1, 0), (0, 0)).route() == [(0, 0), (1, 0), (0, 0)]


class TestVisits:
    """Per-cell tick indices."""

    def test_each_cell_maps_to_the_ticks_it_was_entered(self) -> None:
        """The worked example: entered on tick 0 and tick 2."""
        visits = track((0, 0), (1, 0), (0, 0)).visits()

        assert visits[(0, 0)] == [0, 2]
        assert visits[(1, 0)] == [1]

    def test_dwelling_counts_as_one_entry(self) -> None:
        """Entries, not dwell time — otherwise a parked drone looks busy."""
        assert track((4, 4), (4, 4), (4, 4)).visits() == {(4, 4): [0]}

    def test_revisited_keeps_only_repeat_entries(self) -> None:
        """A cell seen once is not a revisit."""
        revisited = track((0, 0), (1, 0), (0, 0)).revisited()

        assert list(revisited) == [(0, 0)]

    def test_gap_summary_ranks_the_longest_return_first(self) -> None:
        """A long gap means the drone left and came back — the wasteful kind.

        A short gap is just turning round on the spot.
        """
        cells = [(0, 0), (1, 0), (0, 0)] + [(9, 9)] * 20 + [(1, 0)]
        gaps = DroneTrack(drone_id=0, cells=cells).gap_summary()

        assert gaps[0][0] == (1, 0)  # returned to after 22 ticks
        assert gaps[0][1] > gaps[1][1]


class TestDivisionOfLabour:
    """The measure the log exists for."""

    def test_separate_areas_score_one(self) -> None:
        """Drones that never overlap have divided the work perfectly."""
        log = PathLog()
        for _ in range(2):
            log.record({0: FakeState((0, 0)), 1: FakeState((9, 9))})
        log.record({0: FakeState((1, 0)), 1: FakeState((8, 9))})

        assert log.exclusive_fraction() == 1.0
        assert log.shared_cells() == {}

    def test_shared_ground_lowers_the_fraction(self) -> None:
        """Two drones over the same cell is the thing being measured."""
        log = PathLog()
        log.record({0: FakeState((5, 5)), 1: FakeState((5, 5))})
        log.record({0: FakeState((6, 5)), 1: FakeState((7, 5))})

        assert log.exclusive_fraction() == pytest.approx(2 / 3)
        assert log.shared_cells() == {(5, 5): [0, 1]}

    def test_empty_log_is_zero_not_an_error(self) -> None:
        """A mission that recorded nothing must not divide by zero."""
        assert PathLog().exclusive_fraction() == 0.0


class TestExport:
    """The JSON a later analysis reads."""

    def test_written_json_round_trips_the_route_and_visits(
        self, tmp_path: Path
    ) -> None:
        """Cells are string keys because JSON demands it; routes stay pairs."""
        log = PathLog()
        log.record({0: FakeState((0, 0))})
        log.record({0: FakeState((1, 0))})
        log.record({0: FakeState((0, 0))})
        destination = tmp_path / "paths.json"

        save_path_log(log, destination)
        payload = json.loads(destination.read_text())

        assert payload["drones"]["0"]["route"] == [[0, 0], [1, 0], [0, 0]]
        assert payload["drones"]["0"]["visits"]["0,0"] == [0, 2]
        assert payload["summary"]["0"]["revisited_cells"] == 1


class TestSegments:
    """Route pairs, for drawing."""

    def test_segments_pair_consecutive_cells(self) -> None:
        """Three cells make two segments."""
        assert list(iter_segments([(0, 0), (1, 0), (1, 1)])) == [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
        ]

    def test_a_single_cell_has_no_segments(self) -> None:
        """A drone that never moved draws nothing."""
        assert list(iter_segments([(0, 0)])) == []
