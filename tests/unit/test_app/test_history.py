"""Tests for what the History page and the live run panel show.

Pure: hand-written run directories, no MuJoCo, no streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm_mapping.app.history import (
    compare,
    event_names,
    history_row,
    progress,
    read_log,
    route_images,
)
from swarm_mapping.records import LOG_FILE, load_run

from .conftest import LOG_LINES, make_run

pytestmark = pytest.mark.sprint(3)


class TestHistoryRow:
    """One table row per run, with the columns C5 names."""

    def test_row_fields(self, tmp_path: Path) -> None:
        """Every column comes from the record, the scenario from its config path."""
        record = load_run(make_run(tmp_path, "r1"))

        assert history_row(record) == {
            "started": "2026-09-21T10:00:00.000000+00:00",
            "run": "r1",
            "scenario": "small_indoor",
            "variant": "A+B",
            "drones": 2,
            "ticks": 120,
            "coverage": 0.95,
            "blocked": False,
            "succeeded": True,
            "wall seconds": 4.5,
            "failures detected": 1,
        }

    def test_no_failure_events_means_zero(self, tmp_path: Path) -> None:
        """A run without `drone_failed` lines detected no failures."""
        record = load_run(make_run(tmp_path, "r1", event_counts={}))

        assert history_row(record)["failures detected"] == 0


class TestCompare:
    """Two runs side by side, with the difference where one means something."""

    def test_numeric_metrics_get_b_minus_a(self, tmp_path: Path) -> None:
        """Ticks and coverage differences are the second run minus the first."""
        a = load_run(make_run(tmp_path, "a", ticks=100, coverage=0.90))
        b = load_run(make_run(tmp_path, "b", ticks=130, coverage=0.95))

        deltas = {d.metric: d for d in compare(a, b)}

        assert deltas["ticks"].a == 100
        assert deltas["ticks"].b == 130
        assert deltas["ticks"].difference == 30
        assert deltas["coverage"].difference == pytest.approx(0.05)
        assert deltas["total route length"].a == 60

    def test_text_and_flags_have_no_difference(self, tmp_path: Path) -> None:
        """A difference of variants or of booleans is not a number."""
        a = load_run(make_run(tmp_path, "a", variant="baseline"))
        b = load_run(make_run(tmp_path, "b", variant="A+B"))

        deltas = {d.metric: d for d in compare(a, b)}

        assert deltas["variant"].a == "baseline"
        assert deltas["variant"].difference is None
        assert deltas["succeeded"].difference is None


class TestReadLog:
    """The log viewer and the live tail read `log.jsonl` safely."""

    def test_reads_every_entry_in_order(self, tmp_path: Path) -> None:
        """Entries come back as the dicts that were written."""
        directory = make_run(tmp_path, "r1")

        assert read_log(directory / LOG_FILE) == LOG_LINES

    def test_last_keeps_the_tail(self, tmp_path: Path) -> None:
        """`last=2` is the final two entries."""
        directory = make_run(tmp_path, "r1")

        assert read_log(directory / LOG_FILE, last=2) == LOG_LINES[-2:]

    def test_a_half_written_final_line_is_skipped(self, tmp_path: Path) -> None:
        """A running mission's partial line is not an error."""
        path = tmp_path / LOG_FILE
        path.write_text(json.dumps(LOG_LINES[0]) + '\n{"event": "fron')

        assert read_log(path) == [LOG_LINES[0]]

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        """Before the mission creates its log there is nothing to show."""
        assert read_log(tmp_path / LOG_FILE) == []

    def test_event_names_are_distinct_and_sorted(self) -> None:
        """The filter offers each event once, alphabetically."""
        assert event_names(LOG_LINES) == [
            "drone_failed",
            "frontier_assigned",
            "mission_progress",
            "mission_started",
        ]


class TestProgress:
    """The live panel's tick and coverage, by event name and field."""

    def test_latest_tick_and_progress_coverage(self) -> None:
        """Tick from the last entry, coverage from the last progress event."""
        assert progress(LOG_LINES).tick == 61
        assert progress(LOG_LINES).coverage == pytest.approx(0.4)

    def test_nothing_logged_yet(self) -> None:
        """An empty log has neither."""
        assert progress([]).tick is None
        assert progress([]).coverage is None


class TestRouteImages:
    """One tab per drone that has a route image."""

    def test_numeric_order_and_missing_files_skipped(self, tmp_path: Path) -> None:
        """Drone 1's image is removed, so only drone 0 remains."""
        directory = make_run(tmp_path, "r1")
        (directory / "route_drone_1.png").unlink()

        images = route_images(load_run(directory))

        assert images == [("0", directory / "route_drone_0.png")]
