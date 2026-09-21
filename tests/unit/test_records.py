"""Tests for run records: the JSON Lines log, `run.json`, and run history.

Pure Python — no MuJoCo. That a real run writes both files is checked end to end
in `tests/integration/test_run_records.py`; here the contracts are pinned with
hand-built log calls and hand-written records, so each failure names one rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarm_mapping.records import LOG_FILE, capture_run_log

pytestmark = pytest.mark.sprint(3)

# A child of the captured `swarm_mapping` logger, so records reach the run log
# exactly as a real module's would.
LOGGER = logging.getLogger("swarm_mapping.test_records")


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON Lines file, one object per line."""
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestJsonLinesLog:
    """R3: one JSON object per line, every line carrying `event` and `tick`."""

    def test_each_line_is_an_object_with_event_tick_and_level(
        self, tmp_path: Path
    ) -> None:
        """The three fixed keys, and nothing else for a bare call."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path) as run_log:
            run_log.tick = 7
            LOGGER.warning("mission_blocked")

        assert read_lines(log_path) == [
            {"event": "mission_blocked", "tick": 7, "level": "WARNING"}
        ]

    def test_extras_become_fields(self, tmp_path: Path) -> None:
        """What a call site passes in `extra=` is the event's payload."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("frontier_exhausted", extra={"drone_id": 2, "cell": (3, 4)})

        (line,) = read_lines(log_path)
        assert line["drone_id"] == 2
        assert line["cell"] == [3, 4]

    def test_standard_record_attributes_are_not_leaked(self, tmp_path: Path) -> None:
        """`lineno`, `pathname`, `process` and the like are left out.

        They are noise, and some are machine- or time-specific: they would make
        two identical runs' logs differ.
        """
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("plain")

        (line,) = read_lines(log_path)
        assert set(line) == {"event", "tick", "level"}

    def test_every_line_has_a_tick_even_when_the_call_site_gave_none(
        self, tmp_path: Path
    ) -> None:
        """The filter stamps the loop's current tick; before the first it is 0."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path) as run_log:
            LOGGER.info("starting")
            run_log.tick = 1
            LOGGER.info("first tick")

        assert [line["tick"] for line in read_lines(log_path)] == [0, 1]

    def test_a_call_site_tick_is_kept(self, tmp_path: Path) -> None:
        """The master already passes `tick` on some events; it is not replaced."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path) as run_log:
            run_log.tick = 9
            LOGGER.warning("mission_stalled", extra={"tick": 3})

        (line,) = read_lines(log_path)
        assert line["tick"] == 3

    def test_message_arguments_are_interpolated(self, tmp_path: Path) -> None:
        """`event` is `record.getMessage()`, so %-style calls read as sent."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("Tick %d/%d", 50, 2000)

        (line,) = read_lines(log_path)
        assert line["event"] == "Tick 50/2000"

    def test_numpy_and_unknown_values_do_not_break_the_line(
        self, tmp_path: Path
    ) -> None:
        """A log call must never crash a mission over serialisation."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info(
                "odd",
                extra={"count": np.int64(4), "where": Path("a/b"), "obj": object()},
            )

        (line,) = read_lines(log_path)
        assert line["count"] == 4
        assert line["where"] == "a/b"
        assert isinstance(line["obj"], str)

    def test_info_is_captured_even_when_the_logger_is_quieter(
        self, tmp_path: Path
    ) -> None:
        """The file log is complete regardless of `--verbose`.

        The logger's own level is restored afterwards, so what reaches stderr
        after the run is exactly what reached it before.
        """
        package = logging.getLogger("swarm_mapping")
        previous = package.level
        package.setLevel(logging.WARNING)
        try:
            log_path = tmp_path / LOG_FILE
            with capture_run_log(log_path):
                LOGGER.info("assignment_dropped")
            assert package.level == logging.WARNING
        finally:
            package.setLevel(previous)

        assert [line["event"] for line in read_lines(log_path)] == [
            "assignment_dropped"
        ]

    def test_loggers_outside_the_package_are_not_captured(self, tmp_path: Path) -> None:
        """Third-party chatter (matplotlib, MuJoCo) is not the run's log."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            logging.getLogger("matplotlib.font_manager").warning("third party")

        assert read_lines(log_path) == []

    def test_the_handler_is_detached_afterwards(self, tmp_path: Path) -> None:
        """A later run, or a test, never writes into this run's log."""
        package = logging.getLogger("swarm_mapping")
        handlers_before = list(package.handlers)
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("during")
        LOGGER.info("after")

        assert [line["event"] for line in read_lines(log_path)] == ["during"]
        assert package.handlers == handlers_before

    def test_the_handler_is_detached_on_an_exception_too(self, tmp_path: Path) -> None:
        """A crashed run releases the file and the logger just the same."""
        package = logging.getLogger("swarm_mapping")
        handlers_before = list(package.handlers)
        log_path = tmp_path / LOG_FILE
        with pytest.raises(RuntimeError), capture_run_log(log_path):
            LOGGER.info("during")
            raise RuntimeError
        LOGGER.info("after")

        assert [line["event"] for line in read_lines(log_path)] == ["during"]
        assert package.handlers == handlers_before

    def test_a_reused_file_is_replaced_not_appended(self, tmp_path: Path) -> None:
        """Re-running into the same `--output` must not splice two runs' logs."""
        log_path = tmp_path / LOG_FILE
        for _ in range(2):
            with capture_run_log(log_path):
                LOGGER.info("only once")

        assert len(read_lines(log_path)) == 1

    def test_event_counts_tally_lines_per_event(self, tmp_path: Path) -> None:
        """`event_counts` is what run.json reports; keys sorted for stability."""
        with capture_run_log(tmp_path / LOG_FILE) as run_log:
            LOGGER.info("b")
            LOGGER.info("a")
            LOGGER.info("b")
            counts = run_log.event_counts()

        assert counts == {"a": 1, "b": 2}
        assert list(counts) == ["a", "b"]
