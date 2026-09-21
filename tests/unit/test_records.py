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

from swarm_mapping.records import (
    LOG_FILE,
    RUN_FILE,
    SCHEMA_VERSION,
    VARIANTS,
    RunInputs,
    RunOutputs,
    RunRecord,
    RunRecordError,
    capture_run_log,
    list_runs,
    load_run,
    variant_label,
    write_run_record,
)

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

    def test_an_event_extra_names_the_event(self, tmp_path: Path) -> None:
        """A prose message for the terminal still logs under a stable name.

        Without this, every progress line ("Tick 50/60 — coverage 12.3%")
        would be its own event and its own `event_counts` key.
        """
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path) as run_log:
            for tick in (50, 100):
                LOGGER.info(
                    "Tick %d — coverage %.1f%%",
                    tick,
                    12.3,
                    extra={"event": "mission_progress", "coverage": 0.123},
                )
            counts = run_log.event_counts()

        lines = read_lines(log_path)
        assert [line["event"] for line in lines] == ["mission_progress"] * 2
        assert lines[0]["coverage"] == 0.123
        assert counts == {"mission_progress": 2}

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


class TestVariants:
    """The allocation variants, labelled as in the sprint plan's table."""

    def test_the_table_is_exactly_the_sprint_plan(self) -> None:
        """The four labels are these four (assignment, tolerance) pairs."""
        assert VARIANTS == {
            "baseline": ("greedy", 0),
            "A": ("global", 0),
            "B": ("greedy", 3),
            "A+B": ("global", 3),
        }

    @pytest.mark.parametrize(("label", "pair"), sorted(VARIANTS.items()))
    def test_each_pair_maps_to_its_label(
        self, label: str, pair: tuple[str, int]
    ) -> None:
        """Each of the four pairs reads back as its label."""
        assert variant_label(*pair) == label

    @pytest.mark.parametrize("pair", [("greedy", 1), ("global", 5)])
    def test_anything_else_is_custom(self, pair: tuple[str, int]) -> None:
        """A tolerance off the table is not silently rounded to a variant."""
        assert variant_label(*pair) == "custom"


def make_record(started_at: str = "2026-09-21T10:00:00.000000+00:00") -> RunRecord:
    """A small but complete record, as `run_pipeline` would build one."""
    return RunRecord(
        started_at=started_at,
        inputs=RunInputs(
            config_path="/scenarios/small_indoor/config.yaml",
            config={"scene_path": "small_indoor.xml"},
            drones=2,
            assignment="global",
            target_tolerance_cells=3,
            variant="A+B",
            view=False,
        ),
        outputs=RunOutputs(
            ticks=30,
            coverage=0.25,
            blocked=False,
            unreachable_frontiers=0,
            tick_capped=True,
            succeeded=False,
            wall_seconds=1.5,
            paths={"0": {"ticks": 30, "route_length": 12}},
            event_counts={"mission_blocked": 1},
            files=[LOG_FILE, "map.npz", RUN_FILE],
        ),
    )


class TestRunRecordFile:
    """`run.json` round-trips through `write_run_record` / `load_run`."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """What is written is what is read, plus where it was read from."""
        record = make_record()
        write_run_record(tmp_path, record)

        loaded = load_run(tmp_path)
        assert loaded.inputs == record.inputs
        assert loaded.outputs == record.outputs
        assert loaded.started_at == record.started_at
        assert loaded.directory == tmp_path

    def test_file_carries_the_schema_version(self, tmp_path: Path) -> None:
        """Four top-level keys, versioned, so a reader can refuse a future one."""
        write_run_record(tmp_path, make_record())

        data = json.loads((tmp_path / RUN_FILE).read_text())
        assert data["schema_version"] == SCHEMA_VERSION == 1
        assert set(data) == {"schema_version", "started_at", "inputs", "outputs"}

    def test_writing_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        """The write is replace-on-complete and leaves nothing else behind.

        A reader never sees half a file, and the run directory holds only what
        the run produced.
        """
        write_run_record(tmp_path, make_record())

        assert sorted(p.name for p in tmp_path.iterdir()) == [RUN_FILE]

    def test_a_missing_file_is_a_run_record_error(self, tmp_path: Path) -> None:
        """A directory without run.json is reported, naming the file."""
        with pytest.raises(RunRecordError, match=RUN_FILE):
            load_run(tmp_path)

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            "[]",
            json.dumps({"schema_version": 99}),
            json.dumps({"schema_version": 1, "started_at": "x"}),
            json.dumps(
                {
                    "schema_version": 1,
                    "started_at": "x",
                    "inputs": {"surprise": 1},
                    "outputs": {},
                }
            ),
        ],
        ids=["corrupt", "not-an-object", "future-schema", "no-sections", "bad-keys"],
    )
    def test_a_malformed_file_is_a_run_record_error(
        self, tmp_path: Path, content: str
    ) -> None:
        """Every way a file can be wrong surfaces as one exception type."""
        (tmp_path / RUN_FILE).write_text(content)

        with pytest.raises(RunRecordError):
            load_run(tmp_path)


class TestListRuns:
    """Test case 4: history newest-first; broken runs skipped, and named."""

    def write_run(self, root: Path, name: str, started_at: str) -> None:
        """Write a valid record into `root/name`."""
        directory = root / name
        directory.mkdir()
        write_run_record(directory, make_record(started_at))

    def test_newest_first(self, tmp_path: Path) -> None:
        """Ordered by start time, not by name or by filesystem order."""
        self.write_run(tmp_path, "a", "2026-09-21T09:00:00.000000+00:00")
        self.write_run(tmp_path, "b", "2026-09-21T11:00:00.000000+00:00")
        self.write_run(tmp_path, "c", "2026-09-21T10:00:00.000000+00:00")

        listing = list_runs(tmp_path)
        assert [r.directory.name for r in listing.runs if r.directory] == [
            "b",
            "c",
            "a",
        ]
        assert listing.skipped == []

    def test_ties_break_by_directory_name(self, tmp_path: Path) -> None:
        """Equal timestamps still give one deterministic order."""
        same = "2026-09-21T10:00:00.000000+00:00"
        for name in ("run_b", "run_a", "run_c"):
            self.write_run(tmp_path, name, same)

        names = [r.directory.name for r in list_runs(tmp_path).runs if r.directory]
        assert names == ["run_a", "run_b", "run_c"]

    def test_broken_runs_are_skipped_and_named(self, tmp_path: Path) -> None:
        """A crashed or half-written run neither breaks the listing nor hides.

        The history page must render with one in the folder, and say which runs
        it could not show.
        """
        self.write_run(tmp_path, "good", "2026-09-21T10:00:00.000000+00:00")
        (tmp_path / "corrupt").mkdir()
        (tmp_path / "corrupt" / RUN_FILE).write_text("{truncated")
        (tmp_path / "missing").mkdir()

        listing = list_runs(tmp_path)
        assert [r.directory.name for r in listing.runs if r.directory] == ["good"]
        assert [s.directory.name for s in listing.skipped] == ["corrupt", "missing"]
        reasons = {s.directory.name: s.reason for s in listing.skipped}
        assert RUN_FILE in reasons["missing"]
        assert reasons["corrupt"]

    def test_plain_files_in_the_root_are_ignored(self, tmp_path: Path) -> None:
        """Only directories are runs; a stray file is neither run nor skip."""
        (tmp_path / "notes.txt").write_text("hello")

        listing = list_runs(tmp_path)
        assert listing.runs == []
        assert listing.skipped == []

    def test_a_missing_root_is_an_empty_history(self, tmp_path: Path) -> None:
        """First launch: no runs yet is not an error."""
        listing = list_runs(tmp_path / "never_created")
        assert listing.runs == []
        assert listing.skipped == []
