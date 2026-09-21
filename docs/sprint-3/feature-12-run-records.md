# Feature 12 — Run records & scenario validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run leaves a self-describing record in its output directory — a JSON Lines log, a `run.json` summary, and its drone paths — and an uploaded room can be validated and installed without ever overwriting anything. This is everything the operator console (Feature 13) needs that is not UI, and it is usable from the CLI on its own.

**Architecture:** A new top-level module `swarm_mapping.records` owns the run-record format: the JSON Lines log handler, the `run.json` schema and writer, and the readers `load_run` / `list_runs`. `cli.run_pipeline` is its only writer. A new top layer `swarm_mapping.app` (the console's package, created here without any UI) holds `validate_scenario` / `install_scenario`; it may import `cli`, and nothing imports it. No domain module changes.

**Tech Stack:** Python 3.13, MuJoCo 3.8, PyYAML, pytest, ruff, mypy (strict). No new dependencies; nothing here imports streamlit.

**Spec:** [`docs/sprint-3-plan.md`](../sprint-3-plan.md) — "D2 — what the console must do", the Feature 12 row, "Features 12–13 — test cases" 1–11, Cross-cutting constraints; and the controller rulings R1–R8 in the Feature 12 brief, restated below.

## Rulings this plan implements

| # | Ruling |
| --- | --- |
| R1 | `--output DIR` keeps its meaning. `run.json` and `log.jsonl` go into the same DIR. No new CLI flags. |
| R2 | `swarm_mapping/records.py` (schema, writer, `load_run`, `list_runs`), used by `cli` and later the console. `swarm_mapping/app/validation.py` (`validate_scenario`, `install_scenario`) may import `cli`; `cli` never imports `app`. |
| R3 | For the duration of `run_pipeline` a handler writes `DIR/log.jsonl`, detached afterwards (also on exception). One object per line: `{"event": getMessage(), "tick": ..., "level": levelname, ...extras}`. `tick` is stamped by a `logging.Filter` the loop updates every tick. Stderr logging unchanged. |
| R4 | `run.json`, `schema_version: 1`, generic: inputs (config path, config snapshot, drones, assignment, target tolerance, variant, view) and outputs (ticks, coverage, blocked, unreachable frontiers, tick_capped, succeeded, wall_seconds, per-drone path stats, event_counts, files). |
| R5 | `PathLog` always on; `paths.json` and `route_drone_<id>.png` always written. Visit heatmaps stay behind `--visit-heatmaps`. |
| R6 | `install_scenario(name, config_text, scene_text, scenarios_root, assets_dir)`: name `^[a-z][a-z0-9_]{0,40}$`; never overwrite; validate in a temp dir first with `require_five=True`; write nothing unless clean. |
| R7 | `validate_scenario(config_path, *, require_five=False) -> list[str]`, never raises. Config parses; scene exists (before MuJoCo); five spawns if required; spawns clear of geometry by MuJoCo contacts; separation/bounds via `build_mission` raising. |
| R8 | Test 5 becomes: two headless runs give identical `run.json` except the wall clock; `view` is the only input that differs when recorded as `True`. |

## Global Constraints

- `from __future__ import annotations` at the top of every file; Google-style docstrings on every module and public function/class; `uv run mypy` (strict) clean.
- Dependency direction: `records` imports nothing from the domain modules (it serialises plain data). `cli → records`. `app → cli, config, simulation, records`. Nothing imports `app`.
- No MuJoCo mocks. End-to-end tests use `small_indoor` with 1–2 drones, capped to a few dozen ticks; validation tests use the real scenes and tiny inline MJCF. The single monkeypatch (test 10) is a tripwire that fails if the simulator is reached — it replaces nothing that runs.
- Tests write only into pytest tmp dirs.
- Determinism: the log carries no wall-clock field, so two identical runs write byte-identical `log.jsonl`. `event_counts` and `files` are sorted. `list_runs` sorts newest-first by `started_at`, ties by directory name.
- New test modules carry `pytestmark = pytest.mark.sprint(3)`.
- Existing outputs (`map.npz`, `map.png`), exit codes, stdout lines and stderr logs are unchanged.
- Run before every commit: `uv run ruff check src/ tests/ benchmarks/ && uv run ruff format --check src/ tests/ benchmarks/ && uv run mypy && uv run pytest -q -m "not acceptance"`.

## Formats

### `log.jsonl`

One JSON object per line, in emission order:

```json
{"event": "Starting exploration: 2 drone(s), max 60 ticks", "tick": 0, "level": "INFO"}
{"event": "mission_blocked", "tick": 212, "level": "WARNING", "unreachable_frontiers": 3}
```

- `event` is `record.getMessage()`; `tick` is the call site's own `tick` extra if it passed one, otherwise the loop's current tick (0 before the first tick, the tick being executed during it); `level` is the level name; every other `extra=` field follows. Standard `LogRecord` attributes are excluded — several (`process`, `pathname`, `created`) are machine- or time-specific and would break byte-identical logs.
- Captures the `swarm_mapping` logger tree at INFO and above, regardless of `--verbose`. Third-party loggers are not captured.
- Values JSON cannot encode become JSON-native where possible (numpy scalars via `.item()`), else `str()`. A log call never crashes a mission.

### `run.json`

```json
{
  "schema_version": 1,
  "started_at": "2026-09-21T10:00:00.123456+00:00",
  "inputs": {
    "config_path": "/abs/path/scenarios/small_indoor/config.yaml",
    "config": { "...": "dataclasses.asdict(ScenarioConfig), overrides applied" },
    "drones": 2,
    "assignment": "global",
    "target_tolerance_cells": 3,
    "variant": "A+B",
    "view": false
  },
  "outputs": {
    "ticks": 60, "coverage": 0.31, "blocked": false, "unreachable_frontiers": 0,
    "tick_capped": true, "succeeded": false, "wall_seconds": 1.42,
    "paths": {"0": {"ticks": 60, "route_length": 58, "distinct_cells": 58,
                    "revisited_cells": 0, "longest_gap": 0}},
    "event_counts": {"Starting exploration: 2 drone(s), max 60 ticks": 1},
    "files": ["log.jsonl", "map.npz", "map.png", "paths.json",
              "route_drone_0.png", "run.json"]
  }
}
```

- `started_at` (UTC ISO-8601) is required by `list_runs` ordering; with `wall_seconds` it is one of the two wall-clock fields R8's comparison excludes.
- `drones` is the count actually flown (after `--drones`); with the snapshot's start positions it reproduces the run, because `--drones N` takes the first N.
- `variant` is the sprint-plan label for `(assignment, target_tolerance_cells)`, or `"custom"`.
- `wall_seconds` spans config load to export; the post-mission wait in `--view` (the operator inspecting the map) is excluded, and `run.json` is written before it.
- `files` lists what this run wrote into DIR, including `run.json` itself — not a directory listing, so a user's unrelated files are never claimed.
- Written to a temp file and `os.replace`d, so a reader never sees half a record. A run that crashes leaves `log.jsonl` but no `run.json`; `list_runs` reports it as skipped.

## File map

| File | Change | Responsibility |
| --- | --- | --- |
| `src/swarm_mapping/records.py` | create | JSON Lines log capture; `RunInputs` / `RunOutputs` / `RunRecord`; `VARIANTS`, `variant_label`; `write_run_record`, `load_run`, `list_runs` |
| `src/swarm_mapping/cli.py` | modify | `run_pipeline` captures the log, always records paths, writes `run.json`; `describe_inputs`; `main` pins the stderr handler's level |
| `src/swarm_mapping/app/__init__.py` | create | the console package (no UI yet) |
| `src/swarm_mapping/app/validation.py` | create | `validate_scenario`, `install_scenario` |
| `tests/unit/test_records.py` | create | log, schema, history — pure Python |
| `tests/integration/test_run_records.py` | create | a real run writes the right records |
| `tests/integration/test_scenario_validation.py` | create | validation and installation on real scenes |

## Test-case map (sprint plan, Features 12–13)

| # | Case | Covered by |
| --- | --- | --- |
| 1 | own directory, two runs never share | `TestEveryRunIsRecorded` |
| 2 | `run.json` reproduces the run | `TestRunJson::test_the_record_reproduces_the_map` (+ inputs tests) |
| 3 | every log line JSON with `event`/`tick` | `TestJsonLinesLog`, `TestLogFile`. The failure-event half needs Features 9/10's events and is asserted where those land. |
| 4 | `list_runs` newest-first, skips and names broken runs | `TestListRuns` |
| 5 | replaced by R8 | `TestViewIsViewOnly` |
| 6 | shipped scenarios validate clean | `TestShippedScenarios` |
| 7 | fewer than five spawns → count in message | `TestSpawnCount`, `TestInstallScenario::test_an_invalid_room_writes_nothing` |
| 8 | spawn in a wall → index, by MuJoCo contacts | `TestSpawnInsideGeometry` |
| 9 | spawns too close → rejected | `TestSpawnSeparation` |
| 10 | missing scene → rejected before MuJoCo | `TestMissingScene` |
| 11 | overwrite / `../` → rejected | `TestInstallScenario` |

---

### Task 1: The run log is JSON Lines

**Files:**
- Create: `src/swarm_mapping/records.py` (log half)
- Test: `tests/unit/test_records.py` (`TestJsonLinesLog`)

**Interfaces:**
- Produces: `LOG_FILE = "log.jsonl"`; `capture_run_log(path: Path, logger_name: str = "swarm_mapping") -> Iterator[RunLog]` (context manager); `RunLog.tick` (settable int); `RunLog.event_counts() -> dict[str, int]` (sorted by event).

- [ ] **Step 1: Write the failing tests**

```python
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
        """`lineno`, `pathname`, `process`... are noise, and some are
        machine-specific — they would make two identical runs' logs differ."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("plain")

        (line,) = read_lines(log_path)
        assert set(line) == {"event", "tick", "level"}

    def test_every_line_has_a_tick_even_when_the_call_site_gave_none(
        self, tmp_path: Path
    ) -> None:
        """The filter stamps the loop's current tick; before the first tick it
        is 0."""
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path) as run_log:
            LOGGER.info("starting")
            run_log.tick = 1
            LOGGER.info("first tick")

        assert [line["tick"] for line in read_lines(log_path)] == [0, 1]

    def test_a_call_site_tick_is_kept(self, tmp_path: Path) -> None:
        """The master already passes `tick` on some events; it is not
        overwritten."""
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
        """The file log is complete regardless of `--verbose`; the logger's own
        level is restored afterwards so stderr is unaffected."""
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
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            logging.getLogger("matplotlib.font_manager").warning("third party")

        assert read_lines(log_path) == []

    def test_the_handler_is_detached_afterwards(self, tmp_path: Path) -> None:
        package = logging.getLogger("swarm_mapping")
        handlers_before = list(package.handlers)
        log_path = tmp_path / LOG_FILE
        with capture_run_log(log_path):
            LOGGER.info("during")
        LOGGER.info("after")

        assert [line["event"] for line in read_lines(log_path)] == ["during"]
        assert package.handlers == handlers_before

    def test_the_handler_is_detached_on_an_exception_too(self, tmp_path: Path) -> None:
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
        with capture_run_log(tmp_path / LOG_FILE) as run_log:
            LOGGER.info("b")
            LOGGER.info("a")
            LOGGER.info("b")
            counts = run_log.event_counts()

        assert counts == {"a": 1, "b": 2}
        assert list(counts) == ["a", "b"], "keys sorted, for a stable run.json"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_records.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'swarm_mapping.records'`.

- [ ] **Step 3: Implement**

- `_TickStamp(logging.Filter)` holding `tick: int = 0`; `filter()` sets `record.tick` only when the record has none, returns True.
- `_JsonLinesFormatter.format()`: `{"event", "tick", "level"}` first, then every attribute of `record.__dict__` not in the standard set (computed once from a blank `LogRecord`, plus `message`, `asctime`) and not already present; `exc_info` rendered under `exception`. `json.dumps(..., default=_jsonable)` where `_jsonable` turns numpy scalars into Python via `.item()`, and anything else into `str`.
- `_JsonLinesHandler(logging.FileHandler)` opened `mode="w"` (a reused DIR is replaced, not appended), level INFO, counts `getMessage()` per emitted record.
- `capture_run_log`: attach handler + filter to `logging.getLogger(logger_name)`; if the logger's effective level is above INFO, lower **that logger's** level to INFO for the duration; `finally`: remove and close the handler, restore the level.

Lowering the package logger's level lets INFO records propagate to the root handler too, so `main` must set the **handler's** level (Task 3), which is what keeps stderr unchanged.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_records.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/records.py tests/unit/test_records.py
git commit -m "feat(records): JSON Lines run log with a per-tick stamp"
```

---

### Task 2: `run.json`, and reading the history back

**Files:**
- Modify: `src/swarm_mapping/records.py` (schema half)
- Test: `tests/unit/test_records.py` (`TestVariants`, `TestRunRecordFile`, `TestListRuns`)

**Interfaces:**
- Produces: `RUN_FILE = "run.json"`, `SCHEMA_VERSION = 1`; `VARIANTS: dict[str, tuple[str, int]]`; `variant_label(assignment: str, target_tolerance_cells: int) -> str`; frozen dataclasses `RunInputs`, `RunOutputs`, `RunRecord(started_at, inputs, outputs, directory: Path | None = None)`; `write_run_record(directory: Path, record: RunRecord) -> Path`; `load_run(directory: Path) -> RunRecord` raising `RunRecordError(ValueError)`; `list_runs(root: Path) -> RunListing(runs, skipped)`, `SkippedRun(directory, reason)`.

- [ ] **Step 1: Write the failing tests** (appended to `tests/unit/test_records.py`; its import block grows to the names above)

```python
class TestVariants:
    """The allocation variants, labelled as the sprint plan's table."""

    def test_the_table_is_exactly_the_sprint_plan(self) -> None:
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
        assert variant_label(*pair) == label

    @pytest.mark.parametrize("pair", [("greedy", 1), ("global", 5)])
    def test_anything_else_is_custom(self, pair: tuple[str, int]) -> None:
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
        record = make_record()
        write_run_record(tmp_path, record)

        loaded = load_run(tmp_path)
        assert loaded.inputs == record.inputs
        assert loaded.outputs == record.outputs
        assert loaded.started_at == record.started_at
        assert loaded.directory == tmp_path

    def test_file_carries_the_schema_version(self, tmp_path: Path) -> None:
        write_run_record(tmp_path, make_record())

        data = json.loads((tmp_path / RUN_FILE).read_text())
        assert data["schema_version"] == SCHEMA_VERSION == 1
        assert set(data) == {"schema_version", "started_at", "inputs", "outputs"}

    def test_writing_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        """The write is replace-on-complete, so a reader never sees half a
        file — and nothing else is left in the run directory."""
        write_run_record(tmp_path, make_record())

        assert sorted(p.name for p in tmp_path.iterdir()) == [RUN_FILE]

    def test_a_missing_file_is_a_run_record_error(self, tmp_path: Path) -> None:
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
        (tmp_path / RUN_FILE).write_text(content)

        with pytest.raises(RunRecordError):
            load_run(tmp_path)


class TestListRuns:
    """Test case 4: history newest-first; broken runs skipped, and named."""

    def write_run(self, root: Path, name: str, started_at: str) -> None:
        directory = root / name
        directory.mkdir()
        write_run_record(directory, make_record(started_at))

    def test_newest_first(self, tmp_path: Path) -> None:
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
        same = "2026-09-21T10:00:00.000000+00:00"
        for name in ("run_b", "run_a", "run_c"):
            self.write_run(tmp_path, name, same)

        names = [r.directory.name for r in list_runs(tmp_path).runs if r.directory]
        assert names == ["run_a", "run_b", "run_c"]

    def test_broken_runs_are_skipped_and_named(self, tmp_path: Path) -> None:
        """The history page must render with a crashed or half-written run in
        the folder — and say which ones it could not show."""
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
        (tmp_path / "notes.txt").write_text("hello")

        listing = list_runs(tmp_path)
        assert listing.runs == []
        assert listing.skipped == []

    def test_a_missing_root_is_an_empty_history(self, tmp_path: Path) -> None:
        """First launch: no runs yet is not an error."""
        listing = list_runs(tmp_path / "never_created")
        assert listing.runs == []
        assert listing.skipped == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_records.py -q`
Expected: `ImportError: cannot import name 'RUN_FILE'`.

- [ ] **Step 3: Implement**

- `RunRecord.to_json()` → `{"schema_version", "started_at", "inputs": asdict(inputs), "outputs": asdict(outputs)}` (`directory` is where it was loaded from, never serialised).
- `write_run_record`: `json.dumps(indent=2) + "\n"` into `run.json.tmp`, then `os.replace` → `run.json`.
- `load_run`: missing file, invalid JSON, non-object, `schema_version != 1`, missing sections, or unknown/missing fields (`TypeError` from the dataclass constructor) all become `RunRecordError` naming the file.
- `list_runs`: missing root → empty. Every sub-directory (sorted by name) is loaded; `RunRecordError` → `SkippedRun`. Runs sorted by `(started_at descending, name ascending)` — two stable sorts: by name, then by `started_at` with `reverse=True`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_records.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/records.py tests/unit/test_records.py
git commit -m "feat(records): run.json schema, load_run and list_runs"
```

---

### Task 3: Every CLI run writes its record

**Files:**
- Modify: `src/swarm_mapping/cli.py` (`run_pipeline`, new `describe_inputs`, `main`'s logging setup)
- Test: `tests/integration/test_run_records.py`

**Interfaces:**
- Consumes: Task 1 and 2.
- Produces: `describe_inputs(config_path: Path, config: ScenarioConfig, drones: int, view: bool) -> RunInputs`. `run_pipeline`'s signature and `MissionResult` are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
"""A real run writes its records: `log.jsonl`, `run.json`, paths and routes.

End to end through `run_pipeline` on `small_indoor`, capped at a few dozen
ticks so the whole module stays in seconds. The record's *contracts* (schema,
parsing, history listing) are unit-tested in `tests/unit/test_records.py`;
this module checks that the CLI fills them with the run it actually did.
"""

from __future__ import annotations

import collections
import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from swarm_mapping.cli import MissionResult, describe_inputs, run_pipeline
from swarm_mapping.config.loader import load_config
from swarm_mapping.records import LOG_FILE, RUN_FILE, load_run

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_INDOOR = REPO_ROOT / "scenarios" / "small_indoor" / "config.yaml"

# Past the CLI's 50-tick progress line, so the log holds more than the start.
CAPPED_TICKS = 60


def capped_config(directory: Path, max_ticks: int = CAPPED_TICKS) -> Path:
    """small_indoor, capped so a run takes a second or two."""
    raw: dict[str, Any] = copy.deepcopy(yaml.safe_load(SMALL_INDOOR.read_text()))
    raw["coordination"]["max_ticks"] = max_ticks
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def read_log(directory: Path) -> list[dict[str, Any]]:
    """The run's log, one parsed object per line."""
    text = (directory / LOG_FILE).read_text()
    return [json.loads(line) for line in text.splitlines()]


def run_json(directory: Path) -> dict[str, Any]:
    """The raw `run.json` document."""
    data: dict[str, Any] = json.loads((directory / RUN_FILE).read_text())
    return data


def without_wall_clock(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the two fields that measure the wall clock rather than the run."""
    out = copy.deepcopy(data)
    del out["started_at"]
    del out["outputs"]["wall_seconds"]
    return out


def snapshot_to_yaml(snapshot: dict[str, Any], directory: Path) -> Path:
    """Turn a run.json config snapshot back into a loadable scenario file.

    The snapshot is `dataclasses.asdict(ScenarioConfig)`, whose sections are the
    YAML sections field for field — only `scene_path` is spelled `scene.path`
    in YAML.
    """
    document = copy.deepcopy(snapshot)
    document["scene"] = {"path": document.pop("scene_path")}
    path = directory / "replay.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture(scope="module")
def two_drone_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, MissionResult]:
    """One capped two-drone run, shared by the read-only tests below."""
    workdir = tmp_path_factory.mktemp("two_drone")
    config = capped_config(workdir)
    output = workdir / "run"
    result = run_pipeline(config, output, drones=2)
    return config, output, result


class TestEveryRunIsRecorded:
    """Test case 1: each run writes both files into its own directory."""

    def test_both_files_are_written(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, _ = two_drone_run
        assert (output / RUN_FILE).is_file()
        assert (output / LOG_FILE).is_file()

    def test_files_lists_exactly_what_the_run_wrote(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, _ = two_drone_run
        listed = run_json(output)["outputs"]["files"]
        assert listed == sorted(p.name for p in output.iterdir())

    def test_a_second_run_leaves_the_first_record_untouched(
        self, two_drone_run: tuple[Path, Path, MissionResult], tmp_path: Path
    ) -> None:
        config, first, _ = two_drone_run
        before = (first / RUN_FILE).read_bytes(), (first / LOG_FILE).read_bytes()

        run_pipeline(config, tmp_path / "second", drones=1)

        after = (first / RUN_FILE).read_bytes(), (first / LOG_FILE).read_bytes()
        assert after == before
        assert load_run(tmp_path / "second").inputs.drones == 1

    def test_rerunning_into_the_same_directory_replaces_the_log(
        self, tmp_path: Path
    ) -> None:
        """`--output` keeps its meaning (R1): same DIR, files replaced — and the
        log is replaced too, never spliced onto the previous run's."""
        config = capped_config(tmp_path, max_ticks=5)
        output = tmp_path / "run"
        run_pipeline(config, output, drones=1)
        first = read_log(output)
        run_pipeline(config, output, drones=1)

        assert read_log(output) == first


class TestRunJson:
    """Test case 2 and R4: run.json holds the inputs and the outcome."""

    def test_outcome_matches_the_mission_result(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, result = two_drone_run
        outputs = run_json(output)["outputs"]

        assert outputs["ticks"] == result.ticks == CAPPED_TICKS
        assert outputs["coverage"] == result.coverage
        assert outputs["blocked"] == result.blocked
        assert outputs["unreachable_frontiers"] == result.unreachable_frontiers
        assert outputs["tick_capped"] == result.tick_capped
        assert outputs["succeeded"] == result.succeeded
        assert outputs["wall_seconds"] > 0

    def test_inputs_record_what_was_run(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        config, output, _ = two_drone_run
        inputs = run_json(output)["inputs"]

        assert inputs["config_path"] == str(config.resolve())
        assert inputs["drones"] == 2
        # small_indoor ships A+B: global allocation, 3-cell tolerance.
        assert inputs["assignment"] == "global"
        assert inputs["target_tolerance_cells"] == 3
        assert inputs["variant"] == "A+B"
        assert inputs["view"] is False
        assert inputs["config"] == json.loads(json.dumps(asdict(load_config(config))))

    def test_overrides_are_in_the_record_and_the_snapshot(self, tmp_path: Path) -> None:
        """A CLI override is part of the run; the snapshot is the config *as
        run*, not as written in the file."""
        output = tmp_path / "run"
        run_pipeline(
            capped_config(tmp_path, max_ticks=5),
            output,
            drones=1,
            assignment="greedy",
            target_tolerance=0,
        )
        inputs = run_json(output)["inputs"]

        assert inputs["variant"] == "baseline"
        assert inputs["config"]["coordination"]["assignment"] == "greedy"
        assert inputs["config"]["coordination"]["target_tolerance_cells"] == 0

    def test_per_drone_path_stats(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, result = two_drone_run
        paths = run_json(output)["outputs"]["paths"]

        assert sorted(paths) == ["0", "1"]
        for stats in paths.values():
            assert stats["ticks"] == result.ticks
            assert 1 <= stats["distinct_cells"] <= stats["route_length"]

    def test_event_counts_match_the_log(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, _ = two_drone_run
        counted = collections.Counter(line["event"] for line in read_log(output))

        assert run_json(output)["outputs"]["event_counts"] == dict(counted)

    def test_the_record_reproduces_the_map(
        self, two_drone_run: tuple[Path, Path, MissionResult], tmp_path: Path
    ) -> None:
        """Enough is recorded to re-run from run.json alone — config snapshot
        plus drone count — and get a byte-identical map."""
        _, output, _ = two_drone_run
        inputs = run_json(output)["inputs"]

        replay = tmp_path / "replay"
        run_pipeline(
            snapshot_to_yaml(inputs["config"], tmp_path),
            replay,
            drones=inputs["drones"],
        )

        assert (replay / "map.npz").read_bytes() == (output / "map.npz").read_bytes()


class TestLogFile:
    """Test case 3 (generic half): every line is JSON with `event` and `tick`.

    The failure half — `failure_injected` / `drone_failed` appear in a failure
    run — needs Features 9 and 10's events and is checked where they land.
    """

    def test_every_line_has_event_and_tick(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, _ = two_drone_run
        lines = read_log(output)

        assert lines
        for line in lines:
            assert isinstance(line["event"], str)
            assert isinstance(line["tick"], int)

    def test_ticks_follow_the_mission(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, result = two_drone_run
        ticks = [line["tick"] for line in read_log(output)]

        assert ticks == sorted(ticks)
        assert ticks[0] == 0, "the start line precedes the first tick"
        progress = [
            line for line in read_log(output) if line["event"].startswith("Tick 50/")
        ]
        assert [line["tick"] for line in progress] == [50]
        assert ticks[-1] <= result.ticks

    def test_structured_events_keep_their_fields(self, tmp_path: Path) -> None:
        """The master's own events arrive with their `extra=` payload.

        Uses a room split by a gap too narrow to fly — it ends `blocked` in
        well under a second, which guarantees a `mission_blocked` event.
        """
        scene = tmp_path / "blocked.xml"
        scene.write_text(BLOCKED_SCENE)
        raw: dict[str, Any] = yaml.safe_load(SMALL_INDOOR.read_text())
        raw["scene"]["path"] = str(scene)
        raw["drones"]["start_positions"] = [[-3.0, 0.0, 1.0]]
        raw["map"].update(
            {"origin_x": -5.0, "origin_y": -5.0, "grid_width": 100, "grid_height": 100}
        )
        raw["coordination"]["max_ticks"] = 400
        config = tmp_path / "blocked.yaml"
        config.write_text(yaml.safe_dump(raw))

        result = run_pipeline(config, tmp_path / "run")

        blocked = [
            line
            for line in read_log(tmp_path / "run")
            if line["event"] == "mission_blocked"
        ]
        assert blocked
        assert blocked[-1]["tick"] == result.ticks
        assert blocked[-1]["unreachable_frontiers"] == result.unreachable_frontiers
        assert blocked[-1]["level"] == "WARNING"


class TestPathsAlwaysRecorded:
    """R5: paths.json and one route PNG per drone, with or without a flag."""

    def test_written_without_any_flag(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        _, output, _ = two_drone_run
        assert (output / "paths.json").is_file()
        assert (output / "route_drone_0.png").is_file()
        assert (output / "route_drone_1.png").is_file()
        assert not list(output.glob("visits_drone_*.png")), (
            "visit heatmaps stay behind --visit-heatmaps"
        )

    def test_heatmaps_still_come_with_their_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "run"
        run_pipeline(
            capped_config(tmp_path, max_ticks=5),
            output,
            drones=1,
            visit_heatmaps=True,
        )

        assert (output / "visits_drone_0.png").is_file()
        assert (output / "route_drone_0.png").is_file()
        assert "visits_drone_0.png" in run_json(output)["outputs"]["files"]


class TestViewIsViewOnly:
    """R8, replacing test case 5 — a viewer cannot open headless.

    Two headless runs of the same inputs agree on everything but the wall
    clock; and `view` changes nothing in the record's inputs but `view`.
    """

    def test_two_headless_runs_have_identical_records(
        self, two_drone_run: tuple[Path, Path, MissionResult], tmp_path: Path
    ) -> None:
        config, first, _ = two_drone_run
        second = tmp_path / "again"
        run_pipeline(config, second, drones=2)

        assert without_wall_clock(run_json(first)) == without_wall_clock(
            run_json(second)
        )
        assert (first / LOG_FILE).read_bytes() == (second / LOG_FILE).read_bytes()

    def test_view_is_the_only_input_it_changes(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        config, _, _ = two_drone_run
        scenario = load_config(config)

        headless = describe_inputs(config, scenario, drones=2, view=False)
        viewed = describe_inputs(config, scenario, drones=2, view=True)

        assert viewed == replace(headless, view=True)
        assert headless.view is False


class TestStderrIsUnchanged:
    """The file log captures INFO; the terminal still shows only what it did."""

    def test_quiet_cli_prints_no_info_but_logs_it(self, tmp_path: Path) -> None:
        config = capped_config(tmp_path, max_ticks=3)
        output = tmp_path / "run"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "swarm_mapping.cli",
                "--config",
                str(config),
                "--output",
                str(output),
                "--drones",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Capped at 3 ticks: the mission did not converge, exit code 1 as ever.
        assert completed.returncode == 1, completed.stderr
        assert "Starting exploration" not in completed.stderr
        assert "INFO" not in completed.stderr
        events = [line["event"] for line in read_log(output)]
        assert any(event.startswith("Starting exploration") for event in events)


BLOCKED_SCENE = """<mujoco model="blocked">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    <geom name="wall_east" type="box" pos="5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_west" type="box" pos="-5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_north" type="box" pos="0 5 1.5" size="5 0.1 1.5"/>
    <geom name="wall_south" type="box" pos="0 -5 1.5" size="5 0.1 1.5"/>
    <geom name="divider_north" type="box" pos="0 2.55 1.5" size="0.1 2.45 1.5"/>
    <geom name="divider_south" type="box" pos="0 -2.55 1.5" size="0.1 2.45 1.5"/>
  </worldbody>
</mujoco>
"""
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/integration/test_run_records.py -q`
Expected: `ImportError: cannot import name 'describe_inputs'`.

- [ ] **Step 3: Implement**

1. `run_pipeline`: record `started_at` (UTC) and a monotonic start; `output_dir.mkdir(parents=True, exist_ok=True)` moves to the top; the whole body runs inside `with capture_run_log(output_dir / LOG_FILE) as run_log:`.
2. The loop sets `run_log.tick = master.tick_count + 1` before each `master.tick()` — the master increments its counter first thing in `tick()`, so this is the tick being executed and agrees with the `tick` extras the master already passes.
3. `PathLog()` unconditionally; `paths.json` and `route_drone_<id>.png` for every drone. Heatmaps, and every existing `print`, stay behind `--visit-heatmaps` so stdout is unchanged.
4. After export and before the post-mission viewer wait: `write_run_record(output_dir, RunRecord(started_at, describe_inputs(...), RunOutputs(...)))`. `files` is the sorted list of names this run wrote, including `log.jsonl` and `run.json`.
5. `main`: build the stderr `StreamHandler` explicitly with `setLevel(INFO if verbose else WARNING)` and pass it to `basicConfig(handlers=[...])`. Same format, same output — but the level now lives on the handler, so the file log lowering the package logger's level cannot leak INFO to the terminal.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/integration/test_run_records.py tests/integration/test_e2e.py -q`
Expected: all pass; every existing e2e test unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/cli.py tests/integration/test_run_records.py
git commit -m "feat(cli): every run writes log.jsonl, run.json and its paths"
```

---

### Task 4: `validate_scenario`

**Files:**
- Create: `src/swarm_mapping/app/__init__.py`, `src/swarm_mapping/app/validation.py`
- Test: `tests/integration/test_scenario_validation.py` (all but `TestInstallScenario`)

**Interfaces:**
- Produces: `validate_scenario(config_path: str | Path, *, require_five: bool = False) -> list[str]`; `UPLOAD_DRONES = 5`.

- [ ] **Step 1: Write the failing tests**

```python
"""Scenario validation and installation — the D2 rules for uploaded rooms.

Uses the real MuJoCo scenes and small inline MJCF; nothing is mocked. The one
monkeypatch below is a tripwire, not a stand-in: it fails the test if the
validator reaches the simulator when it should have stopped first.

Every write goes into pytest tmp dirs — `install_scenario` takes both roots as
arguments precisely so tests never touch the real `scenarios/` or assets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from swarm_mapping.app import validation
from swarm_mapping.app.validation import install_scenario, validate_scenario
from swarm_mapping.cli import resolve_scene_path

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = REPO_ROOT / "scenarios"
SHIPPED = sorted(p.parent.name for p in SCENARIOS.glob("*/config.yaml"))


def scenario_dict(name: str) -> dict[str, Any]:
    """A shipped scenario's config as a raw dict, for targeted edits."""
    raw: dict[str, Any] = yaml.safe_load((SCENARIOS / name / "config.yaml").read_text())
    return copy.deepcopy(raw)


def write_config(directory: Path, raw: dict[str, Any]) -> Path:
    """Write a raw config dict and return its path."""
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


class TestShippedScenarios:
    """Test case 6: every shipped scenario validates clean."""

    def test_there_are_shipped_scenarios(self) -> None:
        assert len(SHIPPED) >= 4

    @pytest.mark.parametrize("name", SHIPPED)
    def test_validates_clean(self, name: str) -> None:
        assert validate_scenario(SCENARIOS / name / "config.yaml") == []

    @pytest.mark.parametrize("name", ["large_indoor", "loop_indoor"])
    def test_five_spawn_scenarios_meet_the_upload_rule(self, name: str) -> None:
        """The two shipped scenarios that declare five spawns would be accepted
        as uploads too."""
        path = SCENARIOS / name / "config.yaml"
        assert validate_scenario(path, require_five=True) == []


class TestSpawnCount:
    """Test case 7: "up to 5 drones" means five usable start positions."""

    def test_fewer_than_five_is_rejected_with_the_count(self) -> None:
        problems = validate_scenario(
            SCENARIOS / "small_indoor" / "config.yaml", require_five=True
        )

        assert len(problems) == 1
        assert "3" in problems[0]
        assert "5" in problems[0]


class TestSpawnInsideGeometry:
    """Test case 8: a spawn inside a wall is named by index."""

    def test_spawn_in_a_wall_is_rejected_by_index(self, tmp_path: Path) -> None:
        raw = scenario_dict("small_indoor")
        # small_indoor's east wall is centred on x = 10 with 0.1 m half-width.
        raw["drones"]["start_positions"][1] = [9.95, 0.0, 1.0]

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "start_positions[1]" in problems[0]
        assert "wall_east" in problems[0]

    def test_geometry_is_checked_on_the_compiled_scene(self, tmp_path: Path) -> None:
        """The obstacle's position comes from its parent body's frame, so only
        the compiled model knows where it is — reading the geom's own `pos`
        attribute would put it at the origin and miss the spawn entirely."""
        scene = tmp_path / "pillar.xml"
        scene.write_text(PILLAR_SCENE)
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = str(scene)
        raw["drones"]["start_positions"] = [
            [-3.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, -3.0, 1.0],
        ]

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "start_positions[1]" in problems[0]
        assert "pillar" in problems[0]

    def test_every_offending_spawn_is_reported(self, tmp_path: Path) -> None:
        raw = scenario_dict("small_indoor")
        raw["drones"]["start_positions"][0] = [9.95, 0.0, 1.0]
        raw["drones"]["start_positions"][2] = [-9.95, 0.0, 1.0]

        problems = validate_scenario(write_config(tmp_path, raw))

        joined = "\n".join(problems)
        assert "start_positions[0]" in joined
        assert "start_positions[2]" in joined
        assert "start_positions[1]" not in joined


class TestSpawnSeparation:
    """Test case 9: two spawns closer than `min_separation` are rejected."""

    def test_spawns_too_close_are_rejected(self, tmp_path: Path) -> None:
        raw = scenario_dict("small_indoor")
        raw["drones"]["start_positions"][1] = [0.0, 0.3, 1.0]  # sep 0.5 m

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "min_separation" in problems[0]


class TestMissingScene:
    """Test case 10: a missing scene is caught before MuJoCo is touched."""

    def test_missing_scene_is_rejected_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def tripwire(*_args: object, **_kwargs: object) -> None:
            pytest.fail("the validator built a simulation for a missing scene")

        monkeypatch.setattr(validation, "SimulationEngine", tripwire)
        monkeypatch.setattr(validation, "build_mission", tripwire)
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = "no_such_room.xml"

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "no_such_room.xml" in problems[0]
        assert str(resolve_scene_path("no_such_room.xml")) in problems[0]


class TestNeverRaises:
    """R7: bad input is a list of problems, never an exception."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        problems = validate_scenario(tmp_path / "absent.yaml")
        assert len(problems) == 1
        assert "absent.yaml" in problems[0]

    @pytest.mark.parametrize(
        "text", ["key: [unclosed", "- just\n- a list\n", ""], ids=str
    )
    def test_unparseable_config(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(text)
        assert len(validate_scenario(path)) == 1

    def test_schema_error_names_the_key(self, tmp_path: Path) -> None:
        raw = scenario_dict("small_indoor")
        del raw["sensor"]["max_range"]

        (problem,) = validate_scenario(write_config(tmp_path, raw))
        assert "sensor.max_range" in problem

    def test_broken_scene_xml(self, tmp_path: Path) -> None:
        scene = tmp_path / "broken.xml"
        scene.write_text("<mujoco><worldbody><geom type='nonsense'/></worldbody>")
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = str(scene)

        (problem,) = validate_scenario(write_config(tmp_path, raw))
        assert "scene" in problem


PILLAR_SCENE = """<mujoco model="pillar">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <body name="pillar_mount" pos="3 0 0">
      <geom name="pillar" type="box" pos="0 0 1.5" size="0.5 0.5 1.5"/>
    </body>
  </worldbody>
</mujoco>
"""
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/integration/test_scenario_validation.py -q`
Expected: `ModuleNotFoundError: No module named 'swarm_mapping.app'`.

- [ ] **Step 3: Implement** — in order, returning early where later checks would be meaningless:

1. `load_config`: `OSError` → "cannot read config ..."; `yaml.YAMLError` → "config is not valid YAML ..."; `ValueError` → its message (already names file and key). Stop.
2. `resolve_scene_path(config.scene_path)` not a file → problem naming `scene.path` and the resolved path. Stop — MuJoCo is never reached.
3. `require_five` and `count != 5` → "declares N start position(s); an uploaded room must declare exactly 5 ...". Continue.
4. `SimulationEngine(scene, all spawns)` (its constructor runs `mj_forward`, which fills `data.contact`). A load failure → "scene failed to load: ..." and stop. For each contact where exactly one geom belongs to a drone body, record `(spawn index, other geom name)`; one problem per spawn, sorted by index, naming every geom it touches.
5. `build_mission(config, drones=count)`; `ValueError` → its message (separation floor, spawn separation, planner/sensor floors).

- [ ] **Step 4: Run to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/app tests/integration/test_scenario_validation.py
git commit -m "feat(app): validate_scenario — spawns checked on the compiled scene"
```

---

### Task 5: `install_scenario`

**Files:**
- Modify: `src/swarm_mapping/app/validation.py`
- Test: `tests/integration/test_scenario_validation.py` (`TestInstallScenario`)

**Interfaces:**
- Produces: `install_scenario(name: str, config_text: str, scene_text: str, scenarios_root: Path, assets_dir: Path) -> list[str]`; `SCENARIO_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")`.

- [ ] **Step 1: Write the failing tests**

```python
class TestInstallScenario:
    """R6 and test case 11: uploads are validated first and never overwrite."""

    @pytest.fixture
    def roots(self, tmp_path: Path) -> tuple[Path, Path]:
        scenarios_root = tmp_path / "scenarios"
        assets_dir = tmp_path / "assets"
        scenarios_root.mkdir()
        assets_dir.mkdir()
        return scenarios_root, assets_dir

    @staticmethod
    def upload(name: str = "large_indoor") -> tuple[str, str]:
        """A valid five-spawn upload: a shipped config and its scene."""
        config_text = (SCENARIOS / name / "config.yaml").read_text()
        scene_text = resolve_scene_path(f"{name}.xml").read_text()
        return config_text, scene_text

    @staticmethod
    def everything_under(*roots: Path) -> list[Path]:
        return sorted(p for root in roots for p in root.rglob("*"))

    def test_a_valid_room_is_installed(self, roots: tuple[Path, Path]) -> None:
        scenarios_root, assets_dir = roots
        config_text, scene_text = self.upload()

        problems = install_scenario(
            "my_room", config_text, scene_text, scenarios_root, assets_dir
        )

        assert problems == []
        assert (assets_dir / "my_room.xml").read_text() == scene_text
        installed = yaml.safe_load(
            (scenarios_root / "my_room" / "config.yaml").read_text()
        )
        original = yaml.safe_load(config_text)
        assert installed["scene"] == {"path": "my_room.xml"}
        del installed["scene"], original["scene"]
        assert installed == original

    @pytest.mark.parametrize(
        "name",
        ["../escape", "a/b", "Room", "1room", "", "room-1", "r" * 42, "room.xml"],
    )
    def test_bad_names_are_rejected_and_nothing_written(
        self, roots: tuple[Path, Path], name: str
    ) -> None:
        config_text, scene_text = self.upload()

        problems = install_scenario(name, config_text, scene_text, *roots)

        assert problems
        assert self.everything_under(*roots) == []

    def test_an_existing_scenario_is_never_overwritten(
        self, roots: tuple[Path, Path]
    ) -> None:
        scenarios_root, assets_dir = roots
        existing = scenarios_root / "small_indoor"
        existing.mkdir()
        (existing / "config.yaml").write_text("shipped")
        config_text, scene_text = self.upload()

        problems = install_scenario(
            "small_indoor", config_text, scene_text, scenarios_root, assets_dir
        )

        assert any("already exists" in p for p in problems)
        assert (existing / "config.yaml").read_text() == "shipped"
        assert list(assets_dir.iterdir()) == []

    def test_an_existing_scene_file_is_never_overwritten(
        self, roots: tuple[Path, Path]
    ) -> None:
        scenarios_root, assets_dir = roots
        (assets_dir / "my_room.xml").write_text("shipped scene")
        config_text, scene_text = self.upload()

        problems = install_scenario(
            "my_room", config_text, scene_text, scenarios_root, assets_dir
        )

        assert any("already exists" in p for p in problems)
        assert (assets_dir / "my_room.xml").read_text() == "shipped scene"
        assert list(scenarios_root.iterdir()) == []

    def test_an_invalid_room_writes_nothing(self, roots: tuple[Path, Path]) -> None:
        """Three spawns fails the upload rule; the problem says so and the
        roots stay empty."""
        config_text, scene_text = self.upload("small_indoor")

        problems = install_scenario("my_room", config_text, scene_text, *roots)

        assert len(problems) == 1
        assert "3" in problems[0]
        assert self.everything_under(*roots) == []

    def test_an_unparseable_config_writes_nothing(
        self, roots: tuple[Path, Path]
    ) -> None:
        _, scene_text = self.upload()

        problems = install_scenario("my_room", "key: [unclosed", scene_text, *roots)

        assert problems
        assert self.everything_under(*roots) == []
```

- [ ] **Step 2: Run to verify they fail**

Expected: `ImportError: cannot import name 'install_scenario'`.

- [ ] **Step 3: Implement**

1. Name fails `SCENARIO_NAME.fullmatch` → problem, stop. (The pattern admits no `/` or `.`, so `../` is rejected here.)
2. `scenarios_root/<name>` or `assets_dir/<name>.xml` exists → "already exists" problem(s), stop.
3. `yaml.safe_load(config_text)`: a YAML error, a non-mapping, or a missing/non-mapping `scene` section → problem, stop.
4. In a `TemporaryDirectory`: write the scene, set `scene.path` to its absolute path, write the config, `validate_scenario(..., require_five=True)`. Any problem → return them; nothing outside the temp dir was written.
5. Write the scene with mode `"x"` (exclusive create — never overwrite, even in a race), `mkdir(exist_ok=False)` the scenario directory, write `config.yaml` with `scene.path: <name>.xml` via `yaml.safe_dump(sort_keys=False)`.

- [ ] **Step 4: Run to verify they pass** — then the full pre-commit check.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/app/validation.py tests/integration/test_scenario_validation.py
git commit -m "feat(app): install_scenario — validate first, never overwrite"
```

---

## Done when

- Every `run_pipeline` call leaves `log.jsonl`, `run.json`, `paths.json` and one route PNG per drone in its output directory; `map.npz`/`map.png`, exit codes, stdout and stderr are unchanged.
- `list_runs` / `load_run` read them back, skipping and naming broken runs.
- `validate_scenario` accepts every shipped scenario and rejects each D2 violation with a message naming it; `install_scenario` writes nothing unless the room is clean and never overwrites.
- Nothing imports `swarm_mapping.app`; nothing imports streamlit.
- ruff, mypy and the non-acceptance suite are green.

What Feature 12 deliberately does **not** do: anything failure-specific. Failure events reach `event_counts` automatically once Features 9/10 log them; failure latency is computed from `log.jsonl` by Feature 11. No new CLI flag, no config-schema change, no UI.
