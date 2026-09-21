"""Run records: what every run leaves behind in its output directory.

Two files, written by `cli.run_pipeline` and read back by the operator console:

- `log.jsonl` — every log record the run emitted, one JSON object per line.
  This is CLAUDE.md's logging convention ("JSON Lines, per-run output
  directory"), which the stderr-only logging of Sprints 1-2 never met.
- `run.json` — the run's inputs and outcome, enough to compare runs and to
  reproduce one.

This module owns the *format* of both. It imports no domain module: it
serialises plain data handed to it, so the console can read a run history
without importing the simulator, and nothing here can perturb a mission.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOG_FILE = "log.jsonl"
"""Name of the per-run JSON Lines log inside the output directory."""

RUN_FILE = "run.json"
"""Name of the per-run summary inside the output directory."""

SCHEMA_VERSION = 1
"""Version of the `run.json` layout. Bump on any incompatible change, so a
reader refuses a record it would misread rather than guessing."""

VARIANTS: dict[str, tuple[str, int]] = {
    "baseline": ("greedy", 0),
    "A": ("global", 0),
    "B": ("greedy", 3),
    "A+B": ("global", 3),
}
"""The allocation variants by label, as `(assignment, target_tolerance_cells)`.

The sprint plan's table (`docs/sprint-3-plan.md`, D2). These are allocation
variants, not frontier strategies — Sprint 2.5 showed `FrontierStrategy` is not
what they change. Kept here, beside the record that stores the label, so the
console's selector and the record can never disagree about what "B" means.
"""

# The package's own logger tree. Third-party loggers (matplotlib, MuJoCo's
# Python bindings) are deliberately outside it: their chatter is not the run.
_PACKAGE_LOGGER = "swarm_mapping"

# Attributes every LogRecord carries, derived from a blank one rather than
# listed by hand so a Python upgrade that adds one (3.12 added `taskName`) is
# excluded automatically. `message` and `asctime` are set later by formatters.
# Anything *not* in this set was passed by the call site via `extra=`.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", None, None).__dict__
) | {"message", "asctime"}


def _jsonable(value: object) -> object:
    """Fallback encoder: never let a log call crash a mission.

    Numpy scalars become their Python equivalent, so `np.int64(4)` is logged as
    `4` rather than `"4"`. Anything else unencodable becomes its `str()`.
    """
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _event_name(record: logging.LogRecord) -> str:
    """The event a record reports: its `event` extra, else its message.

    Prose messages meant for a terminal ("Tick 50/60 — coverage 12.3%") carry a
    stable machine name in `extra={"event": ...}`, so the log file and
    `event_counts` get one key per kind of event rather than one per line.
    Structured call sites (`_LOGGER.info("mission_blocked", ...)`) already use
    the name as the message and need no extra.
    """
    event = getattr(record, "event", None)
    return str(event) if event else record.getMessage()


class _TickStamp(logging.Filter):
    """Stamps the loop's current tick on every record that lacks one.

    The coordinator passes `tick` on only a few of its events, and most call
    sites have no tick to pass. Stamping here — rather than at each call site —
    is what guarantees *every* line of the log can be placed on the timeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tick = 0

    def filter(self, record: logging.LogRecord) -> bool:
        """Add `tick` unless the call site already passed its own."""
        if not hasattr(record, "tick"):
            record.tick = self.tick
        return True


class _JsonLinesFormatter(logging.Formatter):
    """One JSON object per record: `event`, `tick`, `level`, then extras."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a single line of JSON."""
        payload: dict[str, object] = {
            "event": _event_name(record),
            "tick": getattr(record, "tick", None),
            "level": record.levelname,
        }
        # `__dict__` preserves insertion order, which is the call site's
        # `extra=` order — so the same call always renders the same line.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_jsonable)


class _JsonLinesHandler(logging.FileHandler):
    """Writes JSON Lines and tallies how many lines each event produced."""

    def __init__(self, path: Path) -> None:
        # mode="w": re-running into the same --output replaces the log, as it
        # replaces map.npz. Appending would splice two runs into one file.
        super().__init__(path, mode="w", encoding="utf-8")
        self.counts: collections.Counter[str] = collections.Counter()

    def emit(self, record: logging.LogRecord) -> None:
        """Write the record, then count it under its event name."""
        super().emit(record)
        self.counts[_event_name(record)] += 1


class RunLog:
    """Handle on an active run log, returned by `capture_run_log`.

    The run loop sets `tick` before each coordinator tick; every record logged
    until the next update is stamped with it.
    """

    def __init__(self, handler: _JsonLinesHandler, stamp: _TickStamp) -> None:
        self._handler = handler
        self._stamp = stamp

    @property
    def tick(self) -> int:
        """The tick stamped on records whose call site passed none."""
        return self._stamp.tick

    @tick.setter
    def tick(self, value: int) -> None:
        self._stamp.tick = value

    def event_counts(self) -> dict[str, int]:
        """Lines written so far, per event, with keys sorted.

        Sorted so `run.json` is byte-stable across identical runs.

        Returns:
            Event name to line count.
        """
        return dict(sorted(self._handler.counts.items()))


@contextmanager
def capture_run_log(path: Path, logger_name: str = _PACKAGE_LOGGER) -> Iterator[RunLog]:
    """Write the package's log records to `path` as JSON Lines, for one run.

    Captures INFO and above whatever `--verbose` says: the file is the complete
    record, the terminal is the operator's summary. To get INFO records at all
    when the logger tree is set quieter, the named logger's level is lowered to
    INFO for the duration and restored afterwards. That lets INFO propagate
    further up too, so a caller that must keep INFO off stderr sets the level on
    its stderr *handler* — `cli.main` does.

    The handler is removed and the file closed on exit, including when the run
    raises, so a later run never writes into this one's log.

    Args:
        path: Destination file. Replaced if it exists.
        logger_name: Root of the logger tree to capture.

    Yields:
        The active run log, whose `tick` the caller advances.
    """
    logger = logging.getLogger(logger_name)
    handler = _JsonLinesHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_JsonLinesFormatter())
    stamp = _TickStamp()
    handler.addFilter(stamp)

    previous_level = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield RunLog(handler, stamp)
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)


def variant_label(assignment: str, target_tolerance_cells: int) -> str:
    """Name the allocation variant a run used.

    Args:
        assignment: `coordination.assignment` as run.
        target_tolerance_cells: `coordination.target_tolerance_cells` as run.

    Returns:
        "baseline", "A", "B" or "A+B" for the four pairs in `VARIANTS`, and
        "custom" for anything else — a tolerance of 1 is not silently "B".
    """
    for label, pair in VARIANTS.items():
        if pair == (assignment, target_tolerance_cells):
            return label
    return "custom"


class RunRecordError(ValueError):
    """A run directory's `run.json` is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class RunInputs:
    """Everything that decides a run's outcome.

    Attributes:
        config_path: Absolute path of the scenario YAML the run loaded.
        config: `dataclasses.asdict` of the `ScenarioConfig` actually run, with
            CLI overrides applied. The file at `config_path` may change later;
            this snapshot is what the run used.
        drones: Drones flown. With the snapshot's start positions this pins the
            swarm, because `--drones N` takes the first N.
        assignment: `coordination.assignment` as run.
        target_tolerance_cells: `coordination.target_tolerance_cells` as run.
        variant: `variant_label` of the two fields above.
        view: Whether the live viewer was open. View-only: it changes nothing
            else in the record.
    """

    config_path: str
    config: dict[str, Any]
    drones: int
    assignment: str
    target_tolerance_cells: int
    variant: str
    view: bool


@dataclass(frozen=True)
class RunOutputs:
    """What the run produced.

    Attributes:
        ticks: Ticks executed.
        coverage: Fraction of cells classified free or occupied.
        blocked: The mission ended with frontiers it could not reach.
        unreachable_frontiers: Frontier regions left at termination.
        tick_capped: The mission stopped at `max_ticks`.
        succeeded: `MissionResult.succeeded` — the mission reached its own end.
        wall_seconds: Wall-clock duration from config load to export. With
            `RunRecord.started_at`, the only field that differs between two
            identical runs.
        paths: Per-drone `PathLog.summary()`, keyed by drone id as a string
            (JSON object keys are strings).
        event_counts: Log lines per event, as in `log.jsonl`.
        files: Names of the files this run wrote into its directory, sorted.
    """

    ticks: int
    coverage: float
    blocked: bool
    unreachable_frontiers: int
    tick_capped: bool
    succeeded: bool
    wall_seconds: float
    paths: dict[str, dict[str, int | float]]
    event_counts: dict[str, int]
    files: list[str]


@dataclass(frozen=True)
class RunRecord:
    """One run's `run.json`.

    Attributes:
        started_at: UTC start time, ISO-8601. Orders the run history.
        inputs: What was run.
        outputs: What came of it.
        directory: The run directory this record was loaded from; None for a
            record not yet written. Not serialised — the directory is where
            the file is, not something the file says.
    """

    started_at: str
    inputs: RunInputs
    outputs: RunOutputs
    directory: Path | None = field(default=None, compare=False)

    def to_json(self) -> dict[str, Any]:
        """The `run.json` document.

        Returns:
            A JSON-serialisable dict with `schema_version` first.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "started_at": self.started_at,
            "inputs": asdict(self.inputs),
            "outputs": asdict(self.outputs),
        }


def write_run_record(directory: Path, record: RunRecord) -> Path:
    """Write `run.json` into a run directory.

    Written to a temporary name and renamed into place: `os.replace` is atomic,
    so the console's history page, reading while a run finishes, sees either no
    record or a whole one — never half a file.

    Args:
        directory: The run's output directory. Must exist.
        record: The record to write.

    Returns:
        The path of the written `run.json`.
    """
    target = directory / RUN_FILE
    temporary = directory / f"{RUN_FILE}.tmp"
    temporary.write_text(json.dumps(record.to_json(), indent=2) + "\n")
    os.replace(temporary, target)
    return target


def load_run(directory: Path) -> RunRecord:
    """Read a run directory's `run.json`.

    Args:
        directory: A run's output directory.

    Returns:
        The record, with `directory` set.

    Raises:
        RunRecordError: If the file is missing or unreadable, is not valid
            JSON, has another `schema_version`, or lacks or adds fields. The
            message names the file.
    """
    path = directory / RUN_FILE
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        msg = f"{path}: no {RUN_FILE} (the run crashed, or is still running)"
        raise RunRecordError(msg) from exc
    except (OSError, ValueError) as exc:
        msg = f"{path}: unreadable: {exc}"
        raise RunRecordError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{path}: expected a JSON object, got {type(data).__name__}"
        raise RunRecordError(msg)
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = (
            f"{path}: schema_version {version!r}; this reader handles {SCHEMA_VERSION}"
        )
        raise RunRecordError(msg)
    try:
        return RunRecord(
            started_at=str(data["started_at"]),
            inputs=RunInputs(**data["inputs"]),
            outputs=RunOutputs(**data["outputs"]),
            directory=directory,
        )
    except (KeyError, TypeError) as exc:
        msg = f"{path}: malformed record: {exc}"
        raise RunRecordError(msg) from exc


@dataclass(frozen=True)
class SkippedRun:
    """A directory `list_runs` could not show, and why.

    Attributes:
        directory: The run directory.
        reason: The `RunRecordError` message.
    """

    directory: Path
    reason: str


@dataclass(frozen=True)
class RunListing:
    """The run history under one root.

    Attributes:
        runs: Loaded runs, newest first; ties broken by directory name.
        skipped: Directories without a readable record, by directory name.
    """

    runs: list[RunRecord]
    skipped: list[SkippedRun]


def list_runs(root: Path) -> RunListing:
    """Load every run directly under `root`, newest first.

    A directory with a missing or broken `run.json` — a crashed run, one still
    in progress, a hand-edited file — is skipped and reported rather than
    raised, so one bad run cannot take down the history page.

    Args:
        root: The directory holding one sub-directory per run. Need not exist:
            no runs yet is an empty history, not an error.

    Returns:
        The loaded runs and the skipped directories.
    """
    if not root.is_dir():
        return RunListing(runs=[], skipped=[])

    runs: list[RunRecord] = []
    skipped: list[SkippedRun] = []
    # Sorted by name first: the filesystem's order is arbitrary, and the stable
    # sort below then leaves equal timestamps in name order.
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            runs.append(load_run(directory))
        except RunRecordError as exc:
            skipped.append(SkippedRun(directory=directory, reason=str(exc)))

    # ISO-8601 UTC timestamps in one format sort correctly as strings.
    runs.sort(key=lambda run: run.started_at, reverse=True)
    return RunListing(runs=runs, skipped=skipped)
