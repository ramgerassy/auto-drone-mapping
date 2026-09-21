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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LOG_FILE = "log.jsonl"
"""Name of the per-run JSON Lines log inside the output directory."""

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
            "event": record.getMessage(),
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
        self.counts[record.getMessage()] += 1


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
