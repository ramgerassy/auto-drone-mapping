"""What the console shows about runs, as plain data.

The History page's table rows, the two-run comparison, the live progress of a
running mission and the per-drone route images are all decided here, so the
Streamlit page only lays them out. Pure Python, no streamlit; reads files but
never writes one.

Everything read from `log.jsonl` goes by event *name* and structured fields
(`tick`, `coverage`), never by message text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_mapping.records import RunRecord

FAILURE_EVENT = "drone_failed"
"""The coordinator's event for a failure it detected."""

PROGRESS_EVENT = "mission_progress"
"""The CLI's periodic progress event; carries `coverage`."""


def scenario_name(record: RunRecord) -> str:
    """The scenario a run used: the directory holding its config.

    `run.json` stores the config's path, and every scenario lives in
    `<scenarios root>/<name>/config.yaml`.
    """
    return Path(record.inputs.config_path).parent.name


def failures_detected(record: RunRecord) -> int:
    """How many drone failures the coordinator detected during the run."""
    return record.outputs.event_counts.get(FAILURE_EVENT, 0)


def history_row(record: RunRecord) -> dict[str, object]:
    """One row of the History table.

    Args:
        record: A loaded run.

    Returns:
        Column name to value, in display order.
    """
    outputs = record.outputs
    return {
        "started": record.started_at,
        "run": record.directory.name if record.directory else "",
        "scenario": scenario_name(record),
        "variant": record.inputs.variant,
        "drones": record.inputs.drones,
        "ticks": outputs.ticks,
        "coverage": outputs.coverage,
        "blocked": outputs.blocked,
        "succeeded": outputs.succeeded,
        "wall seconds": outputs.wall_seconds,
        "failures detected": failures_detected(record),
    }


def _metrics(record: RunRecord) -> dict[str, object]:
    """The values two runs are compared on, in display order."""
    outputs = record.outputs
    route_total = sum(
        int(stats.get("route_length", 0)) for stats in outputs.paths.values()
    )
    return {
        "scenario": scenario_name(record),
        "variant": record.inputs.variant,
        "drones": record.inputs.drones,
        "ticks": outputs.ticks,
        "coverage": outputs.coverage,
        "blocked": outputs.blocked,
        "succeeded": outputs.succeeded,
        "unreachable frontiers": outputs.unreachable_frontiers,
        "tick capped": outputs.tick_capped,
        "total route length": route_total,
        "failures detected": failures_detected(record),
        "wall seconds": outputs.wall_seconds,
    }


@dataclass(frozen=True)
class MetricDelta:
    """One metric of a two-run comparison.

    Attributes:
        metric: The metric's name.
        a: Its value in the first run.
        b: Its value in the second run.
        difference: `b - a` for numbers; None for text and flags, where a
            difference means nothing.
    """

    metric: str
    a: object
    b: object
    difference: float | None


def compare(a: RunRecord, b: RunRecord) -> list[MetricDelta]:
    """Compare two runs metric by metric.

    Args:
        a: The first run.
        b: The second run.

    Returns:
        One entry per metric, in display order.
    """
    left, right = _metrics(a), _metrics(b)
    deltas = []
    for metric, value_a in left.items():
        value_b = right[metric]
        difference: float | None = None
        # bool is an int subclass; True - False is not a meaningful difference.
        numeric = (
            isinstance(value_a, int | float)
            and isinstance(value_b, int | float)
            and not isinstance(value_a, bool)
            and not isinstance(value_b, bool)
        )
        if numeric:
            difference = float(value_b) - float(value_a)  # type: ignore[arg-type]
        deltas.append(MetricDelta(metric, value_a, value_b, difference))
    return deltas


def read_log(path: Path, last: int | None = None) -> list[dict[str, Any]]:
    """Parse a `log.jsonl`, tolerating a run that is still writing it.

    Args:
        path: The log file. Need not exist yet.
        last: Keep only the final `last` entries; all of them if None.

    Returns:
        The log entries in file order. A line that is not a JSON object —
        typically the half-written final line of a running mission — is
        skipped, never raised.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries if last is None else entries[-last:] if last > 0 else []


@dataclass(frozen=True)
class Progress:
    """A running mission's progress, as far as its log shows.

    Attributes:
        tick: The tick of the latest log entry; None before the first.
        coverage: The latest `mission_progress` coverage; None before one.
    """

    tick: int | None
    coverage: float | None


def progress(entries: list[dict[str, Any]]) -> Progress:
    """Read a mission's progress from its log entries.

    Args:
        entries: `read_log` output, in file order.

    Returns:
        The latest tick and coverage seen.
    """
    tick = None
    for entry in reversed(entries):
        if isinstance(entry.get("tick"), int):
            tick = entry["tick"]
            break
    coverage = None
    for entry in reversed(entries):
        value = entry.get("coverage")
        if entry.get("event") == PROGRESS_EVENT and isinstance(value, int | float):
            coverage = float(value)
            break
    return Progress(tick=tick, coverage=coverage)


def event_names(entries: list[dict[str, Any]]) -> list[str]:
    """The distinct event names in a log, sorted, for the log viewer's filter."""
    return sorted({str(entry.get("event")) for entry in entries})


def route_images(record: RunRecord) -> list[tuple[str, Path]]:
    """Each drone's route PNG that exists, in drone-id order.

    Args:
        record: A loaded run (with `directory` set).

    Returns:
        `(drone id, path)` pairs; ids sorted numerically so drone 10 follows
        drone 2. Drones whose image is missing are left out.
    """
    if record.directory is None:
        return []
    images = []
    for drone in sorted(record.outputs.paths, key=int):
        image = record.directory / f"route_drone_{drone}.png"
        if image.is_file():
            images.append((drone, image))
    return images
