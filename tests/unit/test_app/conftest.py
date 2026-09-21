"""Hand-written run directories for the console's tests.

Built with Feature 12's own writer (`write_run_record`), so a fixture run is
exactly what `list_runs` reads from a real one — without running a mission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from swarm_mapping.records import (
    LOG_FILE,
    RunInputs,
    RunOutputs,
    RunRecord,
    write_run_record,
)

LOG_LINES: list[dict[str, Any]] = [
    {"event": "mission_started", "tick": 0, "level": "INFO", "drones": 2},
    {"event": "frontier_assigned", "tick": 1, "level": "INFO", "drone_id": 0},
    {"event": "mission_progress", "tick": 50, "level": "INFO", "coverage": 0.4},
    {"event": "drone_failed", "tick": 60, "level": "WARNING", "drone_id": 1},
    {"event": "frontier_assigned", "tick": 61, "level": "INFO", "drone_id": 0},
]


def _png(path: Path, colour: tuple[int, int, int]) -> None:
    """A tiny solid-colour PNG, standing in for a rendered map or route."""
    Image.new("RGB", (8, 8), colour).save(path)


def make_run(
    root: Path,
    name: str,
    *,
    started_at: str = "2026-09-21T10:00:00.000000+00:00",
    variant: str = "A+B",
    ticks: int = 120,
    coverage: float = 0.95,
    event_counts: dict[str, int] | None = None,
) -> Path:
    """Write a complete, valid run directory under `root`.

    Returns:
        The run directory.
    """
    directory = root / name
    directory.mkdir(parents=True)
    _png(directory / "map.png", (200, 200, 200))
    _png(directory / "route_drone_0.png", (255, 0, 0))
    _png(directory / "route_drone_1.png", (0, 0, 255))
    (directory / LOG_FILE).write_text(
        "".join(json.dumps(line) + "\n" for line in LOG_LINES)
    )
    record = RunRecord(
        started_at=started_at,
        inputs=RunInputs(
            config_path="/repo/scenarios/small_indoor/config.yaml",
            config={},
            drones=2,
            assignment="global",
            target_tolerance_cells=3,
            variant=variant,
            view=False,
        ),
        outputs=RunOutputs(
            ticks=ticks,
            coverage=coverage,
            blocked=False,
            unreachable_frontiers=0,
            tick_capped=False,
            succeeded=True,
            wall_seconds=4.5,
            paths={
                "0": {"ticks": ticks, "route_length": 40, "distinct_cells": 35},
                "1": {"ticks": 60, "route_length": 20, "distinct_cells": 18},
            },
            event_counts=event_counts
            if event_counts is not None
            else {"drone_failed": 1, "frontier_assigned": 2},
            files=[LOG_FILE, "map.png", "route_drone_0.png", "route_drone_1.png"],
        ),
    )
    write_run_record(directory, record)
    return directory
