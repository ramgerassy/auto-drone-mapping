"""Benchmark the allocation variants across every scenario.

Not a test: full missions over several maps take minutes, and the output is a
comparison table rather than a pass/fail. Run it by hand when changing how
frontiers are handed out.

    uv run python benchmarks/strategy_matrix.py            # all scenarios
    uv run python benchmarks/strategy_matrix.py large_indoor

Multi-drone only. The single-drone case already improved measurably under the
existing strategy, and it cannot show the thing these variants are about, which
is how work divides *between* drones.

Metrics, and why each is here:

- **ticks** — mission length. The headline efficiency number.
- **coverage** — the Tier-1 KPI. A variant that finishes sooner by exploring
  less has not improved anything, so this guards the others.
- **revisited** — cells a drone entered more than once, summed over drones.
  The direct measure of wasted travel, and the number that exposed the problem.
- **worst cell** — the most times any single cell was re-entered *while the
  drone held a target*. Catches a drone shuttling in one spot. Counting idle
  ticks too would measure the return-to-base feature instead: a drone parked at
  base for 400 ticks scores 400 and looks like pathological thrashing.
- **idle** — share of drone-ticks spent with no target at all. A variant that
  finishes by parking two of three drones has not divided the work, it has
  shed it.
- **swaps** — target changes per drone. Distinguishes "explores efficiently"
  from "keeps changing its mind".
- **spread** — fraction of visited cells reached by exactly one drone. Low
  means drones covered the same ground; it is the division-of-labour measure.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from swarm_mapping.cli import build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config

ROOT = Path(__file__).resolve().parent.parent

# (label, assignment mode, target tolerance in cells)
VARIANTS = (
    ("baseline", "greedy", 0),
    ("B: tolerance", "greedy", 3),
    ("A: global", "global", 0),
    ("A+B", "global", 3),
)


def run(scenario: str, mode: str, tolerance: int) -> dict[str, Any]:
    """Fly one mission and collect its metrics.

    Args:
        scenario: Scenario directory name under `scenarios/`.
        mode: `coordination.assignment` value.
        tolerance: `coordination.target_tolerance_cells` value.

    Returns:
        One row of the comparison table.
    """
    base = load_config(ROOT / "scenarios" / scenario / "config.yaml")
    config = replace(
        base,
        coordination=replace(
            base.coordination, assignment=mode, target_tolerance_cells=tolerance
        ),
    )
    mission = build_mission(config)
    master, mapper = mission.master, mission.mapper

    shape = (config.map.grid_height, config.map.grid_width)
    visits = {d: np.zeros(shape, dtype=np.int64) for d in master.drone_states}
    # Separate tally: cells entered while actually working. `visits` still
    # counts every tick, because spread and coverage care where drones went.
    working = {d: np.zeros(shape, dtype=np.int64) for d in master.drone_states}
    idle = dict.fromkeys(master.drone_states, 0)
    previous = dict.fromkeys(master.drone_states)
    swaps = dict.fromkeys(master.drone_states, 0)

    started = time.time()
    while not master.is_complete and master.tick_count < config.coordination.max_ticks:
        master.tick()
        for drone_id, state in master.drone_states.items():
            col, row = state.cell
            visits[drone_id][row, col] += 1
            if state.assignment is None:
                idle[drone_id] += 1
            else:
                working[drone_id][row, col] += 1
            goal = state.assignment.region.cell if state.assignment else None
            if goal != previous[drone_id]:
                swaps[drone_id] += 1
                previous[drone_id] = goal

    stacked = np.stack(list(visits.values()))
    touched_by = (stacked > 0).sum(axis=0)
    visited = int(np.count_nonzero(touched_by))
    exclusive = int(np.count_nonzero(touched_by == 1))

    return {
        "scenario": scenario,
        "variant": mode,
        "tolerance": tolerance,
        "drones": len(visits),
        "ticks": master.tick_count,
        "capped": master.tick_count >= config.coordination.max_ticks,
        "coverage": round(coverage_fraction(mapper.grid), 4),
        "revisited": int(sum(int(np.count_nonzero(v > 1)) for v in visits.values())),
        "worst_cell": int(max(int(v.max()) for v in working.values())),
        "idle": round(sum(idle.values()) / (len(idle) * max(1, master.tick_count)), 3),
        "swaps": round(sum(swaps.values()) / len(swaps), 1),
        "spread": round(exclusive / visited, 3) if visited else 0.0,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    """Run the matrix and write `benchmarks/results.json`."""
    requested = sys.argv[1:]
    scenarios = requested or [
        path.name
        for path in sorted((ROOT / "scenarios").iterdir())
        if (path / "config.yaml").exists()
    ]

    rows: list[dict[str, Any]] = []
    header = (
        f"{'scenario':14s} {'variant':14s} {'ticks':>6s} {'cov':>7s} "
        f"{'revis':>6s} {'worst':>6s} {'idle':>6s} {'swaps':>6s} "
        f"{'spread':>7s} {'sec':>6s}"
    )
    print(header)
    print("-" * len(header))
    for scenario in scenarios:
        for label, mode, tolerance in VARIANTS:
            row = run(scenario, mode, tolerance)
            row["label"] = label
            rows.append(row)
            print(
                f"{scenario:14s} {label:14s} {row['ticks']:6d} {row['coverage']:6.1%} "
                f"{row['revisited']:6d} {row['worst_cell']:6d} {row['idle']:5.0%} "
                f"{row['swaps']:6.1f} "
                f"{row['spread']:6.1%} {row['seconds']:6.1f}"
                + ("  CAPPED" if row["capped"] else ""),
                flush=True,
            )

    out = ROOT / "benchmarks" / "results.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
