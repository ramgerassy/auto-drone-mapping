"""Measures the "<2 s reassignment latency" KPI on the large map (Feature 11).

CLAUDE.md's Tier-2 KPI is "frontier reassignment latency <2s after drone
failure". Before D1 (`drones.cruise_speed` / `ScenarioConfig.tick_seconds`),
a tick had no duration, so that KPI had no unit — this benchmark is what
turns it into a number, on the map Sprint 2's coordination KPI already runs
on: `large_indoor`.

    uv run python benchmarks/failure_recovery.py

Three missions, each `large_indoor`'s five spawns, driven through
`Mission.tick()` — never `master.tick()`, which would bypass the scenario's
`failures:` schedule entirely:

- **large_indoor** — the healthy reference. No failure; establishes the
  baseline ticks and coverage a run with a drone lost is measured against.
- **failure_injection** — drone 1 fails silently at tick 300.
- **failure_stuck** — drone 1's motors jam at tick 300, while it keeps
  reporting.

Metrics, and why each is here:

- **ticks / t95 / coverage / blocked / tick_capped** — the same mission-shape
  metrics `oracle_ceiling.py` reports. `t95` is the tick coverage first
  crosses 95%, the Tier-1 indoor target; `tick_capped` distinguishes a mission
  that reached its own terminal state from one that hit `coordination.max_ticks`
  (F11-R4: the mission may run longer with a drone down, but it must still end
  on its own).
- **injected_tick / declared_tick / health / latency_ticks / latency_seconds /
  kpi_met** — the KPI itself. Latency is ticks between `failure_injected.tick`
  and `drone_failed.tick` (F10-R8), converted to seconds by `config.tick_seconds`
  — never wall-clock, because a deterministic run has no seconds of its own.
  When the schedule fires but no `drone_failed` ever follows, F11-R3 applies:
  the drone was never commanded (nowhere to go), so nothing was there to
  detect, and that is correct behaviour, not a KPI miss. `kpi_met` is then
  omitted in favour of that string.
- **delta_ticks / delta_t95 / delta_ticks_pct** — how much longer a failure
  row's mission ran than the `large_indoor` healthy row (F11-R4: "the
  benchmark reports by how much"). None on the healthy row itself, and None
  for `delta_t95` if either side's `t95` never landed within `max_ticks`.
- **wall_seconds** — how long the benchmark itself took to run this mission.
  Tier-3, measured but not committed; not to be confused with `latency_seconds`.

Results are written to `benchmarks/failure_recovery.json`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from swarm_mapping.cli import Mission, build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config

ROOT = Path(__file__).resolve().parent.parent

# Tier-1 indoor coverage target, from CLAUDE.md's KPI table.
_TARGET_COV = 0.95

# Tier-2 KPI: frontier reassignment latency after a drone failure.
_KPI_SECONDS = 2.0

SCENARIOS = ("large_indoor", "failure_injection", "failure_stuck")

_UNDETECTED = "undetected (never commanded)"


def _event_tick(
    records: list[logging.LogRecord], name: str, drone_id: int
) -> int | None:
    """Return the `tick` of the one record `name` logs for `drone_id`, if any."""
    for record in records:
        if record.getMessage() == name and getattr(record, "drone_id", None) == (
            drone_id
        ):
            return int(record.tick)  # type: ignore[attr-defined]
    return None


def _deltas(row: dict[str, Any], healthy: dict[str, Any] | None) -> dict[str, Any]:
    """How much longer this mission ran than the healthy reference (F11-R4).

    Args:
        row: This mission's own row so far (`ticks`, `t95`).
        healthy: The `large_indoor` reference row, or None for that row itself.

    Returns:
        `delta_ticks`, `delta_t95`, `delta_ticks_pct` — all None when `healthy`
        is None (this call *is* computing the healthy row) or when `t95`
        never landed on one side of the comparison.
    """
    if healthy is None:
        return {"delta_ticks": None, "delta_t95": None, "delta_ticks_pct": None}

    delta_ticks = row["ticks"] - healthy["ticks"]
    delta_t95 = (
        row["t95"] - healthy["t95"]
        if row["t95"] is not None and healthy["t95"] is not None
        else None
    )
    delta_ticks_pct = (
        round(100.0 * delta_ticks / healthy["ticks"], 1) if healthy["ticks"] else None
    )
    return {
        "delta_ticks": delta_ticks,
        "delta_t95": delta_t95,
        "delta_ticks_pct": delta_ticks_pct,
    }


def run(scenario: str, healthy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fly one mission through `Mission.tick()` and collect its metrics.

    Args:
        scenario: Scenario directory name under `scenarios/`.
        healthy: The `large_indoor` healthy reference row, already computed,
            or None while computing that row itself. When given, the failure
            row also reports `delta_ticks`, `delta_t95`, and `delta_ticks_pct`
            against it (F11-R4: "the benchmark reports by how much").

    Returns:
        One row of the comparison table.
    """
    config = load_config(ROOT / "scenarios" / scenario / "config.yaml")

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    mission: Mission
    started = time.time()
    try:
        mission = build_mission(config)
        master, mapper = mission.master, mission.mapper
        t95: int | None = None
        max_ticks = config.coordination.max_ticks
        while not master.is_complete and master.tick_count < max_ticks:
            mission.tick()
            if t95 is None and coverage_fraction(mapper.grid) >= _TARGET_COV:
                t95 = master.tick_count
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
    wall_seconds = time.time() - started

    row: dict[str, Any] = {
        "scenario": scenario,
        "ticks": master.tick_count,
        "t95": t95,
        "coverage": round(coverage_fraction(mapper.grid), 4),
        "blocked": master.is_blocked,
        "tick_capped": not master.is_complete,
        "wall_seconds": round(wall_seconds, 1),
    }
    row.update(_deltas(row, healthy))

    if not config.failures:
        # The healthy reference: nothing was injected, so there is nothing to
        # detect and no latency to report.
        row.update(
            injected_tick=None,
            declared_tick=None,
            health=None,
            latency_ticks=None,
            latency_seconds=None,
            kpi_met=None,
        )
        return row

    failure = config.failures[0]
    injected_tick = _event_tick(records, "failure_injected", failure.drone_id)
    declared_tick = _event_tick(records, "drone_failed", failure.drone_id)

    if declared_tick is None:
        # F11-R3: a stuck drone that is never commanded has nothing to detect
        # from — that is correct, not a KPI miss.
        row.update(
            injected_tick=injected_tick,
            declared_tick=None,
            health=None,
            latency_ticks=None,
            latency_seconds=None,
            kpi_met=_UNDETECTED,
        )
        return row

    if injected_tick is None:
        # A failure was scheduled and the master went on to declare a drone
        # failed, but the simulator never logged `failure_injected` for it.
        # That is not a legitimate outcome to report a latency for — it means
        # the log is missing an event the harness relies on, so the harness
        # itself is broken. Silently falling back to `declared_tick` would
        # report `latency_ticks == 0` and `kpi_met: True` for a run that
        # never actually measured detection latency.
        msg = (
            f"{scenario!r}: drone {failure.drone_id} was declared failed at "
            f"tick {declared_tick} but no 'failure_injected' event was logged "
            "for it — the benchmark harness is broken, not the mission"
        )
        raise RuntimeError(msg)
    latency_ticks = declared_tick - injected_tick
    latency_seconds = latency_ticks * config.tick_seconds
    row.update(
        injected_tick=injected_tick,
        declared_tick=declared_tick,
        health=master.drone_states[failure.drone_id].health.value,
        latency_ticks=latency_ticks,
        latency_seconds=round(latency_seconds, 3),
        kpi_met=latency_seconds < _KPI_SECONDS,
    )
    return row


def main() -> None:
    """Run the three missions and write `benchmarks/failure_recovery.json`.

    `large_indoor` runs first so its row is the `healthy` reference the two
    failure rows report their `delta_*` fields against (F11-R4).
    """
    rows: list[dict[str, Any]] = []
    header = (
        f"{'scenario':<18}{'ticks':>7}{'t@95%':>7}{'cov':>8}{'cap':>5}"
        f"{'health':>8}{'lat(t)':>8}{'lat(s)':>8}{'kpi':>10}"
        f"{'d.ticks':>9}{'d.t95':>7}{'d.tick%':>9}{'wall':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    healthy: dict[str, Any] | None = None
    for scenario in SCENARIOS:
        row = run(scenario, healthy=healthy)
        if scenario == "large_indoor":
            healthy = row
        rows.append(row)
        t95 = "—" if row["t95"] is None else row["t95"]
        health = row["health"] or "—"
        lat_t = "—" if row["latency_ticks"] is None else row["latency_ticks"]
        lat_s = (
            "—" if row["latency_seconds"] is None else f"{row['latency_seconds']:.2f}"
        )
        kpi = row["kpi_met"]
        kpi_str = "—" if kpi is None else (kpi if isinstance(kpi, str) else str(kpi))
        d_ticks = "—" if row["delta_ticks"] is None else f"+{row['delta_ticks']}"
        d_t95 = "—" if row["delta_t95"] is None else f"+{row['delta_t95']}"
        d_pct = (
            "—" if row["delta_ticks_pct"] is None else f"+{row['delta_ticks_pct']:.1f}%"
        )
        print(
            f"{row['scenario']:<18}{row['ticks']:>7}{t95:>7}{row['coverage']:>8.2%}"
            f"{'Y' if row['tick_capped'] else 'N':>5}{health:>8}{lat_t:>8}"
            f"{lat_s:>8}{kpi_str:>10}{d_ticks:>9}{d_t95:>7}{d_pct:>9}"
            f"{row['wall_seconds']:>6.0f}s",
            flush=True,
        )

    out = ROOT / "benchmarks" / "failure_recovery.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
