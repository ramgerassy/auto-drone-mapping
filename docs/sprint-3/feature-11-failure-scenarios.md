# Feature 11 — Failure scenarios and the latency KPI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the failure-injection scenarios CLAUDE.md names, give ticks a duration (D1) so the "< 2 s reassignment latency" KPI means something, guard recovery with a fast integration test on every push, and measure the KPI on the large map.

**Architecture:** `drones.cruise_speed` joins the config and `ScenarioConfig.tick_seconds` derives from it. Two scenario directories reuse `large_indoor.xml` and add a `failures:` block. A fast integration test drives `small_indoor` through `Mission.tick()`. A benchmark drives the large scenarios the same way, captures the `failure_injected` and `drone_failed` events, and reports the latency in ticks and seconds.

**Tech Stack:** Python 3.13, MuJoCo 3.8, numpy, pytest.

**Spec:** [`docs/sprint-3-plan.md`](../sprint-3-plan.md) — D1, "KPIs this sprint closes", Feature 11 row.

**Builds on:** Feature 9 (`failures:`, `Mission.tick()`) and Feature 10 (`DroneHealth`, the `drone_failed` event with `tick`, and the `heartbeat_timeout_ticks`/`stuck_timeout_ticks` keys). Branch `feat/failure-scenarios` **off `feat/failure-handling`**.

## Global Constraints

- `from __future__ import annotations`; Google docstrings; type hints; `uv run mypy` clean.
- No MuJoCo mocks. The integration test uses `small_indoor` and must stay under ~20 s total.
- Latency is **ticks between `failure_injected.tick` and `drone_failed.tick`** (F10-R8), converted with `tick_seconds`. It is never measured with wall-clock time: a deterministic run has no seconds of its own.
- Determinism: same config ⇒ identical ticks and identical event ticks. The integration test asserts it.
- New test modules: `pytestmark = pytest.mark.sprint(3)`.
- Before every commit: `uv run ruff check src/ tests/ benchmarks/ && uv run ruff format --check src/ tests/ benchmarks/ && uv run mypy && uv run pytest -q -m "not acceptance"`.

## Controller rulings (decided; do not re-decide)

- **F11-R1 — `cruise_speed` is required**, in `drones:`, positive, **1.0 m/s in every shipped scenario**. `ScenarioConfig.tick_seconds` is a property, `map.resolution / drones.cruise_speed`, and is never stored, so the two can't disagree. At `large_indoor`'s 0.2 m cells that is 0.2 s/tick, so the KPI is **< 10 ticks**. At `small_indoor`'s 0.1 m it is 0.1 s/tick, so < 20 ticks.
- **F11-R2 — Scenarios.** `scenarios/failure_injection/config.yaml` is a copy of `large_indoor`'s config — same scene, same five spawns, same parameters — plus `failures: [{drone: 1, tick: 300, mode: silent}]`. `scenarios/failure_stuck/config.yaml` is identical except `mode: stuck`. Tick 300 is mid-mission: 5 healthy drones reach 95% at tick 517 and finish at 894. Scene: `path: large_indoor.xml`, not a copy of the MJCF. Each file's header comment says what it tests and why tick 300.
- **F11-R3 — A stuck drone that is never commanded can't be detected, and that's correct.** A motor fault on a drone with nowhere to go harms nothing. The benchmark reports latency only for a failure followed by a `drone_failed`, and reports "undetected (never commanded)" otherwise, without counting it as a KPI miss. In both shipped scenarios drone 1 holds a target at tick 300, so both must be detected.
- **F11-R4 — Recovery KPI thresholds.** Tier 1 still holds with a drone lost: coverage ≥ 95%, and the mission ends on its own terminal state rather than `max_ticks`. The mission may take longer than the healthy run. The benchmark reports by how much.

## File map

| File | Change | Responsibility |
| --- | --- | --- |
| `src/swarm_mapping/config/schema.py` | modify | `DroneSettings.cruise_speed`; `ScenarioConfig.tick_seconds` |
| `scenarios/{small_indoor,large_indoor,comb_indoor,loop_indoor}/config.yaml` | modify | `cruise_speed: 1.0` |
| `scenarios/failure_injection/config.yaml` | create | silent failure, drone 1, tick 300 |
| `scenarios/failure_stuck/config.yaml` | create | stuck failure, drone 1, tick 300 |
| `tests/unit/test_config/test_schema.py` | modify | `valid_config()` gains `cruise_speed` |
| `tests/unit/test_config/test_tick_seconds.py` | create | parsing, rejection, derivation |
| `tests/unit/test_simulation/test_failure_scenarios.py` | create | both scenario files load and fail the drone the header says |
| `tests/integration/test_failure_recovery.py` | create | fast recovery guard on `small_indoor` |
| `benchmarks/failure_recovery.py` | create | the KPI measurement on the large map |
| `benchmarks/failure_recovery.json` | create | its results |

---

### Task 1: A tick has a duration (D1)

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_config/test_tick_seconds.py`:

```python
"""Tests for `drones.cruise_speed` and the tick duration derived from it (D1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.config.loader import load_config
from tests.unit.test_config.test_schema import load_broken, valid_config, write_config

pytestmark = pytest.mark.sprint(3)

SCENARIOS = Path(__file__).resolve().parents[3] / "scenarios"


def test_tick_seconds_is_one_cell_at_cruise_speed(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, valid_config()))  # 0.2 m cells, 1.0 m/s
    assert config.tick_seconds == pytest.approx(0.2)


@pytest.mark.parametrize("value", [0, -1.0])
def test_a_non_positive_cruise_speed_is_rejected(tmp_path: Path, value: float) -> None:
    excinfo = load_broken(tmp_path, lambda c: c["drones"].__setitem__("cruise_speed", value))
    assert "drones.cruise_speed" in str(excinfo.value)


def test_a_missing_cruise_speed_is_rejected(tmp_path: Path) -> None:
    excinfo = load_broken(tmp_path, lambda c: c["drones"].pop("cruise_speed"))
    assert "drones.cruise_speed" in str(excinfo.value)


@pytest.mark.parametrize(
    ("scenario", "seconds"), [("small_indoor", 0.1), ("large_indoor", 0.2)]
)
def test_shipped_tick_durations(scenario: str, seconds: float) -> None:
    """The numbers the KPI conversion in the sprint plan relies on."""
    config = load_config(SCENARIOS / scenario / "config.yaml")
    assert config.tick_seconds == pytest.approx(seconds)
```

- [ ] **Step 2: Run** — fails with `AttributeError: ... 'tick_seconds'`.
- [ ] **Step 3: Implement** — `DroneSettings` gains `cruise_speed: float` (docstring: *"Cruise speed in m/s. One tick moves a drone one cell, so this is what gives a tick a duration: `tick_seconds = map.resolution / cruise_speed`. The simulation itself has no clock — `mj_step` is never called — so without this, time-based KPIs have no unit."*), parsed with `_positive(drones_section, "drones", "cruise_speed")`. `ScenarioConfig` gains:

```python
    @property
    def tick_seconds(self) -> float:
        """Nominal duration of one tick, in seconds.

        Derived rather than stored, so it can never disagree with the grid
        resolution or the cruise speed it comes from.
        """
        return self.map.resolution / self.drones.cruise_speed
```

Add `cruise_speed: 1.0  # m/s — one cell per tick; see docs/sprint-3-plan.md D1` under `drones:` in all four shipped scenario YAMLs, and `"cruise_speed": 1.0` to `valid_config()["drones"]`.

- [ ] **Step 4–5:** green → commit `feat(config): cruise_speed gives a tick a duration`

---

### Task 2: The failure scenarios

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_simulation/test_failure_scenarios.py`:

```python
"""The shipped failure scenarios load, and fail the drone they say they fail."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings

pytestmark = pytest.mark.sprint(3)

SCENARIOS = Path(__file__).resolve().parents[3] / "scenarios"


@pytest.mark.parametrize(("name", "mode"), [("failure_injection", "silent"), ("failure_stuck", "stuck")])
def test_the_scenario_schedules_one_mid_mission_failure(name: str, mode: str) -> None:
    config = load_config(SCENARIOS / name / "config.yaml")
    assert config.failures == (FailureSettings(drone_id=1, tick=300, mode=mode),)


@pytest.mark.parametrize("name", ["failure_injection", "failure_stuck"])
def test_the_scenario_is_large_indoor_plus_a_failure(name: str) -> None:
    """CLAUDE.md: 'same as large indoor, with one drone scripted to fail'."""
    large = load_config(SCENARIOS / "large_indoor" / "config.yaml")
    failing = load_config(SCENARIOS / name / "config.yaml")
    assert failing.scene_path == large.scene_path
    assert failing.drones == large.drones
    assert failing.map == large.map
    assert failing.planning == large.planning
    assert failing.coordination == large.coordination
```

- [ ] **Step 2–4:** create the two YAMLs per F11-R2; green.
- [ ] **Step 5: Commit** — `feat(scenarios): failure_injection and failure_stuck on the large indoor map`

---

### Task 3: Recovery is guarded on every push

- [ ] **Step 1: Write the test** — `tests/integration/test_failure_recovery.py`:

```python
"""Recovery from a mid-mission failure, end to end, fast enough for every push.

`small_indoor` with three drones finishes in 197 healthy ticks, so a failure
at tick 40 is mid-mission. Driven through `Mission.tick()`, the one place the
schedule meets the loop: the master finds the failure from symptoms.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import pytest

from swarm_mapping.cli import Mission, build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings
from swarm_mapping.coordination.types import DroneHealth

pytestmark = pytest.mark.sprint(3)

SMALL_INDOOR = Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
FAIL_AT = 40
EXPECTED = {"silent": DroneHealth.LOST, "stuck": DroneHealth.STUCK}


def fly(mode: str) -> tuple[Mission, list[logging.LogRecord]]:
    """Run small_indoor, 3 drones, drone 2 failing at FAIL_AT; return the mission and its events."""
    config = replace(
        load_config(SMALL_INDOOR),
        failures=(FailureSettings(drone_id=2, tick=FAIL_AT, mode=mode),),
    )
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    try:
        mission = build_mission(config, drones=3)
        separation = config.coordination.min_separation / config.map.resolution
        while not mission.master.is_complete and mission.master.tick_count < config.coordination.max_ticks:
            mission.tick()
            wreck = mission.master.drone_states[2]
            if wreck.health is not DroneHealth.ACTIVE:
                for drone_id in (0, 1):
                    col, row = mission.master.drone_states[drone_id].cell
                    assert math.hypot(col - wreck.cell[0], row - wreck.cell[1]) >= separation
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
    return mission, records


def event_tick(records: list[logging.LogRecord], name: str) -> int:
    (record,) = [r for r in records if r.getMessage() == name]
    return int(record.tick)


@pytest.mark.parametrize("mode", ["silent", "stuck"])
def test_the_swarm_detects_the_failure_and_still_maps_the_room(mode: str) -> None:
    mission, records = fly(mode)
    config = mission.config

    assert mission.master.drone_states[2].health is EXPECTED[mode]
    latency = event_tick(records, "drone_failed") - event_tick(records, "failure_injected")
    assert latency * config.tick_seconds < 2.0, f"{latency} ticks"

    assert mission.master.tick_count < config.coordination.max_ticks
    assert coverage_fraction(mission.mapper.grid) >= 0.95


def test_a_failure_run_is_deterministic() -> None:
    first, first_events = fly("silent")
    second, second_events = fly("silent")
    assert first.master.tick_count == second.master.tick_count
    assert event_tick(first_events, "drone_failed") == event_tick(second_events, "drone_failed")
```

If `failure_injected` carries its tick under a different key than `tick` (check Feature 9's `FailureInjector.apply`), read that key. Both events must be on the same tick index — the master's `tick_count` at the time.

- [ ] **Step 2: Run** — should pass on Features 9+10. If coverage misses 0.95 or latency misses the KPI, **stop and report NEEDS_CONTEXT with the numbers**: that is a finding about Feature 10, not a threshold to loosen.
- [ ] **Step 3: Commit** — `test(integration): recovery from silent and stuck failures on every push`

---

### Task 4: Measure the KPI on the large map

- [ ] **Step 1: Write `benchmarks/failure_recovery.py`** — same shape as `benchmarks/oracle_ceiling.py`: module docstring stating the question, metrics and why each is there; a `run(scenario) -> dict` that builds the mission, attaches a list-collecting logging handler (as in Task 3), ticks with **`mission.tick()`** (never `master.tick()` — that bypasses injection), and records:
  - `ticks`, `t95` (first tick with coverage ≥ 0.95), `coverage`, `blocked`, `tick_capped`
  - `injected_tick`, `declared_tick`, `health`, `latency_ticks`, `latency_seconds` (× `config.tick_seconds`), `kpi_met` (`latency_seconds < 2.0`), or `"undetected (never commanded)"` per F11-R3
  - `wall_seconds`

  Scenarios: `large_indoor` (the healthy reference), `failure_injection`, `failure_stuck`. Print a table and write `benchmarks/failure_recovery.json`. Usage line in the docstring: `uv run python benchmarks/failure_recovery.py`.

- [ ] **Step 2: Run it** (a few minutes; run in the background) and paste the printed table into the report.
- [ ] **Step 3: Commit** — `feat(benchmarks): measure failure-recovery latency on the large map`, including the JSON. Put the measured latency and the coverage/ticks delta against the healthy run in the commit message.

---

## Done when

- `failure_injection` and `failure_stuck` load, validate, and run from the CLI: `uv run swarm-mapping --config scenarios/failure_injection/config.yaml`.
- The integration test passes both modes, deterministically, in under ~20 s.
- The benchmark's table shows latency < 2 s for both modes and coverage ≥ 95% with a drone lost.
