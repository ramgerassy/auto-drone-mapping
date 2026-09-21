# Feature 10 — Failure detection and recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `CentralizedMaster` notices a failed drone from symptoms alone, declares it `LOST` or `STUCK`, releases its frontier before the same tick's assignment pass, never tasks it again, routes teammates around its wreck, and ends the mission cleanly if the whole swarm is lost.

**Architecture:** A new first phase, `_observe`, runs before sense → assign → move. It polls heartbeats and settles the motion evidence `_move` gathered last tick by reading each moved drone's pose back from the localizer, then declares failures, so a release always precedes that tick's assignment pass. Health lives on the frozen `DroneState` (D3), so `Coordinator.drone_states` carries it without a signature change. Wrecks enter planning through a per-tick **overlay** copy of the grid that is never written back to the map (D4).

**Tech Stack:** Python 3.13, MuJoCo 3.8, numpy, pytest, ruff, mypy (strict).

**Spec:** [`docs/sprint-3-plan.md`](../sprint-3-plan.md) — "How failure works", decisions D1/D3/D4, "Feature 10 — test cases" 1–14, Locked decisions 2, 3, 5, 6.

**Builds on:** Feature 9 (`SimulationEngine.fail_drone`, `SimulationEngine.heartbeat`, `FailureMode`). Branch `feat/failure-handling` **off `feat/failure-injection`**.

## Global Constraints

- `from __future__ import annotations`; Google docstrings on every module and public function/class; type hints; `uv run mypy` clean.
- No MuJoCo mocks. Master tests use the real `SimulationEngine` on the inline 6 m room from `tests/unit/test_coordination/test_master.py`. Failures are injected with `engine.fail_drone` directly, so the master has to *find* them.
- Determinism: iterate drones in `self._ordered_ids` (descending id) order; never iterate a set in a decision path.
- Dependency direction: `coordination` may import `simulation` (engine, `DRONE_HALF_EXTENT`) and `mapping`. It must **not** import `simulation.failure` (`FailureInjector` / `ScheduledFailure`): the schedule is not the master's business.
- New test modules: `pytestmark = pytest.mark.sprint(3)`.
- Before every commit: `uv run ruff check src/ tests/ benchmarks/ && uv run ruff format --check src/ tests/ benchmarks/ && uv run mypy && uv run pytest -q -m "not acceptance"`.

## Controller rulings (decided; do not re-decide)

- **F10-R1 — Timeouts.** `heartbeat_timeout_ticks` and `stuck_timeout_ticks` are **required** keys in `coordination:`, both `3` in all four shipped scenarios, and both must be ≥ 1: detection cannot be switched off. The `CentralizedMaster` keyword defaults are also `3`, so the existing tests that build a master directly keep working. At `large_indoor`'s 0.2 s/tick (D1), 3 ticks is 0.6 s, well inside the < 10-tick KPI.
- **F10-R2 — Observe first.** Each tick runs **observe → sense → assign → move**. Both detectors declare in `_observe`, so the frontier a failed drone held is back in the pool for that same tick's assignment pass (plan test 7). `_move` only *counts* unrealized moves; it never declares.
- **F10-R3 — No heartbeat, no action.** A drone that missed its heartbeat this tick is not sensed, not commanded, and gathers no motion evidence either way: without telemetry, the master has nothing to act on. It keeps its claim until declared `LOST`, and that hold *is* the latency the KPI measures. A drone declared `STUCK` still reports, so it keeps being **sensed** (plan test 6) but is never commanded again.
  - *Undeclared is still ACTIVE.* Within the ≤ `heartbeat_timeout_ticks` window before declaration, a silent drone is still `ACTIVE` in the assignment pass, so if its target evaporates it can be handed a **new** claim. That claim is released on declaration like any other.
- **F10-R4 — What counts as stuck.** The unrealized-move counter increments only when `resolve_moves` **granted** a move and the localizer says the drone did not arrive. Waiting to yield to a teammate never counts. Any realized move resets it to 0.
- **F10-R5 — Wreck footprint.** Chebyshev radius `k = ceil(DRONE_HALF_EXTENT / resolution + 0.5) - 1` around the wreck's cell. This is the same overlap rule `AStarPlanner` uses for inflation, and `k = 1` at every resolution the scenarios use. Those cells are set to log-odds `+2.0` in the overlay, which is used for `assign_all` and for the return-home plan, and for nothing else. Frontier detection, progress checks and the exported map all read the real grid.
  - *Exception (benign):* `assign_all` also derives its target-survival set, `frontier_cells(grid)`, from the grid it is given, so it sees the overlay. Frontier cells under a wreck's footprint are unreachable anyway, so dropping them from the survival check changes nothing that matters.
- **F10-R6 — Swarm lost.** When no drone is `ACTIVE`, the mission is complete, `blocked` if frontiers remain, and `swarm_lost` is logged exactly once.
- **F10-R7 — Zero-regression gate.** The test asserts exact tick counts and coverage on `small_indoor` at 1/2/3 drones against the baseline measured at `02a9a13`. Map SHA-256s are checked **locally only** and recorded in the report, not asserted in CI: an exact float hash can differ across CPUs, and a flaky gate is worse than none.
- **F10-R8 — `drone_failed` event.** `_LOGGER.warning("drone_failed", extra={"drone_id": int, "health": "lost"|"stuck", "tick": int, "released": [col, row] | None})`. Feature 11 computes latency as `drone_failed.tick − failure_injected.tick`.

Baseline at `02a9a13` (`small_indoor`, `build_mission(cfg, d)`, loop to `is_complete` or `max_ticks`):

| drones | ticks | coverage | map sha256[:16] |
| --- | --- | --- | --- |
| 1 | 498 | 0.9799 | `09f950b017aff423` |
| 2 | 254 | 0.9822 | `4c8826303df6733d` |
| 3 | 197 | 0.9831 | `ced633862f48d60e` |

## File map

| File | Change | Responsibility |
| --- | --- | --- |
| `src/swarm_mapping/coordination/types.py` | modify | `DroneHealth`; `DroneState.health` |
| `src/swarm_mapping/coordination/master.py` | modify | `_observe`, pose read-back, detectors, `_declare_failed`, overlay, swarm-lost |
| `src/swarm_mapping/coordination/protocols.py` | modify | `drone_states` docstring mentions health (no signature change) |
| `src/swarm_mapping/config/schema.py` | modify | two required `coordination` keys |
| `src/swarm_mapping/cli.py` | modify | pass the timeouts to the master |
| `scenarios/*/config.yaml` (4 files) | modify | `heartbeat_timeout_ticks: 3`, `stuck_timeout_ticks: 3` |
| `tests/unit/test_config/test_schema.py` | modify | `valid_config()` gains the two keys |
| `tests/unit/test_config/test_detection_settings.py` | create | parsing and rejection of the new keys |
| `tests/unit/test_coordination/test_master.py` | modify | `build_master` accepts the two timeouts |
| `tests/unit/test_coordination/test_failure_handling.py` | create | plan tests 1–12, 14 |
| `tests/integration/test_zero_regression.py` | create | plan test 13 |

Run `grep -rn "return_to_base_ticks" src tests scenarios` first: every place that builds a coordination section or `CoordinationSettings` needs the two new keys.

---

### Task 1: Health on the state, timeouts in the config (plumbing only — no behaviour)

**Files:** `coordination/types.py`, `coordination/protocols.py`, `config/schema.py`, `cli.py`, the four scenario YAMLs, `tests/unit/test_config/test_schema.py`, `tests/unit/test_config/test_detection_settings.py` (create), `tests/unit/test_coordination/test_master.py`, `master.py` (constructor only).

**Interfaces — Produces:**
- `DroneHealth.ACTIVE | LOST | STUCK` (values `"active"`, `"lost"`, `"stuck"`)
- `DroneState.health: DroneHealth = DroneHealth.ACTIVE` (last field)
- `CoordinationSettings.heartbeat_timeout_ticks: int`, `.stuck_timeout_ticks: int`
- `CentralizedMaster(..., heartbeat_timeout_ticks: int = 3, stuck_timeout_ticks: int = 3)`; `ValueError` if either is < 1
- `build_master(..., heartbeat_timeout_ticks: int = 3, stuck_timeout_ticks: int = 3)` in `test_master.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config/test_detection_settings.py`:

```python
"""Tests for the failure-detection keys in `coordination:`.

Required, like every other key: a scenario that silently ran with detection off
would pass a recovery test for the wrong reason. And ≥ 1, because 0 would mean
"never declare", which is detection switched off by another name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.config.loader import load_config
from tests.unit.test_config.test_schema import load_broken, valid_config, write_config

pytestmark = pytest.mark.sprint(3)

KEYS = ("heartbeat_timeout_ticks", "stuck_timeout_ticks")


def test_the_timeouts_load(tmp_path: Path) -> None:
    coordination = load_config(write_config(tmp_path, valid_config())).coordination
    assert (coordination.heartbeat_timeout_ticks, coordination.stuck_timeout_ticks) == (3, 3)


@pytest.mark.parametrize("key", KEYS)
def test_a_missing_timeout_is_rejected(tmp_path: Path, key: str) -> None:
    excinfo = load_broken(tmp_path, lambda c: c["coordination"].pop(key))
    assert f"coordination.{key}" in str(excinfo.value)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", [0, -1])
def test_a_timeout_below_one_is_rejected(tmp_path: Path, key: str, value: int) -> None:
    excinfo = load_broken(tmp_path, lambda c: c["coordination"].__setitem__(key, value))
    assert f"coordination.{key}" in str(excinfo.value)
```

Append to `tests/unit/test_coordination/test_failure_handling.py` (create the module with this header):

```python
"""Failure detection and recovery in CentralizedMaster (Sprint 3, Feature 10).

Real SimulationEngine on the inline 6 m room from test_master — never a MuJoCo
mock. Failures are injected with `engine.fail_drone`, never told to the master:
finding them from symptoms is the property under test.
"""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarm_mapping.coordination.master import CentralizedMaster
from swarm_mapping.coordination.types import Cell, DroneHealth, DroneState
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.simulation.types import FailureMode
from tests.unit.test_coordination.test_master import ROOM_XML, build_master

pytestmark = pytest.mark.sprint(3)

MIN_SEPARATION_CELLS = 0.5 / 0.25  # test_master's MIN_SEPARATION / RESOLUTION


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    path = tmp_path / "room.xml"
    path.write_text(ROOM_XML)
    return path


def run(master: CentralizedMaster, ticks: int) -> None:
    for _ in range(ticks):
        master.tick()


def run_to_end(master: CentralizedMaster, cap: int = 600) -> None:
    while not master.is_complete and master.tick_count < cap:
        master.tick()


def known_cells(mapper: Mapper) -> int:
    prob = mapper.grid.probability()
    return int(np.count_nonzero((prob < 0.4) | (prob > 0.6)))


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == name]


class TestHealthDefaults:
    def test_every_drone_starts_active(self, scene: Path) -> None:
        master, _, _ = build_master(scene, {0: (0.0, 0.0), 1: (1.5, 0.0)})
        assert all(s.health is DroneHealth.ACTIVE for s in master.drone_states.values())

    def test_a_state_built_without_health_is_active(self) -> None:
        """The default is what keeps every existing DroneState(...) call valid."""
        state = DroneState(drone_id=0, cell=(0, 0), assignment=None, path_index=0, waited_ticks=0)
        assert state.health is DroneHealth.ACTIVE

    @pytest.mark.parametrize("key", ["heartbeat_timeout_ticks", "stuck_timeout_ticks"])
    def test_detection_cannot_be_switched_off(self, scene: Path, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            build_master(scene, {0: (0.0, 0.0)}, **{key: 0})
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_config/test_detection_settings.py tests/unit/test_coordination/test_failure_handling.py -v`
Expected: `ImportError: cannot import name 'DroneHealth'`.

- [ ] **Step 3: Implement**

`coordination/types.py` — add `from enum import Enum`, then before `DroneState`:

```python
class DroneHealth(Enum):
    """The master's *diagnosis* of a drone — inferred, never told.

    Distinct from `simulation.FailureMode`, which is the simulator's ground
    truth. Tests compare the two; the master only ever sees symptoms.

    Attributes:
        ACTIVE: Reporting and moving as commanded.
        LOST: Missed `heartbeat_timeout_ticks` consecutive heartbeats.
        STUCK: Still reporting, but `stuck_timeout_ticks` consecutive granted
            moves did not happen.
    """

    ACTIVE = "active"
    LOST = "lost"
    STUCK = "stuck"
```

On `DroneState`, add to the docstring's Attributes `health: The master's diagnosis. Anything but ACTIVE is permanent: the drone is never tasked again.` and as the **last** field:

```python
    health: DroneHealth = DroneHealth.ACTIVE
```

`coordination/protocols.py` — `drone_states` docstring becomes `"""Read-only view of drone state, including each drone's health, for visualization."""`. No signature change.

`config/schema.py` — `CoordinationSettings`: document and append

```python
    heartbeat_timeout_ticks: int
    stuck_timeout_ticks: int
```

Docstring text for both (Attributes): *"Consecutive missed heartbeats before a drone is declared lost. At least 1: detection cannot be switched off."* and *"Consecutive granted-but-unrealized moves before a drone is declared stuck. Waiting to yield never counts. At least 1."* In `parse_config`'s `CoordinationSettings(...)`:

```python
            heartbeat_timeout_ticks=_positive_int(
                coordination_section, "coordination", "heartbeat_timeout_ticks"
            ),
            stuck_timeout_ticks=_positive_int(
                coordination_section, "coordination", "stuck_timeout_ticks"
            ),
```

Four scenario YAMLs, in `coordination:` after `return_to_base_ticks`:

```yaml
  heartbeat_timeout_ticks: 3  # missed heartbeats before a drone is declared lost
  stuck_timeout_ticks: 3      # granted moves that did not happen before "stuck"
```

`tests/unit/test_config/test_schema.py` `valid_config()` — add both keys, value `3`, to `"coordination"`.

`master.py` constructor — add keyword params after `target_tolerance_cells`, document them in the class docstring (Args and Raises), and validate:

```python
        heartbeat_timeout_ticks: int = 3,
        stuck_timeout_ticks: int = 3,
    ) -> None:
        ...
        for name, value in (
            ("heartbeat_timeout_ticks", heartbeat_timeout_ticks),
            ("stuck_timeout_ticks", stuck_timeout_ticks),
        ):
            if value < 1:
                msg = (
                    f"{name} must be at least 1, got {value}: failure "
                    "detection cannot be switched off"
                )
                raise ValueError(msg)
        self._heartbeat_timeout = heartbeat_timeout_ticks
        self._stuck_timeout = stuck_timeout_ticks
```

`cli.py` `build_mission` — pass `heartbeat_timeout_ticks=config.coordination.heartbeat_timeout_ticks, stuck_timeout_ticks=config.coordination.stuck_timeout_ticks`.

`test_master.py` `build_master` — add `heartbeat_timeout_ticks: int = 3, stuck_timeout_ticks: int = 3` params, passed through to `CentralizedMaster`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q -m "not acceptance"` — Expected: all green; no behaviour has changed.

- [ ] **Step 5: Commit** — `feat(coordination): DroneHealth on DroneState, detection timeouts in config`

---

### Task 2: Pose read-back, the stuck detector, and declaring a failure

**Files:** `master.py`; tests in `test_failure_handling.py`.

**Interfaces — Produces (private, used by Tasks 3–5):** `_observe()`, `_declare_failed(drone_id, health)`, `_read_cell(drone_id) -> Cell`, `self._heard: list[int]`, `self._unrealized: dict[int, int]`, `self._missed: dict[int, int]`.

- [ ] **Step 1: Write the failing tests** (append to `test_failure_handling.py`)

```python
class TestStuck:
    def test_stuck_is_declared_after_the_timeout_th_unrealized_move(self, scene: Path) -> None:
        """Ticks 0-2 each grant a move that never happens; tick 3 observes and declares."""
        master, engine, _ = build_master(scene, {0: (0.0, 0.0)}, stuck_timeout_ticks=3)
        engine.fail_drone(0, FailureMode.STUCK)
        run(master, 3)
        assert master.drone_states[0].health is DroneHealth.ACTIVE
        master.tick()
        assert master.drone_states[0].health is DroneHealth.STUCK

    def test_the_state_follows_the_localizer_not_the_command(self, scene: Path) -> None:
        """A stuck drone's state must not claim it moved (Locked decision 5)."""
        master, engine, _ = build_master(scene, {0: (0.0, 0.0)})
        start = master.drone_states[0].cell
        engine.fail_drone(0, FailureMode.STUCK)
        run(master, 2)
        assert master.drone_states[0].cell == start

    def test_a_yielding_drone_is_never_stuck(self, scene: Path) -> None:
        """Waiting on a teammate is not an unrealized move.

        At the most sensitive setting — one miscount declares a healthy drone
        stuck — over a whole mission. The `saw_wait` check keeps the test from
        passing vacuously: if these starts ever stop producing a wait, move
        them closer until they do; do not delete the check.
        """
        master, _, _ = build_master(
            scene, {0: (0.0, 0.0), 1: (0.75, 0.0), 2: (0.0, 0.75)}, stuck_timeout_ticks=1
        )
        saw_wait = False
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            saw_wait |= any(s.waited_ticks > 0 for s in master.drone_states.values())
            assert all(s.health is DroneHealth.ACTIVE for s in master.drone_states.values())
        assert saw_wait, "no drone ever yielded — the test is not exercising waits"

    def test_a_stuck_drone_is_still_sensed_after_it_is_declared(self, scene: Path) -> None:
        """Its sensor works, and its reports are still worth having (F10-R3).

        Counts calls through a proxy on the master's sensor — a wrapper around
        our own Rangefinder, not a MuJoCo mock.
        """
        master, engine, _ = build_master(scene, {0: (0.0, 0.0), 1: (1.5, 1.5)})
        scanned: list[int] = []
        real_scan = master._sensor.scan  # noqa: SLF001

        def counting_scan(drone_id: int) -> Any:
            scanned.append(drone_id)
            return real_scan(drone_id)

        master._sensor.scan = counting_scan  # type: ignore[method-assign]  # noqa: SLF001
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 6)
        assert master.drone_states[1].health is DroneHealth.STUCK
        scanned.clear()
        master.tick()
        assert 1 in scanned


class TestReclaim:
    def test_the_released_frontier_is_not_orphaned(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A claim held by a dead drone would be unassignable forever.

        Drone 2 has right of way, so it claims first. After it is declared,
        the frontier it held must either be picked up by a teammate or stop
        being a frontier because a teammate mapped it.
        """
        caplog.set_level(logging.INFO)
        master, engine, mapper = build_master(
            scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 1.5)}
        )
        engine.fail_drone(2, FailureMode.STUCK)
        while not events(caplog, "drone_failed") and master.tick_count < 50:
            master.tick()
        (failed,) = events(caplog, "drone_failed")
        assert failed.drone_id == 2 and failed.health == "stuck"
        released = tuple(failed.released)
        assert master.drone_states[2].assignment is None

        taken_over = False
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            taken_over |= any(
                s.assignment is not None and s.assignment.region.cell == released
                for s in master.drone_states.values()
            )
        still_frontier = any(r.cell == released for r in mapper.get_frontiers())
        assert taken_over or not still_frontier

    def test_a_failed_drone_is_never_tasked_or_sent_home(self, scene: Path) -> None:
        master, engine, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)}, return_to_base_ticks=5
        )
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 5)
        assert master.drone_states[1].health is DroneHealth.STUCK
        wreck = master.drone_states[1].cell
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            assert master.drone_states[1].assignment is None
            assert master.drone_states[1].cell == wreck

    def test_teammates_keep_their_distance_from_the_wreck(self, scene: Path) -> None:
        master, engine, _ = build_master(
            scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 0.0)}
        )
        engine.fail_drone(2, FailureMode.STUCK)
        wreck = master.drone_states[2].cell
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            for drone_id in (0, 1):
                col, row = master.drone_states[drone_id].cell
                gap = math.hypot(col - wreck[0], row - wreck[1])
                assert gap >= MIN_SEPARATION_CELLS
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_coordination/test_failure_handling.py -v`
Expected: `TestStuck` and `TestReclaim` fail (the drone is never declared; the state claims it moved).

- [ ] **Step 3: Implement** in `master.py`

Imports: `DroneHealth` from `coordination.types`. In `__init__`, after the timeouts:

```python
        # Failure evidence, per drone. Consecutive counts: one good heartbeat
        # or one realized move clears them.
        self._missed: dict[int, int] = {}
        self._unrealized: dict[int, int] = {}
        # Drones whose heartbeat arrived this tick, in descending id order.
        self._heard: list[int] = list(self._ordered_ids)
```

`tick()` becomes:

```python
    def tick(self) -> None:
        """Advance the mission one tick: observe, sense, assign, then move.

        Observing comes first so that a failure declared this tick releases its
        frontier before this tick's assignment pass hands frontiers out.
        """
        self._observe()
        self._sense()
        self._assign()
        self._move()
        self._tick_count += 1
        self._check_progress()
```

New methods:

```python
    def _observe(self) -> None:
        """Poll heartbeats, then declare any drone whose evidence crossed a timeout.

        The master learns of a failure only here, and only from what a ground
        station would see: silence, or commanded moves that did not happen. A
        LOST drone is no longer polled; a STUCK one still reports, and keeps
        being sensed.
        """
        self._heard = []
        for drone_id in self._ordered_ids:
            health = self._states[drone_id].health
            if health is DroneHealth.LOST:
                continue
            if self._engine.heartbeat(drone_id):
                self._heard.append(drone_id)
                self._missed[drone_id] = 0
            elif health is DroneHealth.ACTIVE:
                self._missed[drone_id] = self._missed.get(drone_id, 0) + 1

        for drone_id in self._ordered_ids:
            if self._states[drone_id].health is not DroneHealth.ACTIVE:
                continue
            if self._missed.get(drone_id, 0) >= self._heartbeat_timeout:
                self._declare_failed(drone_id, DroneHealth.LOST)
            elif self._unrealized.get(drone_id, 0) >= self._stuck_timeout:
                self._declare_failed(drone_id, DroneHealth.STUCK)

    def _declare_failed(self, drone_id: int, health: DroneHealth) -> None:
        """Mark a drone failed for good and put its frontier back in the pool."""
        state = self._states[drone_id]
        released = state.assignment.region.cell if state.assignment else None
        self._states[drone_id] = replace(
            state, health=health, assignment=None, path_index=0, waited_ticks=0
        )
        self._going_home.pop(drone_id, None)
        self._idle_ticks.pop(drone_id, None)
        _LOGGER.warning(
            "drone_failed",
            extra={
                "drone_id": drone_id,
                "health": health.value,
                "tick": self._tick_count,
                "released": list(released) if released is not None else None,
            },
        )

    def _read_cell(self, drone_id: int) -> Cell:
        """Where the localizer says the drone is, snapped to the grid."""
        position = self._engine.get_pose(drone_id).position
        return self._mapper.grid.world_to_grid(float(position[0]), float(position[1]))
```

`_sense` scans `self._heard` instead of `self._ordered_ids` (same descending order).

`_assign` — pass only ACTIVE drones to `assign_all`, and merge the result back **preserving key order**:

```python
        active = {d: s for d, s in self._states.items() if s.health is DroneHealth.ACTIVE}
        assigned = assign_all(self._planning_grid(), frontiers, active, ...)  # rest unchanged
        merged = dict(self._states)
        merged.update(assigned)
        self._states = merged
```

(Task 4 adds `_planning_grid`; until then use `self._mapper.grid`.) The `_exhausted` loop at the top of `_assign` is unchanged; failed drones hold no assignment, so they never reach it.

`_update_idle_drones` — first line inside the loop: `if state.health is not DroneHealth.ACTIVE: continue`.

`_move` — only ACTIVE drones that were heard this tick get a desired cell; everyone else holds station and stays in `current`, so `resolve_moves` keeps teammates clear of them:

```python
        heard = set(self._heard)  # membership only
        desired: dict[int, Cell | None] = {
            drone_id: (
                next_cell(state)
                if state.health is DroneHealth.ACTIVE and drone_id in heard
                else None
            )
            for drone_id, state in self._states.items()
        }
        for drone_id, route in self._going_home.items():
            if route and drone_id in heard:
                desired[drone_id] = route[0]
```

and in the per-drone loop, after `self._teleport(drone_id, cell)`:

```python
                actual = self._read_cell(drone_id)
                if actual != cell:
                    # Granted and commanded, but the drone did not arrive. The
                    # state follows the localizer, the path does not advance,
                    # and `_observe` weighs the count next tick.
                    self._unrealized[drone_id] = self._unrealized.get(drone_id, 0) + 1
                    self._states[drone_id] = replace(state, cell=actual)
                    continue
                self._unrealized[drone_id] = 0
```

followed by the existing successful-move update.

For a healthy drone `actual == cell` always: `grid_to_world` returns the cell centre, and `world_to_grid` floors `col + 0.5` back to `col`. That is what makes Task 5's zero-regression test hold.

- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/unit/test_coordination/ -q`, then the full suite.
- [ ] **Step 5: Commit** — `feat(coordination): read poses back, detect stuck drones, release their frontiers`

---

### Task 3: The heartbeat detector

The mechanism landed in Task 2's `_observe`; this task proves it with tests of its own and fixes anything they expose.

- [ ] **Step 1: Write the tests**

```python
class TestLost:
    def test_a_silent_drone_maps_nothing_from_the_tick_it_goes_silent(self, scene: Path) -> None:
        """Before it is declared, not only after — the non-vacuity check is the healthy twin."""
        healthy, _, healthy_map = build_master(scene, {0: (0.0, 0.0)})
        healthy.tick()
        assert known_cells(healthy_map) > 0

        master, engine, mapper = build_master(scene, {0: (0.0, 0.0)})
        engine.fail_drone(0, FailureMode.SILENT)
        run(master, 2)  # still ACTIVE: the timeout is 3
        assert master.drone_states[0].health is DroneHealth.ACTIVE
        assert known_cells(mapper) == 0

    def test_lost_is_declared_on_the_timeout_th_missed_heartbeat(self, scene: Path) -> None:
        master, engine, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)}, heartbeat_timeout_ticks=3
        )
        engine.fail_drone(1, FailureMode.SILENT)
        run(master, 2)
        assert master.drone_states[1].health is DroneHealth.ACTIVE
        master.tick()
        assert master.drone_states[1].health is DroneHealth.LOST

    def test_a_silent_drone_is_not_mistaken_for_stuck(self, scene: Path) -> None:
        """No telemetry is no motion evidence (F10-R3), even at a hair-trigger stuck timeout."""
        master, engine, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)},
            heartbeat_timeout_ticks=3, stuck_timeout_ticks=1,
        )
        engine.fail_drone(1, FailureMode.SILENT)
        run(master, 3)
        assert master.drone_states[1].health is DroneHealth.LOST

    def test_a_healthy_swarm_is_never_declared_lost(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        master, _, _ = build_master(scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 1.5)})
        run_to_end(master)
        assert all(s.health is DroneHealth.ACTIVE for s in master.drone_states.values())
        assert events(caplog, "drone_failed") == []
```

- [ ] **Step 2–4:** run; fix anything they expose; full suite green.
- [ ] **Step 5: Commit** — `test(coordination): the heartbeat detector, including silent-is-not-stuck`

---

### Task 4: Wreck-aware planning overlay (D4)

- [ ] **Step 1: Write the failing tests**

```python
class TestWreck:
    @staticmethod
    def _wrecked(scene: Path) -> tuple[CentralizedMaster, Mapper, Cell]:
        """Drone 1 dies in the middle of the room; drone 0 has to work around it."""
        master, engine, mapper = build_master(scene, {0: (-2.0, 0.0), 1: (0.0, 0.0)})
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 5)
        assert master.drone_states[1].health is DroneHealth.STUCK
        return master, mapper, master.drone_states[1].cell

    def test_paths_route_around_the_wreck(self, scene: Path) -> None:
        """Footprint k=1 plus clearance r=1: nothing within Chebyshev 2 of the wreck."""
        master, _, (wc, wr) = self._wrecked(scene)
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            assignment = master.drone_states[0].assignment
            if assignment is None:
                continue
            for col, row in assignment.path[master.drone_states[0].path_index :]:
                assert max(abs(col - wc), abs(row - wr)) > 2

    def test_the_wreck_never_reaches_the_map(self, scene: Path) -> None:
        """The overlay is planning-only: the exported map must not show the wreck."""
        master, mapper, (wc, wr) = self._wrecked(scene)
        run_to_end(master)
        assert mapper.grid.probability()[wr, wc] <= 0.6
```

Note: the path check skips the part already flown (`path_index:`). A path committed *before* the declaration may pass near the wreck, and the master re-validates it against the overlay on the next assignment pass. If this test shows a stale path surviving one tick, that is the expected re-validation lag. Assert from the first tick *after* declaration plus one, and record that in the report. Do not weaken the distance.

- [ ] **Step 2: Run to verify they fail** — the path assertion fails, because paths run straight through the wreck.
- [ ] **Step 3: Implement** in `master.py`

```python
from swarm_mapping.mapping.grid import OccupancyGrid

# Log-odds written under a wreck in the planning overlay: well past the 0.6
# band, so the planner treats it exactly like a confirmed obstacle.
_WRECK_LOG_ODDS = 2.0
```

In `__init__` (resolution is known there):

```python
        # Cells a wreck's body overlaps — the same overlap rule AStarPlanner
        # uses for inflation. k = 1 at every resolution the scenarios use.
        resolution = grid.config.resolution
        self._wreck_radius = math.ceil(DRONE_HALF_EXTENT / resolution + 0.5) - 1
```

```python
    def _planning_grid(self) -> OccupancyGrid:
        """The grid to plan on: the map, plus every wreck as an obstacle.

        A failed drone is a physical body the map deliberately does not
        contain — the teammate filter keeps drones out of it, and the map's
        accuracy KPI depends on that. So wrecks go into a copy used for
        planning only; nothing written here reaches the exported map.
        """
        grid = self._mapper.grid
        wrecks = [
            self._states[d].cell
            for d in self._ordered_ids
            if self._states[d].health is not DroneHealth.ACTIVE
        ]
        if not wrecks:
            return grid
        overlay = OccupancyGrid(grid.config)
        overlay.log_odds[:] = grid.log_odds
        k = self._wreck_radius
        for col, row in wrecks:
            overlay.log_odds[
                max(0, row - k) : row + k + 1, max(0, col - k) : col + k + 1
            ] = _WRECK_LOG_ODDS
        return overlay
```

Use it: `_assign` passes `self._planning_grid()` to `assign_all` (compute it once per tick into a local); `_update_idle_drones` plans home on `self._planning_grid()`. Frontier detection, `_check_progress` and export keep reading `self._mapper.grid`.

- [ ] **Step 4–5:** tests green → commit `feat(coordination): plan around wrecks without writing them into the map`

---

### Task 5: The whole swarm lost, symptoms-only, and zero regression

- [ ] **Step 1: Write the failing tests**

Append to `test_failure_handling.py`:

```python
class TestSwarmLost:
    def test_losing_every_drone_ends_the_mission(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        master, engine, _ = build_master(scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)})
        run(master, 2)  # map something first, so frontiers remain
        engine.fail_drone(0, FailureMode.SILENT)
        engine.fail_drone(1, FailureMode.SILENT)
        run(master, 3)
        assert master.is_complete
        assert master.is_blocked  # frontiers remained
        run(master, 2)  # ticking a finished mission must not log it again
        assert len(events(caplog, "swarm_lost")) == 1


class TestSymptomsOnly:
    def test_the_master_is_never_given_the_schedule(self) -> None:
        params = inspect.signature(CentralizedMaster.__init__).parameters
        assert not any("fail" in name or "schedule" in name for name in params)

    def test_diagnosis_matches_the_injected_truth(self, scene: Path) -> None:
        master, engine, _ = build_master(
            scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 1.5)}
        )
        engine.fail_drone(0, FailureMode.SILENT)
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 6)
        health = {d: s.health for d, s in master.drone_states.items()}
        assert health == {0: DroneHealth.LOST, 1: DroneHealth.STUCK, 2: DroneHealth.ACTIVE}
```

`tests/integration/test_zero_regression.py`:

```python
"""Zero-regression gate for Sprint 3 (docs/sprint-3-plan.md).

Reading poses back from the localizer must change nothing for a healthy
swarm. Baseline measured at 02a9a13, before Feature 10. Exact integers on
purpose: the decisions are discrete, and a one-tick drift is a behaviour
change. Map hashes are checked locally, not here (F10-R7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm_mapping.cli import build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config

pytestmark = pytest.mark.sprint(3)

SMALL_INDOOR = Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
BASELINE = {1: (498, 0.9799), 2: (254, 0.9822), 3: (197, 0.9831)}


@pytest.mark.parametrize("drones", sorted(BASELINE))
def test_a_healthy_run_is_unchanged(drones: int) -> None:
    config = load_config(SMALL_INDOOR)
    mission = build_mission(config, drones)
    while not mission.master.is_complete and mission.master.tick_count < config.coordination.max_ticks:
        mission.tick()
    ticks, coverage = BASELINE[drones]
    assert mission.master.tick_count == ticks
    assert round(coverage_fraction(mission.mapper.grid), 4) == coverage
```

- [ ] **Step 2: Run to verify** — `TestSwarmLost` fails (no `swarm_lost` event). The regression test should already pass; if it does not, **stop**. A healthy run changing is exactly what this gate exists to catch. Report NEEDS_CONTEXT with the tick counts rather than editing the baseline.
- [ ] **Step 3: Implement** — in `master.py`, `self._swarm_lost_logged = False` in `__init__`. At the end of `_assign`:

```python
        if not active and not self._swarm_lost_logged:
            self._swarm_lost_logged = True
            _LOGGER.warning(
                "swarm_lost",
                extra={"tick": self._tick_count, "unreachable_frontiers": len(frontiers)},
            )
```

(`_complete` and `_blocked` already come out right: every state holds `assignment=None`.)

- [ ] **Step 4: Verify** — full suite green. Then **locally**, record in the report the map SHA-256 at 1/2/3 drones next to the baseline table above, computed as `hashlib.sha256(mission.mapper.grid.log_odds.tobytes()).hexdigest()[:16]`.
- [ ] **Step 5: Commit** — `feat(coordination): end the mission when the whole swarm is lost`

---

## Done when

- Plan tests 1–14 each map to a passing test (list the mapping in the report).
- `small_indoor` at 1/2/3 drones: identical ticks and coverage, and identical map hashes locally.
- `coordination` imports nothing from `simulation.failure`.
- ruff, mypy and the non-acceptance suite are green.
