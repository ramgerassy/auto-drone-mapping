# Feature 9 — Failure injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a scenario script a drone failure at a given tick, applied by the simulator, without the coordinator ever seeing the schedule.

**Architecture:** `simulation` gains a `FailureMode` enum, `SimulationEngine.fail_drone` / `heartbeat`, and a `FailureInjector` that applies a schedule tick by tick. `config` gains an optional, dependency-free `failures:` section. `cli` converts one to the other and adds `Mission.tick()`, the single place the schedule meets the tick loop. `coordination` is untouched — detection is Feature 10.

**Tech Stack:** Python 3.13, MuJoCo 3.8, numpy, pytest, ruff, mypy (strict).

**Spec:** [`docs/sprint-3-plan.md`](../sprint-3-plan.md) — "How failure works" and Locked decisions 2–4, 6; decision D6.

## Global Constraints

- `from __future__ import annotations` at the top of every file; Google-style docstrings on every module and public function/class.
- `config` imports nothing from other domain modules. `simulation` never imports `config`.
- No MuJoCo mocks: simulation tests use a real `SimulationEngine` on `small_indoor.xml`.
- Determinism: schedules are sorted by `(tick, drone_id)`; no set iteration in a decision path.
- New test modules carry `pytestmark = pytest.mark.sprint(3)`.
- Logging is structured: `_LOGGER.info("event_name", extra={...})`, never `print()`.
- Run before every commit: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q -m "not acceptance"`.

## File map

| File | Change | Responsibility |
| --- | --- | --- |
| `src/swarm_mapping/simulation/types.py` | modify | `FailureMode` enum |
| `src/swarm_mapping/simulation/engine.py` | modify | `fail_drone`, `heartbeat`; failed drones ignore motion commands |
| `src/swarm_mapping/simulation/failure.py` | create | `ScheduledFailure`, `FailureInjector` |
| `src/swarm_mapping/config/schema.py` | modify | `FailureSettings`, optional `failures:` section, `ScenarioConfig.failures` |
| `src/swarm_mapping/cli.py` | modify | build the injector; `Mission.tick()`; `run_pipeline` uses it |
| `tests/unit/test_simulation/test_failure.py` | create | engine failure semantics |
| `tests/unit/test_simulation/test_failure_injector.py` | create | schedule application |
| `tests/unit/test_config/test_failures.py` | create | config parsing and rejection |
| `tests/unit/test_cli.py` | modify | wiring and the `--drones` fail-fast |

**Prerequisite:** Task 0 of the sprint plan (`CURRENT_SPRINT = 3`) must land first, or the `sprint(3)` markers below are ahead of the current sprint.

---

### Task 1: The engine can fail a drone

**Files:**
- Modify: `src/swarm_mapping/simulation/types.py`
- Modify: `src/swarm_mapping/simulation/engine.py` (`__init__`, `set_drone_position` at ~line 252, two new methods)
- Test: `tests/unit/test_simulation/test_failure.py`

**Interfaces:**
- Produces: `FailureMode.SILENT`, `FailureMode.STUCK` (values `"silent"`, `"stuck"`); `SimulationEngine.fail_drone(drone_id: int, mode: FailureMode) -> None`; `SimulationEngine.heartbeat(drone_id: int) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for drone failure in SimulationEngine.

A failed drone ignores motion commands in both modes; the modes differ only in
what the drone still *reports*. That difference is the whole reason the
coordinator needs two detectors in Feature 10, so it is what these pin down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.types import FailureMode

pytestmark = pytest.mark.sprint(3)

SCENE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "swarm_mapping" / "simulation" / "assets" / "small_indoor.xml"
)
START = {0: (0.0, 0.0, 1.0), 1: (2.0, 0.0, 1.0)}


@pytest.fixture
def engine() -> SimulationEngine:
    """Two healthy drones in the small indoor room."""
    return SimulationEngine(
        SCENE_PATH, {d: np.array(p, dtype=np.float64) for d, p in START.items()}
    )


def command(engine: SimulationEngine, drone_id: int, target: tuple[float, float, float]) -> bool:
    """Command a move and report whether the drone actually got there."""
    goal = np.array(target, dtype=np.float64)
    engine.set_drone_position(drone_id, goal)
    return bool(np.allclose(engine.get_pose(drone_id).position, goal))


class TestHealthyDrone:
    """The baseline both failure modes are measured against."""

    def test_reports_a_heartbeat(self, engine: SimulationEngine) -> None:
        assert engine.heartbeat(0)

    def test_moves_when_commanded(self, engine: SimulationEngine) -> None:
        assert command(engine, 0, (0.5, 0.0, 1.0))


class TestStuckDrone:
    """Motor fault: still talking, no longer moving."""

    def test_keeps_reporting(self, engine: SimulationEngine) -> None:
        engine.fail_drone(0, FailureMode.STUCK)
        assert engine.heartbeat(0)

    def test_ignores_motion_commands(self, engine: SimulationEngine) -> None:
        engine.fail_drone(0, FailureMode.STUCK)
        assert not command(engine, 0, (0.5, 0.0, 1.0))
        assert np.allclose(engine.get_pose(0).position, START[0])

    def test_its_sensor_still_works(self, engine: SimulationEngine) -> None:
        """A stuck drone still scans — its reports stay useful to the map."""
        engine.fail_drone(0, FailureMode.STUCK)
        hits = engine.cast_rays(0, np.array([[1.0, 0.0, 0.0]]))
        assert hits[0] is not None


class TestSilentDrone:
    """Crash or comms loss: nothing comes back."""

    def test_stops_reporting(self, engine: SimulationEngine) -> None:
        engine.fail_drone(0, FailureMode.SILENT)
        assert not engine.heartbeat(0)

    def test_ignores_motion_commands(self, engine: SimulationEngine) -> None:
        engine.fail_drone(0, FailureMode.SILENT)
        assert not command(engine, 0, (0.5, 0.0, 1.0))


class TestIsolation:
    def test_failing_one_drone_leaves_the_other_untouched(
        self, engine: SimulationEngine
    ) -> None:
        engine.fail_drone(0, FailureMode.SILENT)
        assert engine.heartbeat(1)
        assert command(engine, 1, (2.5, 0.0, 1.0))


class TestMisuse:
    def test_an_unknown_drone_is_rejected(self, engine: SimulationEngine) -> None:
        with pytest.raises(KeyError):
            engine.fail_drone(7, FailureMode.SILENT)
        with pytest.raises(KeyError):
            engine.heartbeat(7)

    def test_a_drone_fails_once(self, engine: SimulationEngine) -> None:
        """Failure is permanent (sprint plan, Locked decision 2)."""
        engine.fail_drone(0, FailureMode.STUCK)
        with pytest.raises(ValueError, match="fails once"):
            engine.fail_drone(0, FailureMode.SILENT)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_simulation/test_failure.py -v`
Expected: collection error — `ImportError: cannot import name 'FailureMode'`.

- [ ] **Step 3: Implement**

`simulation/types.py` — add `from enum import Enum` to the imports, then:

```python
class FailureMode(Enum):
    """How a drone fails.

    Both modes ignore motion commands. They differ in what the drone still
    reports, which is what the coordinator can observe — and so decides which
    detector catches it.

    Attributes:
        SILENT: Crash or comms loss. No heartbeat, no scan, no motion.
        STUCK: Motor fault. Heartbeat and sensor continue; commanded moves do
            not happen.
    """

    SILENT = "silent"
    STUCK = "stuck"
```

`simulation/engine.py` — import `FailureMode` alongside `Pose`/`RayHit`; at the end of `__init__`:

```python
        # Drones that have failed, and how. Read only by `heartbeat` and
        # `set_drone_position`: the simulator knows who failed, and it is the
        # coordinator's job to work that out from symptoms.
        self._failed: dict[int, FailureMode] = {}
```

New methods, after `get_body_id`:

```python
    def fail_drone(self, drone_id: int, mode: FailureMode) -> None:
        """Fail a drone permanently.

        Args:
            drone_id: Integer identifier for the drone.
            mode: How it fails — see `FailureMode`.

        Raises:
            KeyError: If drone_id is not recognized.
            ValueError: If the drone has already failed. Failure is permanent,
                so a second call is a scripting error, not a mode change.
        """
        if drone_id not in self._joint_qpos_adr:
            msg = f"Unknown drone_id: {drone_id}"
            raise KeyError(msg)
        if drone_id in self._failed:
            msg = (
                f"drone {drone_id} has already failed "
                f"({self._failed[drone_id].value}); a drone fails once"
            )
            raise ValueError(msg)
        self._failed[drone_id] = mode

    def heartbeat(self, drone_id: int) -> bool:
        """Whether the drone reported in this tick.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            False only for a silently failed drone. A stuck drone still
            reports — which is exactly why it needs a different detector.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        if drone_id not in self._joint_qpos_adr:
            msg = f"Unknown drone_id: {drone_id}"
            raise KeyError(msg)
        return self._failed.get(drone_id) is not FailureMode.SILENT
```

In `set_drone_position`, directly after the unknown-id check:

```python
        # A failed drone's motors do not respond, in either mode. The command
        # is dropped rather than rejected: the caller cannot know the drone
        # failed, and finding that out is the point.
        if drone_id in self._failed:
            return
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_simulation/ -q`
Expected: all pass, including the existing engine tests.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/simulation/types.py src/swarm_mapping/simulation/engine.py tests/unit/test_simulation/test_failure.py
git commit -m "feat(simulation): drones can fail — silent or stuck"
```

---

### Task 2: A scenario can script failures

**Files:**
- Modify: `src/swarm_mapping/config/schema.py` (new dataclass before `ScenarioConfig`; new field on `ScenarioConfig`; new parser; call in `parse_config`)
- Test: `tests/unit/test_config/test_failures.py`

**Interfaces:**
- Produces: `FailureSettings(drone_id: int, tick: int, mode: str)`; `ScenarioConfig.failures: tuple[FailureSettings, ...]`, default `()`, sorted by `(tick, drone_id)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the optional `failures:` config section.

Optional because every existing scenario has no failures and must load
unchanged. Validated as strictly as every other section: a malformed schedule
is a scenario that silently never fails, which would make a recovery test pass
for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings
from tests.unit.test_config.test_schema import load_broken, valid_config, write_config

pytestmark = pytest.mark.sprint(3)


def load_with(tmp_path: Path, *entries: dict[str, Any]) -> tuple[FailureSettings, ...]:
    """Load the valid three-drone config with these failure entries."""
    config = valid_config()
    config["failures"] = list(entries)
    return load_config(write_config(tmp_path, config)).failures


class TestFailuresLoad:
    def test_an_absent_section_means_no_failures(self, tmp_path: Path) -> None:
        assert load_config(write_config(tmp_path, valid_config())).failures == ()

    def test_a_valid_schedule_loads(self, tmp_path: Path) -> None:
        assert load_with(tmp_path, {"drone": 1, "tick": 400, "mode": "silent"}) == (
            FailureSettings(drone_id=1, tick=400, mode="silent"),
        )

    def test_the_schedule_is_ordered_by_tick_then_drone(self, tmp_path: Path) -> None:
        """YAML order must not decide application order."""
        schedule = load_with(
            tmp_path,
            {"drone": 0, "tick": 500, "mode": "stuck"},
            {"drone": 2, "tick": 100, "mode": "silent"},
            {"drone": 1, "tick": 100, "mode": "stuck"},
        )
        assert [(f.tick, f.drone_id) for f in schedule] == [(100, 1), (100, 2), (500, 0)]


def _set(value: Any) -> Any:
    return lambda c: c.__setitem__("failures", value)


class TestFailuresRejected:
    @pytest.mark.parametrize(
        ("mutate", "fragment"),
        [
            (_set({"drone": 0}), "'failures' must be a list"),
            (_set([3]), "failures[0]' must be a mapping"),
            (_set([{"drone": 0, "mode": "silent"}]), "failures[0].tick"),
            (_set([{"drone": 0, "tick": 10, "mode": "exploded"}]), "failures[0].mode"),
            (_set([{"drone": 3, "tick": 10, "mode": "silent"}]), "failures[0].drone"),
            (_set([{"drone": 0, "tick": -1, "mode": "silent"}]), "failures[0].tick"),
            (_set([{"drone": 0, "tick": True, "mode": "silent"}]), "failures[0].tick"),
            (
                _set([
                    {"drone": 0, "tick": 10, "mode": "silent"},
                    {"drone": 0, "tick": 20, "mode": "stuck"},
                ]),
                "fails once",
            ),
        ],
        ids=[
            "not-a-list", "entry-not-mapping", "missing-tick", "unknown-mode",
            "drone-out-of-range", "negative-tick", "bool-tick", "duplicate-drone",
        ],
    )
    def test_a_malformed_schedule_names_the_offending_key(
        self, tmp_path: Path, mutate: Any, fragment: str
    ) -> None:
        assert fragment in str(load_broken(tmp_path, mutate).value)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_config/test_failures.py -v`
Expected: `ImportError: cannot import name 'FailureSettings'`.

- [ ] **Step 3: Implement**

In `schema.py`, before `class ScenarioConfig`:

```python
# Kept as strings: `config` depends on nothing, so the mapping onto
# `simulation.FailureMode` happens in `cli`.
_FAILURE_MODES = ("silent", "stuck")


@dataclass(frozen=True)
class FailureSettings:
    """One scripted drone failure.

    Attributes:
        drone_id: Which drone fails — an index into `drones.start_positions`.
        tick: The tick it fails at, applied before that tick runs.
        mode: "silent" or "stuck". See `simulation.FailureMode`.
    """

    drone_id: int
    tick: int
    mode: str
```

On `ScenarioConfig`, add to the docstring's Attributes
`failures: Scripted failures, ordered by (tick, drone). Empty for a nominal scenario.`
and as the **last** field (it has a default, so it must come last):

```python
    failures: tuple[FailureSettings, ...] = ()
```

New parser, after `_check_positions_fit_grid`:

```python
def _parse_failures(
    raw: dict[str, Any], drone_count: int
) -> tuple[FailureSettings, ...]:
    """Parse the optional `failures:` section.

    Args:
        raw: The whole config document.
        drone_count: Drones in the scenario, bounding `drone`.

    Returns:
        The schedule sorted by (tick, drone_id), or () when the section is
        absent.

    Raises:
        ValueError: On any malformed entry; the message names
            `failures[i].<key>`.
    """
    if "failures" not in raw:
        return ()
    entries = raw["failures"]
    if not isinstance(entries, list):
        msg = f"'failures' must be a list, got {type(entries).__name__}"
        raise ValueError(msg)

    parsed: list[FailureSettings] = []
    scheduled: set[int] = set()  # membership only; never iterated
    for index, entry in enumerate(entries):
        name = f"failures[{index}]"
        if not isinstance(entry, dict):
            msg = f"'{name}' must be a mapping, got {type(entry).__name__}"
            raise ValueError(msg)
        drone_id = _non_negative_int(entry, name, "drone")
        if drone_id >= drone_count:
            msg = (
                f"'{name}.drone' is {drone_id}, but the scenario has "
                f"{drone_count} drone(s), ids 0-{drone_count - 1}"
            )
            raise ValueError(msg)
        if drone_id in scheduled:
            msg = f"'{name}.drone': drone {drone_id} is scheduled twice; a drone fails once"
            raise ValueError(msg)
        scheduled.add(drone_id)
        tick = _non_negative_int(entry, name, "tick")
        mode = _field(entry, name, "mode")
        if mode not in _FAILURE_MODES:
            msg = f"'{name}.mode' must be one of {', '.join(_FAILURE_MODES)}, got {mode!r}"
            raise ValueError(msg)
        parsed.append(FailureSettings(drone_id=drone_id, tick=tick, mode=mode))

    return tuple(sorted(parsed, key=lambda f: (f.tick, f.drone_id)))
```

In `parse_config`, after `_check_positions_fit_grid(...)`:

```python
    failures = _parse_failures(raw, len(start_positions))
```

and pass `failures=failures,` as the last argument to `ScenarioConfig(...)`.

`_non_negative_int` and `_field` already format errors as `'{section_name}.{field}' …`, so passing `name = "failures[0]"` produces exactly the fragments the tests match. No helper changes.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_config/ -q`
Expected: all pass; the existing four scenario configs still load (they have no `failures:`).

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/config/schema.py tests/unit/test_config/test_failures.py
git commit -m "feat(config): optional failures section for scripted drone failure"
```

---

### Task 3: The schedule reaches the simulator, and only the simulator

**Files:**
- Create: `src/swarm_mapping/simulation/failure.py`
- Modify: `src/swarm_mapping/cli.py` (`Mission` at ~line 110, `build_mission` at ~line 205, the loop in `run_pipeline` at ~line 386)
- Test: `tests/unit/test_simulation/test_failure_injector.py`, `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `FailureMode`, `SimulationEngine.fail_drone`, `SimulationEngine.heartbeat` (Task 1); `ScenarioConfig.failures` (Task 2).
- Produces: `ScheduledFailure(drone_id: int, tick: int, mode: FailureMode)`; `FailureInjector(engine, schedule=())` with `apply(tick: int) -> list[ScheduledFailure]`; `Mission.injector`; `Mission.tick() -> None`. Feature 10's tests drive missions through `Mission.tick()`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_simulation/test_failure_injector.py`:

```python
"""Tests for FailureInjector — applying a failure schedule tick by tick."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.failure import FailureInjector, ScheduledFailure
from swarm_mapping.simulation.types import FailureMode

pytestmark = pytest.mark.sprint(3)

SCENE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "swarm_mapping" / "simulation" / "assets" / "small_indoor.xml"
)


@pytest.fixture
def engine() -> SimulationEngine:
    return SimulationEngine(
        SCENE_PATH,
        {0: np.array([0.0, 0.0, 1.0]), 1: np.array([2.0, 0.0, 1.0])},
    )


def silent(drone_id: int, tick: int) -> ScheduledFailure:
    return ScheduledFailure(drone_id=drone_id, tick=tick, mode=FailureMode.SILENT)


class TestApply:
    def test_nothing_happens_before_the_scheduled_tick(self, engine: SimulationEngine) -> None:
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(99) == []
        assert engine.heartbeat(0)

    def test_the_drone_fails_on_its_tick(self, engine: SimulationEngine) -> None:
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(100) == [silent(0, 100)]
        assert not engine.heartbeat(0)

    def test_each_failure_is_applied_once(self, engine: SimulationEngine) -> None:
        injector = FailureInjector(engine, [silent(0, 100)])
        injector.apply(100)
        assert injector.apply(101) == []  # and no "fails once" ValueError

    def test_a_missed_tick_still_fires(self, engine: SimulationEngine) -> None:
        """A caller that skips ticks must not silently skip a failure."""
        injector = FailureInjector(engine, [silent(0, 100)])
        assert injector.apply(150) == [silent(0, 100)]

    def test_same_tick_failures_apply_in_drone_order(self, engine: SimulationEngine) -> None:
        injector = FailureInjector(engine, [silent(1, 50), silent(0, 50)])
        assert [f.drone_id for f in injector.apply(50)] == [0, 1]


class TestValidation:
    def test_a_drone_not_in_this_run_is_rejected(self, engine: SimulationEngine) -> None:
        """Decision D6: a --drones override must not drop a failure silently."""
        with pytest.raises(ValueError, match="drone 4"):
            FailureInjector(engine, [silent(4, 10)])
```

Append to `tests/unit/test_cli.py`. The module is tagged `sprint(2)`; the class-level `sprint(3)` wins because `tests/conftest.py` resolves markers with `get_closest_marker`. Add these imports to the module:

```python
from dataclasses import replace

from swarm_mapping.cli import build_mission          # extend the existing import
from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import FailureSettings, ScenarioConfig

SMALL_INDOOR = Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
```

```python
@pytest.mark.sprint(3)
class TestFailureWiring:
    """The schedule reaches the engine through Mission.tick(), not the master."""

    @staticmethod
    def _config(*failures: FailureSettings) -> ScenarioConfig:
        return replace(load_config(SMALL_INDOOR), failures=failures)

    def test_mission_tick_applies_a_failure_on_its_tick(self) -> None:
        mission = build_mission(self._config(FailureSettings(0, 2, "silent")), drones=1)
        mission.tick()  # tick 0
        mission.tick()  # tick 1
        assert mission.engine.heartbeat(0)
        mission.tick()  # tick 2 — applied before the master runs
        assert not mission.engine.heartbeat(0)

    def test_a_failure_for_a_dropped_drone_fails_fast(self) -> None:
        """Decision D6, end to end through --drones."""
        config = self._config(FailureSettings(1, 10, "stuck"))
        with pytest.raises(ValueError, match="drone 1"):
            build_mission(config, drones=1)
```

`small_indoor` defines three start positions, so `drones=1` genuinely drops drone 1.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_simulation/test_failure_injector.py tests/unit/test_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'swarm_mapping.simulation.failure'`.

- [ ] **Step 3: Implement**

`src/swarm_mapping/simulation/failure.py`:

```python
"""Scripted drone failures — the simulator's half of Sprint 3.

Injection lives here and detection lives in `coordination`. The coordinator
never sees this schedule: a master that is told which drone failed has detected
nothing. It has to infer failure from missed heartbeats and unrealized moves,
the same evidence a real ground station has.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.types import FailureMode

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledFailure:
    """One failure to inject.

    Attributes:
        drone_id: The drone that fails.
        tick: The tick it fails at, applied before that tick runs.
        mode: How it fails.
    """

    drone_id: int
    tick: int
    mode: FailureMode


class FailureInjector:
    """Applies a failure schedule to the engine, tick by tick.

    Args:
        engine: The simulation to fail drones in.
        schedule: Failures to apply, in any order.

    Raises:
        ValueError: If the schedule names a drone that is not in this run.
            Fail fast: under a `--drones` override that drops the drone, the
            alternative is a failure scenario that quietly runs with no
            failure — and a recovery check that passes for the wrong reason.
    """

    def __init__(
        self, engine: SimulationEngine, schedule: Sequence[ScheduledFailure] = ()
    ) -> None:
        present = set(engine.drone_ids)  # membership only
        for failure in schedule:
            if failure.drone_id not in present:
                msg = (
                    f"failure schedule names drone {failure.drone_id}, which is "
                    f"not in this run (drones {sorted(present)})"
                )
                raise ValueError(msg)
        self._engine = engine
        self._pending = sorted(schedule, key=lambda f: (f.tick, f.drone_id))

    def apply(self, tick: int) -> list[ScheduledFailure]:
        """Fail every drone due at or before `tick` that has not failed yet.

        `<=` rather than `==` so a caller that skips a tick cannot skip a
        failure; each one still fires exactly once.

        Args:
            tick: The tick about to run.

        Returns:
            The failures applied now, in (tick, drone_id) order.
        """
        due = [f for f in self._pending if f.tick <= tick]
        self._pending = [f for f in self._pending if f.tick > tick]
        for failure in due:
            self._engine.fail_drone(failure.drone_id, failure.mode)
            _LOGGER.info(
                "failure_injected",
                extra={
                    "drone_id": failure.drone_id,
                    "tick": tick,
                    "mode": failure.mode.value,
                },
            )
        return due
```

`cli.py`:

1. Imports: `from swarm_mapping.simulation.failure import FailureInjector, ScheduledFailure` and `from swarm_mapping.simulation.types import FailureMode`.
2. `Mission` — add to the docstring's Attributes `injector: Applies the scenario's failure schedule. Empty for a nominal run.`, add the field `injector: FailureInjector`, and the method:

```python
    def tick(self) -> None:
        """Advance one tick: inject any failure due now, then run the master.

        The one place the failure schedule meets the tick loop, so the CLI,
        tests and benchmarks cannot disagree about when a failure lands.
        Applied *before* the master's tick, so a drone scheduled for tick T is
        already down when tick T senses — and the master learns of it only
        through what that tick observes.
        """
        self.injector.apply(self.master.tick_count)
        self.master.tick()
```

3. `build_mission` — after constructing `engine`:

```python
    injector = FailureInjector(
        engine,
        [
            ScheduledFailure(
                drone_id=f.drone_id, tick=f.tick, mode=FailureMode(f.mode)
            )
            for f in config.failures
        ],
    )
```

and pass `injector=injector` to `Mission(...)`.

4. `run_pipeline` — in the `while` loop, replace `master.tick()` with `mission.tick()` (bind `mission` if the function currently only holds `master`).

Note for Feature 10 and for anyone writing a harness: code that calls `mission.master.tick()` directly bypasses injection. `benchmarks/strategy_matrix.py` and `benchmarks/oracle_ceiling.py` do; that is correct for them (no scenario they run has failures) and they are left alone.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q -m "not acceptance"`
Expected: everything passes — the new tests, and every existing e2e test unchanged, because a scenario with no `failures:` builds an empty injector and `Mission.tick()` is then exactly `master.tick()`.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_mapping/simulation/failure.py src/swarm_mapping/cli.py tests/unit/test_simulation/test_failure_injector.py tests/unit/test_cli.py
git commit -m "feat(simulation): FailureInjector applies the schedule through Mission.tick"
```

---

## Done when

- A scenario can declare `failures:` and the named drone fails on its tick, in the requested mode.
- No existing scenario or test changes behaviour.
- `coordination` has no new imports and no knowledge of the schedule.
- ruff, mypy and the non-acceptance suite are green.

What Feature 9 deliberately does **not** do: the master still trusts its own commands, so a failed drone in a Feature-9-only build is silently mis-tracked. Nothing exercises that until Feature 10, which is where detection belongs.
