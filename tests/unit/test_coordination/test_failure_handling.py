"""Failure detection and recovery in CentralizedMaster (Sprint 3, Feature 10).

Real SimulationEngine on the inline 6 m room from test_master — never a MuJoCo
mock. Failures are injected with `engine.fail_drone`, never told to the master:
finding them from symptoms is the property under test.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarm_mapping.coordination.master import CentralizedMaster
from swarm_mapping.coordination.types import DroneHealth, DroneState
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.simulation.types import FailureMode
from tests.unit.test_coordination.test_master import ROOM_XML, build_master

pytestmark = pytest.mark.sprint(3)

MIN_SEPARATION_CELLS = 0.5 / 0.25  # test_master's MIN_SEPARATION / RESOLUTION


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """Write the inline room to a file the engine can load."""
    path = tmp_path / "room.xml"
    path.write_text(ROOM_XML)
    return path


def run(master: CentralizedMaster, ticks: int) -> None:
    """Tick the master a fixed number of times."""
    for _ in range(ticks):
        master.tick()


def run_to_end(master: CentralizedMaster, cap: int = 600) -> None:
    """Tick until the mission completes or the cap is hit."""
    while not master.is_complete and master.tick_count < cap:
        master.tick()


def known_cells(mapper: Mapper) -> int:
    """Count cells that are no longer unknown."""
    prob = mapper.grid.probability()
    return int(np.count_nonzero((prob < 0.4) | (prob > 0.6)))


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    """Captured log records whose message is `name`."""
    return [r for r in caplog.records if r.getMessage() == name]


class TestHealthDefaults:
    """Every drone starts ACTIVE, and detection cannot be disabled."""

    def test_every_drone_starts_active(self, scene: Path) -> None:
        """A fresh master diagnoses nobody."""
        master, _, _ = build_master(scene, {0: (0.0, 0.0), 1: (1.5, 0.0)})
        assert all(s.health is DroneHealth.ACTIVE for s in master.drone_states.values())

    def test_a_state_built_without_health_is_active(self) -> None:
        """The default is what keeps every existing DroneState(...) call valid."""
        state = DroneState(
            drone_id=0, cell=(0, 0), assignment=None, path_index=0, waited_ticks=0
        )
        assert state.health is DroneHealth.ACTIVE

    @pytest.mark.parametrize("key", ["heartbeat_timeout_ticks", "stuck_timeout_ticks"])
    def test_detection_cannot_be_switched_off(self, scene: Path, key: str) -> None:
        """A timeout of 0 is rejected at construction."""
        with pytest.raises(ValueError, match=key):
            build_master(scene, {0: (0.0, 0.0)}, **{key: 0})


class TestStuck:
    """Plan tests 4-6: commanded moves that do not happen."""

    def test_stuck_is_declared_after_the_timeout_th_unrealized_move(
        self, scene: Path
    ) -> None:
        """Ticks 0-2 each grant a move that never happens; tick 3 declares."""
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
        # Starts at exactly min_separation. Measured: 0.75 m apart produced no
        # wait at all; 0.5 m produces nine wait-ticks across the mission.
        master, _, _ = build_master(
            scene,
            {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (0.0, 0.5)},
            stuck_timeout_ticks=1,
        )
        saw_wait = False
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            saw_wait |= any(s.waited_ticks > 0 for s in master.drone_states.values())
            assert all(
                s.health is DroneHealth.ACTIVE for s in master.drone_states.values()
            )
        assert saw_wait, "no drone ever yielded — the test is not exercising waits"

    def test_a_stuck_drone_is_still_sensed_after_it_is_declared(
        self, scene: Path
    ) -> None:
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
    """Plan tests 7-9: the failed drone's frontier, tasking and body."""

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
        """No assignment and no homeward move, ever, once declared."""
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
        """The wreck stays in resolve_moves as a static body."""
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
