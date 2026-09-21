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

    def test_the_released_frontier_is_handed_out_on_the_declaring_tick(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Plan test 7: released in `_observe`, re-assigned by that tick's `_assign`.

        Drone 1 has right of way and claims first. It is stuck from tick 0 and
        declared on tick 3. On that same tick drone 0 is needy (its own target
        evaporated as the map grew) and the nearest frontier for it is exactly
        the one drone 1 held. The setup was found by searching start positions
        in this room, not constructed through private state.
        """
        caplog.set_level(logging.INFO)
        master, engine, _ = build_master(scene, {0: (2.0, -2.0), 1: (-2.0, 0.0)})
        engine.fail_drone(1, FailureMode.STUCK)
        claims_before: dict[int, Cell] = {}
        while not events(caplog, "drone_failed") and master.tick_count < 50:
            claims_before = {
                d: s.assignment.region.cell
                for d, s in master.drone_states.items()
                if s.assignment is not None
            }
            master.tick()
        (failed,) = events(caplog, "drone_failed")
        assert failed.drone_id == 1 and failed.health == "stuck"
        assert failed.tick == master.tick_count - 1  # the tick just run
        released = tuple(failed.released)

        # Going into the tick, the failed drone held it and nobody else did.
        assert claims_before[1] == released
        assert claims_before.get(0) != released
        # On that same tick: the failed drone holds nothing, and the only
        # holder of the released frontier is an ACTIVE teammate.
        assert master.drone_states[1].assignment is None
        holders = [
            d
            for d, s in master.drone_states.items()
            if s.assignment is not None and s.assignment.region.cell == released
        ]
        assert holders == [0]
        assert master.drone_states[0].health is DroneHealth.ACTIVE

    def test_a_failed_drone_is_never_tasked_or_sent_home(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No assignment, no motion, and never counted idle, once declared.

        A stuck drone cannot move, so "not sent home" is invisible in its
        cell; the `returning_to_base` event is what shows it. At
        `return_to_base_ticks=1` any idle ACTIVE drone away from home is sent
        home on its first idle tick. The non-vacuity checks prove that: the
        healthy teammate in this run is sent home, and in a healthy twin run
        drone 1 itself is too.

        Two mechanisms each keep a wreck from being sent home: the ACTIVE-only
        guard in `_update_idle_drones`, and the overlay (a home route planned
        from inside the wreck's own footprint does not exist). Measured by
        mutation: this test fails only with both removed.
        """
        caplog.set_level(logging.INFO)
        twin, _, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)}, return_to_base_ticks=1
        )
        twin_home = twin.drone_states[1].cell
        run_to_end(twin)
        twin_homeward = {r.drone_id for r in events(caplog, "returning_to_base")}
        assert 1 in twin_homeward, "a healthy drone 1 would have been sent home"
        caplog.clear()

        master, engine, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)}, return_to_base_ticks=1
        )
        # Fly a few ticks first so the wreck is away from home: a drone parked
        # on its home cell is never sent home, which would make this vacuous.
        run(master, 3)
        engine.fail_drone(1, FailureMode.STUCK)
        caplog.clear()
        run(master, 4)
        assert master.drone_states[1].health is DroneHealth.STUCK
        wreck = master.drone_states[1].cell
        assert wreck != twin_home
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            assert master.drone_states[1].assignment is None
            assert master.drone_states[1].cell == wreck
        # Only what happens from the declaration on counts, including the rest
        # of the declaring tick itself: that is when a missing guard would fire.
        messages = [r.getMessage() for r in caplog.records]
        after = caplog.records[messages.index("drone_failed") :]
        homeward = [r.drone_id for r in after if r.getMessage() == "returning_to_base"]
        assert 0 in homeward, "the idle teammate was not sent home: test is vacuous"
        assert 1 not in homeward

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


class TestLost:
    """Plan tests 1-3: silence."""

    def test_a_silent_drone_maps_nothing_from_the_tick_it_goes_silent(
        self, scene: Path
    ) -> None:
        """Before it is declared, not only after; the healthy twin is the control."""
        healthy, _, healthy_map = build_master(scene, {0: (0.0, 0.0)})
        healthy.tick()
        assert known_cells(healthy_map) > 0

        master, engine, mapper = build_master(scene, {0: (0.0, 0.0)})
        engine.fail_drone(0, FailureMode.SILENT)
        run(master, 2)  # still ACTIVE: the timeout is 3
        assert master.drone_states[0].health is DroneHealth.ACTIVE
        assert known_cells(mapper) == 0

    def test_lost_is_declared_on_the_timeout_th_missed_heartbeat(
        self, scene: Path
    ) -> None:
        """Not one tick early, not one late."""
        master, engine, _ = build_master(
            scene, {0: (-1.5, 0.0), 1: (1.5, 0.0)}, heartbeat_timeout_ticks=3
        )
        engine.fail_drone(1, FailureMode.SILENT)
        run(master, 2)
        assert master.drone_states[1].health is DroneHealth.ACTIVE
        master.tick()
        assert master.drone_states[1].health is DroneHealth.LOST

    def test_a_silent_drone_is_not_mistaken_for_stuck(self, scene: Path) -> None:
        """No telemetry is no motion evidence (F10-R3), even at a hair-trigger."""
        master, engine, _ = build_master(
            scene,
            {0: (-1.5, 0.0), 1: (1.5, 0.0)},
            heartbeat_timeout_ticks=3,
            stuck_timeout_ticks=1,
        )
        engine.fail_drone(1, FailureMode.SILENT)
        run(master, 3)
        assert master.drone_states[1].health is DroneHealth.LOST

    def test_a_healthy_swarm_is_never_declared_lost(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A full healthy mission declares nobody."""
        caplog.set_level(logging.INFO)
        master, _, _ = build_master(
            scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 1.5)}
        )
        run_to_end(master)
        assert all(s.health is DroneHealth.ACTIVE for s in master.drone_states.values())
        assert events(caplog, "drone_failed") == []


# The 6 m room split north-south by a wall at x = 0, with one gap for
# |y| < 1.0 m. A wreck in the gap leaves just enough room to pass beside it at
# Chebyshev distance 3 (footprint k = 1 plus clearance r = 1), so the only way
# east is the narrow side of the gap.
CORRIDOR_XML = """\
<mujoco model="coord_test_corridor">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <geom name="wall_east" type="box" pos="3 0 1" size="0.1 3 1"/>
    <geom name="wall_west" type="box" pos="-3 0 1" size="0.1 3 1"/>
    <geom name="wall_north" type="box" pos="0 3 1" size="3 0.1 1"/>
    <geom name="wall_south" type="box" pos="0 -3 1" size="3 0.1 1"/>
    <geom name="split_north" type="box" pos="0 2 1" size="0.1 1 1"/>
    <geom name="split_south" type="box" pos="0 -2 1" size="0.1 1 1"/>
  </worldbody>
</mujoco>
"""

# Measured on CORRIDOR_XML with the starts below: drone 0 changes target 4
# times after the declaration and finishes on tick 26. The bound is twice
# that. With the overlay disabled the same run never finishes: it hits the
# 600-tick cap pressed against the wreck.
MAX_TARGET_CHANGES = 8


class TestWreck:
    """Plan tests 11-12: plan around wrecks, never map them (D4)."""

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
        saw_assignment = False
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            assignment = master.drone_states[0].assignment
            if assignment is None:
                continue
            saw_assignment = True
            for col, row in assignment.path[master.drone_states[0].path_index :]:
                assert max(abs(col - wc), abs(row - wr)) > 2
        # Otherwise the loop above asserts nothing: a drone 0 that is never
        # assigned a path would pass this test vacuously.
        assert saw_assignment

    def test_a_wreck_in_the_corridor_is_passed_without_thrashing(
        self, tmp_path: Path
    ) -> None:
        """Plan test 11: drone 0 gets through the gap beside the wreck.

        Bounded target changes (see MAX_TARGET_CHANGES), no path within
        Chebyshev 2 of the wreck, the mission finishes well inside the cap,
        and drone 0 really crosses. Without the crossing the other checks
        could pass by drone 0 never trying.
        """
        corridor = tmp_path / "corridor.xml"
        corridor.write_text(CORRIDOR_XML)
        master, engine, _ = build_master(corridor, {0: (-2.0, 0.0), 1: (0.0, -0.75)})
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 5)
        assert master.drone_states[1].health is DroneHealth.STUCK
        wc, wr = master.drone_states[1].cell

        current = master.drone_states[0].assignment
        target = current.region.cell if current is not None else None
        changes = 0
        crossed = False
        while not master.is_complete and master.tick_count < 600:
            master.tick()
            state = master.drone_states[0]
            crossed |= state.cell[0] > wc + 1
            if state.assignment is None:
                continue
            if state.assignment.region.cell != target:
                changes += 1
                target = state.assignment.region.cell
            for col, row in state.assignment.path[state.path_index :]:
                assert max(abs(col - wc), abs(row - wr)) > 2

        assert master.is_complete, "hit the tick cap: drone 0 thrashed at the wreck"
        assert crossed, "drone 0 never got past the wreck: test is vacuous"
        assert changes <= MAX_TARGET_CHANGES

    def test_the_wreck_never_reaches_the_map(self, scene: Path) -> None:
        """The overlay is planning-only: the exported map must not show the wreck."""
        master, mapper, (wc, wr) = self._wrecked(scene)
        run_to_end(master)
        assert mapper.grid.probability()[wr, wc] <= 0.6


class TestSwarmLost:
    """Plan test 10: every drone failed."""

    def test_losing_every_drone_ends_the_mission(
        self, scene: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Terminates, reports blocked, logs swarm_lost exactly once."""
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
    """Plan test 14: diagnosis comes from symptoms alone."""

    def test_the_master_is_never_given_the_schedule(self) -> None:
        """No constructor parameter carries failure knowledge."""
        params = inspect.signature(CentralizedMaster.__init__).parameters
        assert not any("fail" in name or "schedule" in name for name in params)

    def test_diagnosis_matches_the_injected_truth(self, scene: Path) -> None:
        """SILENT is diagnosed LOST, STUCK is STUCK, and the healthy drone is ACTIVE."""
        master, engine, _ = build_master(
            scene, {0: (-1.5, -1.5), 1: (1.5, -1.5), 2: (0.0, 1.5)}
        )
        engine.fail_drone(0, FailureMode.SILENT)
        engine.fail_drone(1, FailureMode.STUCK)
        run(master, 6)
        health = {d: s.health for d, s in master.drone_states.items()}
        assert health == {
            0: DroneHealth.LOST,
            1: DroneHealth.STUCK,
            2: DroneHealth.ACTIVE,
        }
