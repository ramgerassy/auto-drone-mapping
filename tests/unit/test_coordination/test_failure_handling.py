"""Failure detection and recovery in CentralizedMaster (Sprint 3, Feature 10).

Real SimulationEngine on the inline 6 m room from test_master — never a MuJoCo
mock. Failures are injected with `engine.fail_drone`, never told to the master:
finding them from symptoms is the property under test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.coordination.master import CentralizedMaster
from swarm_mapping.coordination.types import DroneHealth, DroneState
from swarm_mapping.mapping.mapper import Mapper
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
