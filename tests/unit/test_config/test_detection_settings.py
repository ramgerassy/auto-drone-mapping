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
    """Both timeouts parse from a valid config."""
    coordination = load_config(write_config(tmp_path, valid_config())).coordination
    assert (coordination.heartbeat_timeout_ticks, coordination.stuck_timeout_ticks) == (
        3,
        3,
    )


@pytest.mark.parametrize("key", KEYS)
def test_a_missing_timeout_is_rejected(tmp_path: Path, key: str) -> None:
    """Each timeout is required; the error names the key."""
    excinfo = load_broken(tmp_path, lambda c: c["coordination"].pop(key))
    assert f"coordination.{key}" in str(excinfo.value)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", [0, -1])
def test_a_timeout_below_one_is_rejected(tmp_path: Path, key: str, value: int) -> None:
    """Zero or negative would switch detection off; the error names the key."""
    excinfo = load_broken(tmp_path, lambda c: c["coordination"].__setitem__(key, value))
    assert f"coordination.{key}" in str(excinfo.value)
