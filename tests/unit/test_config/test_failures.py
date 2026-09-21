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
    """A well-formed `failures:` section loads into `ScenarioConfig.failures`."""

    def test_an_absent_section_means_no_failures(self, tmp_path: Path) -> None:
        """No `failures:` key means an empty schedule."""
        assert load_config(write_config(tmp_path, valid_config())).failures == ()

    def test_a_valid_schedule_loads(self, tmp_path: Path) -> None:
        """A single entry loads into a matching `FailureSettings`."""
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
        assert [(f.tick, f.drone_id) for f in schedule] == [
            (100, 1),
            (100, 2),
            (500, 0),
        ]


def _set(value: Any) -> Any:
    return lambda c: c.__setitem__("failures", value)


class TestFailuresRejected:
    """A malformed `failures:` entry is rejected, naming the offending key."""

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
                _set(
                    [
                        {"drone": 0, "tick": 10, "mode": "silent"},
                        {"drone": 0, "tick": 20, "mode": "stuck"},
                    ]
                ),
                "fails once",
            ),
        ],
        ids=[
            "not-a-list",
            "entry-not-mapping",
            "missing-tick",
            "unknown-mode",
            "drone-out-of-range",
            "negative-tick",
            "bool-tick",
            "duplicate-drone",
        ],
    )
    def test_a_malformed_schedule_names_the_offending_key(
        self, tmp_path: Path, mutate: Any, fragment: str
    ) -> None:
        """The error message names `failures[i].<key>`."""
        assert fragment in str(load_broken(tmp_path, mutate).value)
