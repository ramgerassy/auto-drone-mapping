"""Unit tests for scenario config loading and validation.

`config` depends on nothing, so everything here is pure dict-and-dataclass
work — no MuJoCo, no grid, no planner. The validation boundary is the point:
these tests pin down exactly which mistakes are caught at startup and which
are deliberately left to the modules that can see the geometry.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from swarm_mapping.config.loader import load_config

pytestmark = pytest.mark.sprint(2)  # tests introduced in Sprint 2


def valid_config() -> dict[str, Any]:
    """Return a minimal, valid three-drone scenario config as a plain dict.

    Every test starts from this and breaks exactly one thing, so a failure
    names the field that broke rather than a whole invalid document.
    """
    return {
        "scene": {"path": "large_indoor.xml"},
        "drones": {
            "start_positions": [
                [0.0, 0.0, 1.0],
                [0.0, -1.2, 1.0],
                [0.0, 1.2, 1.0],
            ],
            "altitude": 1.0,
        },
        "sensor": {
            "num_rays": 72,
            "max_range": 12.0,
            "elevation_layers": 5,
            "elevation_max_deg": 20.0,
            "exclusion_radius": 0.30,
        },
        "map": {
            "resolution": 0.2,
            "origin_x": -25.0,
            "origin_y": -25.0,
            "grid_width": 250,
            "grid_height": 250,
            "max_height": 3.0,
        },
        "planning": {
            "clearance_radius": 0.20,
            "spread_radius": 0.0,
            "spread_penalty": 0,
            "min_frontier_size": 2,
        },
        "coordination": {
            "min_separation": 0.5,
            "max_wait_ticks": 5,
            "max_ticks": 2000,
            "assignment": "greedy",
            "target_tolerance_cells": 0,
            "no_progress_ticks": 200,
            "return_to_base_ticks": 150,
            "heartbeat_timeout_ticks": 3,
            "stuck_timeout_ticks": 3,
        },
    }


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write a config dict to a YAML file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def load_broken(tmp_path: Path, mutate: Any) -> pytest.ExceptionInfo[ValueError]:
    """Apply `mutate` to a valid config, load it, and capture the ValueError.

    Args:
        tmp_path: pytest temp directory for the YAML file.
        mutate: Callable applied to a fresh copy of the valid config.

    Returns:
        The captured exception info, so callers can assert on the message.
    """
    config = copy.deepcopy(valid_config())
    mutate(config)
    with pytest.raises(ValueError) as excinfo:
        load_config(write_config(tmp_path, config))
    return excinfo


class TestValidConfig:
    """A well-formed config loads into the expected typed values."""

    @pytest.mark.sanity
    def test_loads_every_section(self, tmp_path: Path) -> None:
        """All six sections land on the returned ScenarioConfig."""
        config = load_config(write_config(tmp_path, valid_config()))

        assert config.scene_path == "large_indoor.xml"
        assert config.drones.altitude == 1.0
        assert config.sensor.num_rays == 72
        assert config.sensor.max_range == 12.0
        assert config.sensor.exclusion_radius == 0.30
        assert config.map.resolution == 0.2
        assert config.map.grid_width == 250
        assert config.map.max_height == 3.0
        assert config.planning.clearance_radius == 0.20
        assert config.planning.spread_radius == 0.0
        assert config.planning.spread_penalty == 0
        assert config.planning.min_frontier_size == 2
        assert config.coordination.min_separation == 0.5
        assert config.coordination.max_wait_ticks == 5
        assert config.coordination.max_ticks == 2000

    def test_start_positions_are_tuples_of_floats(self, tmp_path: Path) -> None:
        """YAML lists become immutable tuples of floats, not lists of ints."""
        config = load_config(write_config(tmp_path, valid_config()))

        assert config.drones.start_positions == (
            (0.0, 0.0, 1.0),
            (0.0, -1.2, 1.0),
            (0.0, 1.2, 1.0),
        )

    def test_drone_count_comes_from_start_positions(self, tmp_path: Path) -> None:
        """There is no separate count key that could disagree."""
        config = load_config(write_config(tmp_path, valid_config()))

        assert config.drones.count == 3

    def test_config_is_frozen(self, tmp_path: Path) -> None:
        """Config is read-only once validated; nothing mutates it mid-run.

        Binding `drones` first is deliberate: asserting `FrozenInstanceError`
        on `config.drones.altitude = ...` directly would also pass against a
        plain dict, which raises AttributeError for the *attribute lookup*.
        FrozenInstanceError subclasses AttributeError, so the looser assertion
        cannot tell "frozen dataclass" from "no dataclass at all".
        """
        config = load_config(write_config(tmp_path, valid_config()))
        drones = config.drones

        with pytest.raises(dataclasses.FrozenInstanceError):
            drones.altitude = 2.0  # type: ignore[misc]


class TestMissingKeys:
    """A missing key fails at startup, naming the key."""

    @pytest.mark.parametrize("section", ["scene", "drones", "sensor", "map"])
    def test_missing_section_names_the_section(
        self, tmp_path: Path, section: str
    ) -> None:
        """Dropping a whole section raises, naming it."""
        excinfo = load_broken(tmp_path, lambda cfg: cfg.pop(section))

        assert section in str(excinfo.value)

    def test_missing_field_names_the_field(self, tmp_path: Path) -> None:
        """Dropping one field raises, naming that field."""
        excinfo = load_broken(tmp_path, lambda cfg: cfg["sensor"].pop("max_range"))

        assert "max_range" in str(excinfo.value)

    def test_missing_clearance_radius_names_it(self, tmp_path: Path) -> None:
        """clearance_radius is required, not silently defaulted.

        A silent default here is the exact failure Feature 6 exists to fix:
        a safety radius that looks configured and is not.
        """
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg["planning"].pop("clearance_radius")
        )

        assert "clearance_radius" in str(excinfo.value)

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        """An empty YAML file parses to None and must not be treated as {}."""
        path = tmp_path / "config.yaml"
        path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            load_config(path)


class TestDroneCount:
    """CLAUDE.md scopes the system at 1-5 drones; the schema enforces it."""

    def test_zero_drones_rejected(self, tmp_path: Path) -> None:
        """An empty swarm maps nothing and would report instant success."""
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg["drones"].__setitem__("start_positions", [])
        )

        assert "1" in str(excinfo.value) and "5" in str(excinfo.value)

    def test_six_drones_rejected(self, tmp_path: Path) -> None:
        """Six is past the stated range, so it fails rather than degrading."""
        excinfo = load_broken(
            tmp_path,
            lambda cfg: cfg["drones"].__setitem__(
                "start_positions", [[float(i), 0.0, 1.0] for i in range(6)]
            ),
        )

        assert "6" in str(excinfo.value)

    def test_one_drone_accepted(self, tmp_path: Path) -> None:
        """One drone is the low end of the range, not an error."""
        config = copy.deepcopy(valid_config())
        config["drones"]["start_positions"] = [[0.0, 0.0, 1.0]]

        loaded = load_config(write_config(tmp_path, config))

        assert loaded.drones.count == 1

    def test_five_drones_accepted(self, tmp_path: Path) -> None:
        """Five drones is the high end of the range, not an error."""
        config = copy.deepcopy(valid_config())
        config["drones"]["start_positions"] = [
            [0.0, float(i) * 2.0, 1.0] for i in range(5)
        ]

        loaded = load_config(write_config(tmp_path, config))

        assert loaded.drones.count == 5

    def test_start_position_must_have_three_coordinates(self, tmp_path: Path) -> None:
        """A 2D start position would silently drop the altitude."""
        excinfo = load_broken(
            tmp_path,
            lambda cfg: cfg["drones"].__setitem__(
                "start_positions", [[0.0, 0.0], [0.0, 2.0, 1.0]]
            ),
        )

        assert "three" in str(excinfo.value) or "3" in str(excinfo.value)


class TestNumericBounds:
    """Values that are physically meaningless are rejected at startup."""

    @pytest.mark.parametrize(
        ("section", "field"),
        [
            ("planning", "clearance_radius"),
            ("planning", "spread_radius"),
            ("sensor", "exclusion_radius"),
            ("coordination", "min_separation"),
        ],
    )
    def test_negative_radius_rejected(
        self, tmp_path: Path, section: str, field: str
    ) -> None:
        """A negative radius has no meaning; it is caught, not clamped."""
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg[section].__setitem__(field, -0.1)
        )

        assert field in str(excinfo.value)

    def test_zero_radius_accepted(self, tmp_path: Path) -> None:
        """Zero disables a filter deliberately and stays legal."""
        config = copy.deepcopy(valid_config())
        config["sensor"]["exclusion_radius"] = 0.0
        config["planning"]["clearance_radius"] = 0.0

        loaded = load_config(write_config(tmp_path, config))

        assert loaded.sensor.exclusion_radius == 0.0
        assert loaded.planning.clearance_radius == 0.0

    @pytest.mark.parametrize(
        ("section", "field"),
        [
            ("map", "resolution"),
            ("sensor", "num_rays"),
            ("sensor", "max_range"),
            ("map", "grid_width"),
            ("map", "grid_height"),
            ("coordination", "max_ticks"),
        ],
    )
    def test_non_positive_rejected(
        self, tmp_path: Path, section: str, field: str
    ) -> None:
        """Values that must be strictly positive reject zero as well."""
        excinfo = load_broken(tmp_path, lambda cfg: cfg[section].__setitem__(field, 0))

        assert field in str(excinfo.value)

    def test_negative_max_wait_ticks_rejected(self, tmp_path: Path) -> None:
        """Zero wait ticks is legal (abandon immediately); negative is not."""
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg["coordination"].__setitem__("max_wait_ticks", -1)
        )

        assert "max_wait_ticks" in str(excinfo.value)

    def test_non_numeric_value_rejected(self, tmp_path: Path) -> None:
        """A string where a number belongs fails loudly, not at first use."""
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg["map"].__setitem__("resolution", "fine")
        )

        assert "resolution" in str(excinfo.value)

    def test_bool_is_not_an_integer(self, tmp_path: Path) -> None:
        """`num_rays: true` is a typo, not a 1-ray sensor.

        bool subclasses int in Python, so a naive isinstance check accepts it
        and the mission runs with one ray.
        """
        excinfo = load_broken(
            tmp_path, lambda cfg: cfg["sensor"].__setitem__("num_rays", True)
        )

        assert "num_rays" in str(excinfo.value)


class TestGridCoversStartPositions:
    """The grid must contain every drone's start position.

    This replaces the originally-planned "grid covers the scene" check:
    `config` cannot know the scene's extent without parsing the MJCF, and
    that geometry belongs to `simulation`. Containment of the start
    positions is the part visible from inside `config`.
    """

    def test_start_position_outside_grid_rejected(self, tmp_path: Path) -> None:
        """A drone starting off-grid can never be mapped or planned for."""
        excinfo = load_broken(
            tmp_path,
            lambda cfg: cfg["drones"].__setitem__(
                "start_positions", [[100.0, 0.0, 1.0]]
            ),
        )

        message = str(excinfo.value)
        assert "grid" in message.lower()
        assert "100" in message

    def test_start_position_on_far_edge_rejected(self, tmp_path: Path) -> None:
        """The far edge is exclusive — origin + width*resolution is out.

        A drone exactly on the upper bound floors to cell index `grid_width`,
        one past the last valid column.
        """
        excinfo = load_broken(
            tmp_path,
            lambda cfg: cfg["drones"].__setitem__(
                "start_positions", [[25.0, 0.0, 1.0]]
            ),
        )

        assert "grid" in str(excinfo.value).lower()

    def test_start_position_on_origin_corner_accepted(self, tmp_path: Path) -> None:
        """The origin corner is inclusive — it is cell (0, 0)."""
        config = copy.deepcopy(valid_config())
        config["drones"]["start_positions"] = [[-25.0, -25.0, 1.0]]

        loaded = load_config(write_config(tmp_path, config))

        assert loaded.drones.start_positions == ((-25.0, -25.0, 1.0),)


class TestShippedScenarios:
    """The configs committed in `scenarios/` must satisfy their own schema."""

    @pytest.mark.parametrize("scenario", ["small_indoor", "large_indoor"])
    def test_scenario_config_loads(self, scenario: str) -> None:
        """A shipped scenario that fails validation is a broken scenario."""
        path = (
            Path(__file__).resolve().parents[3] / "scenarios" / scenario / "config.yaml"
        )

        config = load_config(path)

        assert 1 <= config.drones.count <= 5
        assert config.map.resolution > 0
