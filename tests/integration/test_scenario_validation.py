"""Scenario validation and installation — the D2 rules for uploaded rooms.

Uses the real MuJoCo scenes and small inline MJCF; nothing is mocked. The one
monkeypatch below is a tripwire, not a stand-in: it fails the test if the
validator reaches the simulator when it should have stopped first.

Every write goes into pytest tmp dirs — `install_scenario` takes both roots as
arguments precisely so tests never touch the real `scenarios/` or assets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from swarm_mapping.app import validation
from swarm_mapping.app.validation import validate_scenario
from swarm_mapping.cli import resolve_scene_path

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = REPO_ROOT / "scenarios"
SHIPPED = sorted(p.parent.name for p in SCENARIOS.glob("*/config.yaml"))


def scenario_dict(name: str) -> dict[str, Any]:
    """A shipped scenario's config as a raw dict, for targeted edits."""
    raw: dict[str, Any] = yaml.safe_load((SCENARIOS / name / "config.yaml").read_text())
    return copy.deepcopy(raw)


def write_config(directory: Path, raw: dict[str, Any]) -> Path:
    """Write a raw config dict and return its path."""
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


class TestShippedScenarios:
    """Test case 6: every shipped scenario validates clean."""

    def test_there_are_shipped_scenarios(self) -> None:
        """Guards the parametrisation below against an empty glob."""
        assert len(SHIPPED) >= 4

    @pytest.mark.parametrize("name", SHIPPED)
    def test_validates_clean(self, name: str) -> None:
        """No problems under the default rules."""
        assert validate_scenario(SCENARIOS / name / "config.yaml") == []

    @pytest.mark.parametrize("name", ["large_indoor", "loop_indoor"])
    def test_five_spawn_scenarios_meet_the_upload_rule(self, name: str) -> None:
        """The shipped five-spawn scenarios would be accepted as uploads too."""
        path = SCENARIOS / name / "config.yaml"
        assert validate_scenario(path, require_five=True) == []


class TestSpawnCount:
    """Test case 7: "up to 5 drones" means five usable start positions."""

    def test_fewer_than_five_is_rejected_with_the_count(self) -> None:
        """small_indoor declares three; the message says three of five."""
        problems = validate_scenario(
            SCENARIOS / "small_indoor" / "config.yaml", require_five=True
        )

        assert len(problems) == 1
        assert "3" in problems[0]
        assert "5" in problems[0]


class TestSpawnInsideGeometry:
    """Test case 8: a spawn inside a wall is named by index."""

    def test_spawn_in_a_wall_is_rejected_by_index(self, tmp_path: Path) -> None:
        """The problem names the spawn index and the geom it is inside."""
        raw = scenario_dict("small_indoor")
        # small_indoor's east wall is centred on x = 10 with 0.1 m half-width.
        raw["drones"]["start_positions"][1] = [9.95, 0.0, 1.0]

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "start_positions[1]" in problems[0]
        assert "wall_east" in problems[0]

    def test_geometry_is_checked_on_the_compiled_scene(self, tmp_path: Path) -> None:
        """The obstacle is found where MuJoCo compiles it, not where XML says.

        Its position comes from its parent body's frame, so reading the
        geom's own `pos` attribute would put it at the origin and miss the
        spawn entirely.
        """
        scene = tmp_path / "pillar.xml"
        scene.write_text(PILLAR_SCENE)
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = str(scene)
        raw["drones"]["start_positions"] = [
            [-3.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, -3.0, 1.0],
        ]

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "start_positions[1]" in problems[0]
        assert "pillar" in problems[0]

    def test_every_offending_spawn_is_reported(self, tmp_path: Path) -> None:
        """All bad spawns are listed at once, and good ones are not."""
        raw = scenario_dict("small_indoor")
        raw["drones"]["start_positions"][0] = [9.95, 0.0, 1.0]
        raw["drones"]["start_positions"][2] = [-9.95, 0.0, 1.0]

        problems = validate_scenario(write_config(tmp_path, raw))

        joined = "\n".join(problems)
        assert "start_positions[0]" in joined
        assert "start_positions[2]" in joined
        assert "start_positions[1]" not in joined


class TestSpawnSeparation:
    """Test case 9: two spawns closer than `min_separation` are rejected."""

    def test_spawns_too_close_are_rejected(self, tmp_path: Path) -> None:
        """0.3 m apart against a 0.5 m min_separation."""
        raw = scenario_dict("small_indoor")
        raw["drones"]["start_positions"][1] = [0.0, 0.3, 1.0]  # sep 0.5 m

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "min_separation" in problems[0]


class TestMissingScene:
    """Test case 10: a missing scene is caught before MuJoCo is touched."""

    def test_missing_scene_is_rejected_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One problem naming the scene, and no simulator ever built."""

        def tripwire(*_args: object, **_kwargs: object) -> None:
            pytest.fail("the validator built a simulation for a missing scene")

        monkeypatch.setattr(validation, "SimulationEngine", tripwire)
        monkeypatch.setattr(validation, "build_mission", tripwire)
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = "no_such_room.xml"

        problems = validate_scenario(write_config(tmp_path, raw))

        assert len(problems) == 1
        assert "no_such_room.xml" in problems[0]
        assert str(resolve_scene_path("no_such_room.xml")) in problems[0]


class TestNeverRaises:
    """R7: bad input is a list of problems, never an exception."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        """A path that does not exist is a problem."""
        problems = validate_scenario(tmp_path / "absent.yaml")
        assert len(problems) == 1
        assert "absent.yaml" in problems[0]

    @pytest.mark.parametrize(
        "text", ["key: [unclosed", "- just\n- a list\n", ""], ids=str
    )
    def test_unparseable_config(self, tmp_path: Path, text: str) -> None:
        """Broken YAML, a list, or an empty file."""
        path = tmp_path / "config.yaml"
        path.write_text(text)
        assert len(validate_scenario(path)) == 1

    def test_schema_error_names_the_key(self, tmp_path: Path) -> None:
        """A `parse_config` error becomes the problem, key and all."""
        raw = scenario_dict("small_indoor")
        del raw["sensor"]["max_range"]

        (problem,) = validate_scenario(write_config(tmp_path, raw))
        assert "sensor.max_range" in problem

    def test_broken_scene_xml(self, tmp_path: Path) -> None:
        """MuJoCo's compile error becomes a problem."""
        scene = tmp_path / "broken.xml"
        scene.write_text("<mujoco><worldbody><geom type='nonsense'/></worldbody>")
        raw = scenario_dict("small_indoor")
        raw["scene"]["path"] = str(scene)

        (problem,) = validate_scenario(write_config(tmp_path, raw))
        assert "scene" in problem


PILLAR_SCENE = """<mujoco model="pillar">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <body name="pillar_mount" pos="3 0 0">
      <geom name="pillar" type="box" pos="0 0 1.5" size="0.5 0.5 1.5"/>
    </body>
  </worldbody>
</mujoco>
"""
