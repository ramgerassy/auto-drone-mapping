"""End-to-end integration tests for the exploration pipeline.

Uses the real MuJoCo scenes — not unit tests. Verifies the full wiring:
load config, build the swarm, explore with `CentralizedMaster`, export.

**These were Sprint-1 tests and have been deliberately re-tagged to Sprint 2.**
They used to guard the lawnmower patrol. Feature 6 deleted that mode, and the
old assertions (output files exist, npz shape, coverage > 80%) would all have
kept passing against exploration — a regression gate quietly guarding a
pipeline that no longer exists is worse than no gate, because it reads as
coverage. The Sprint-1 behaviour is gone, so its tests go with it, and what
replaces them asserts exploration explicitly.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from swarm_mapping.cli import run_pipeline

pytestmark = pytest.mark.sprint(2)  # re-tagged from sprint(1) — see module docstring

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_CONFIG = REPO_ROOT / "scenarios" / "small_indoor" / "config.yaml"

# Tier-1 KPI from CLAUDE.md: coverage >= 95% indoor.
COVERAGE_TARGET = 0.95


def scenario_dict(name: str = "small_indoor") -> dict[str, Any]:
    """Load a shipped scenario config as a raw dict, for targeted overrides.

    Reading the committed file rather than restating it inline means a test
    config cannot silently drift from the scenario it claims to be testing.
    """
    path = REPO_ROOT / "scenarios" / name / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return copy.deepcopy(raw)


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write a config dict to a YAML file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class TestOutputFormat:
    """The exported files keep the format acceptance tests depend on.

    CLAUDE.md calls output format a compatibility surface, so this asserts the
    exact Sprint-1 npz contract: same keys, same shapes, same dtypes.
    """

    def test_pipeline_produces_both_output_files(self, tmp_path: Path) -> None:
        """A single-drone exploration run writes .npz and .png."""
        result = run_pipeline(SCENARIO_CONFIG, tmp_path, drones=1)

        assert result.npz_path.exists()
        assert result.png_path.exists()
        assert result.npz_path == tmp_path / "map.npz"
        assert result.png_path == tmp_path / "map.png"

    def test_npz_keys_and_shapes_match_sprint_1(self, tmp_path: Path) -> None:
        """The npz contract is unchanged by the move to exploration."""
        run_pipeline(SCENARIO_CONFIG, tmp_path, drones=1)

        data = np.load(str(tmp_path / "map.npz"))

        assert set(data.files) == {
            "log_odds",
            "probability",
            "height",
            "resolution",
            "origin",
        }
        assert data["log_odds"].shape == (200, 200)
        assert data["probability"].shape == (200, 200)
        assert data["height"].shape == (200, 200)
        assert data["log_odds"].dtype == np.float64
        assert float(data["resolution"]) == 0.1
        assert np.array_equal(data["origin"], np.array([-10.0, -10.0]))

    def test_png_is_a_valid_image(self, tmp_path: Path) -> None:
        """PNG output is a real image at grid resolution."""
        from PIL import Image

        run_pipeline(SCENARIO_CONFIG, tmp_path, drones=1)

        img = Image.open(str(tmp_path / "map.png"))
        assert img.size == (200, 200)
        assert img.mode == "RGB"


class TestExplorationBeatsPatrol:
    """Exploration is asserted as exploration, not as "some map appeared"."""

    def test_single_drone_explores_and_terminates(self, tmp_path: Path) -> None:
        """One drone finishes the room under its own steam.

        Termination is asserted via `tick_capped`, not via `blocked`: a
        complete run on this scenario *is* blocked, because clearance
        inflation leaves wall-adjacent frontiers visible but unoccupiable.
        Measured: 371 ticks, 98.1% coverage, 3 residual regions — every
        unmapped cell inside an obstacle footprint or in the wall margin.
        """
        result = run_pipeline(SCENARIO_CONFIG, tmp_path, drones=1)

        assert not result.tick_capped, (
            f"mission hit max_ticks at {result.ticks} ticks rather than "
            "finishing; exploration did not converge"
        )
        assert result.ticks > 0
        assert result.coverage >= COVERAGE_TARGET, (
            f"single-drone coverage {result.coverage:.3%} below "
            f"{COVERAGE_TARGET:.0%}; {result.unreachable_frontiers} regions "
            "left unreachable"
        )

    def test_three_drones_reach_the_coverage_kpi(self, tmp_path: Path) -> None:
        """Tier-1 KPI: >= 95% of cells classified, small indoor.

        Asserted rather than assumed — this is the number the project commits
        to, and clearance inflation refusing wall-adjacent frontiers is the
        plausible way it fails.
        """
        result = run_pipeline(SCENARIO_CONFIG, tmp_path)

        assert result.coverage >= COVERAGE_TARGET, (
            f"coverage {result.coverage:.3%} is below the {COVERAGE_TARGET:.0%} "
            f"Tier-1 KPI after {result.ticks} ticks "
            f"(blocked={result.blocked}, "
            f"unreachable={result.unreachable_frontiers})"
        )


class TestMaxTicks:
    """A pathological config must not hang CI."""

    def test_max_ticks_is_honoured(self, tmp_path: Path) -> None:
        """The loop stops at max_ticks and says so."""
        config = scenario_dict()
        config["coordination"]["max_ticks"] = 3

        result = run_pipeline(write_config(tmp_path, config), tmp_path)

        assert result.ticks == 3
        assert result.tick_capped

    def test_a_capped_run_still_exports(self, tmp_path: Path) -> None:
        """Hitting the cap is not a crash — the partial map is still written.

        A mission that gives up without producing its map leaves nothing to
        diagnose the failure with.
        """
        config = scenario_dict()
        config["coordination"]["max_ticks"] = 3

        result = run_pipeline(write_config(tmp_path, config), tmp_path)

        assert result.npz_path.exists()
        assert result.png_path.exists()


class TestBlockedMission:
    """A walled-out mission is reported as blocked, never as success."""

    def blocked_scene(self, tmp_path: Path) -> Path:
        """Write a scene split by a wall whose gap is too narrow to fly.

        The gap is 0.2 m — visible to the rangefinder (rays pass straight
        through it, marking cells beyond as free, which creates frontiers) but
        impassable once obstacles are inflated by `clearance_radius`. That is
        precisely the "walled out, not finished" case: frontiers remain and
        none of them is reachable.
        """
        xml = """<mujoco model="blocked">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="key" directional="true" pos="0 0 8" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.75 0.75 0.78 1"/>
    <geom name="wall_east" type="box" pos="5 0 1.5" size="0.1 5 1.5"
          rgba="0.55 0.58 0.65 1"/>
    <geom name="wall_west" type="box" pos="-5 0 1.5" size="0.1 5 1.5"
          rgba="0.55 0.58 0.65 1"/>
    <geom name="wall_north" type="box" pos="0 5 1.5" size="5 0.1 1.5"
          rgba="0.55 0.58 0.65 1"/>
    <geom name="wall_south" type="box" pos="0 -5 1.5" size="5 0.1 1.5"
          rgba="0.55 0.58 0.65 1"/>
    <!-- Divider at x=0 with a 0.2 m gap centred on y=0, in line with the
         drone's start so rays pass straight through it. -->
    <geom name="divider_north" type="box" pos="0 2.55 1.5" size="0.1 2.45 1.5"
          rgba="0.55 0.58 0.65 1"/>
    <geom name="divider_south" type="box" pos="0 -2.55 1.5" size="0.1 2.45 1.5"
          rgba="0.55 0.58 0.65 1"/>
  </worldbody>
</mujoco>"""
        path = tmp_path / "blocked.xml"
        path.write_text(xml)
        return path

    def blocked_config(self, tmp_path: Path) -> Path:
        """A config for `blocked_scene`, with the drone on the west side."""
        config = scenario_dict()
        config["scene"]["path"] = str(self.blocked_scene(tmp_path))
        config["drones"]["start_positions"] = [[-3.0, 0.0, 1.0]]
        config["map"].update(
            {
                "resolution": 0.1,
                "origin_x": -5.0,
                "origin_y": -5.0,
                "grid_width": 100,
                "grid_height": 100,
            }
        )
        config["coordination"]["max_ticks"] = 400
        return write_config(tmp_path, config)

    def test_blocked_mission_is_reported_as_blocked(self, tmp_path: Path) -> None:
        """Frontiers remain, none reachable — and the result says so."""
        result = run_pipeline(self.blocked_config(tmp_path), tmp_path)

        assert not result.tick_capped, (
            "mission hit the tick cap rather than terminating; this test needs "
            "a genuine walled-out termination to be meaningful"
        )
        assert result.blocked
        assert result.unreachable_frontiers > 0

    def test_a_blocked_mission_does_not_read_as_full_coverage(
        self, tmp_path: Path
    ) -> None:
        """The east half is never mapped, so coverage is visibly short.

        Without `is_blocked`, this run and a complete one are told apart only
        by a coverage number nobody is obliged to look at.
        """
        result = run_pipeline(self.blocked_config(tmp_path), tmp_path)

        assert result.coverage < COVERAGE_TARGET


class TestDeterminism:
    """Same config, same run — CLAUDE.md's hard requirement."""

    def test_two_runs_produce_byte_identical_npz(self, tmp_path: Path) -> None:
        """Byte-level, not approximate.

        `np.savez_compressed` pins its zip entry timestamps to the DOS epoch,
        so the bytes carry no wall-clock and this really is a data comparison.
        """
        config = scenario_dict()
        config["coordination"]["max_ticks"] = 40
        config_path = write_config(tmp_path, config)

        first = tmp_path / "first"
        second = tmp_path / "second"
        run_pipeline(config_path, first)
        run_pipeline(config_path, second)

        assert (first / "map.npz").read_bytes() == (second / "map.npz").read_bytes()

    def test_two_runs_agree_on_the_mission_summary(self, tmp_path: Path) -> None:
        """Tick count and coverage are reproducible, not just the map."""
        config = scenario_dict()
        config["coordination"]["max_ticks"] = 40
        config_path = write_config(tmp_path, config)

        first = run_pipeline(config_path, tmp_path / "first")
        second = run_pipeline(config_path, tmp_path / "second")

        assert first.ticks == second.ticks
        assert first.coverage == second.coverage
        assert first.blocked == second.blocked
