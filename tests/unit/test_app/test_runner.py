"""Tests for the console's runner: a console run is exactly a CLI command.

Pure logic plus one real subprocess (a trivial Python one-liner). No mission is
ever started here: the argv is asserted, not executed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from swarm_mapping.app import runner
from swarm_mapping.app.runner import (
    RunRequest,
    build_argv,
    new_run_dir,
    run_dir_name,
    scenario_choices,
)
from swarm_mapping.records import VARIANTS

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_SOURCE = REPO_ROOT / "src" / "swarm_mapping" / "cli.py"
SMALL_INDOOR = REPO_ROOT / "scenarios" / "small_indoor" / "config.yaml"
NOW = datetime(2026, 9, 21, 14, 30, 5)


def request(**overrides: object) -> RunRequest:
    """A small_indoor request, with fields overridden per test."""
    fields: dict[str, object] = {
        "config_path": Path("scenarios/small_indoor/config.yaml"),
        "scenario": "small_indoor",
        "drones": 3,
        "variant": "A+B",
        "view": False,
    }
    fields.update(overrides)
    return RunRequest(**fields)  # type: ignore[arg-type]


def flag_value(argv: list[str], flag: str) -> str:
    """The argument following `flag` in `argv`."""
    return argv[argv.index(flag) + 1]


class TestVariants:
    """Sprint test 13: the selector's labels are exactly the four pairs."""

    def test_the_four_labels_map_to_exactly_the_four_pairs(self) -> None:
        """The table the selector offers is the sprint plan's D2 table, no more."""
        assert VARIANTS == {
            "baseline": ("greedy", 0),
            "A": ("global", 0),
            "B": ("greedy", 3),
            "A+B": ("global", 3),
        }

    @pytest.mark.parametrize(
        ("label", "assignment", "tolerance"),
        [
            ("baseline", "greedy", "0"),
            ("A", "global", "0"),
            ("B", "greedy", "3"),
            ("A+B", "global", "3"),
        ],
    )
    def test_argv_carries_the_variant_pair(
        self, label: str, assignment: str, tolerance: str
    ) -> None:
        """Each label reaches the CLI as its (assignment, tolerance) pair."""
        argv = build_argv(request(variant=label), Path("runs/x"))

        assert flag_value(argv, "--assignment") == assignment
        assert flag_value(argv, "--target-tolerance") == tolerance

    def test_an_unknown_variant_is_rejected(self) -> None:
        """A label outside the four cannot become a request."""
        with pytest.raises(ValueError, match="variant"):
            request(variant="C")


class TestArgv:
    """C4: the subprocess command, flag for flag."""

    def test_full_argv(self) -> None:
        """The whole command, as a user would type it."""
        argv = build_argv(request(drones=2, variant="B"), Path("runs/r1"))

        assert argv == [
            sys.executable,
            "-m",
            "swarm_mapping.cli",
            "--config",
            "scenarios/small_indoor/config.yaml",
            "--output",
            "runs/r1",
            "--drones",
            "2",
            "--assignment",
            "greedy",
            "--target-tolerance",
            "3",
        ]

    @pytest.mark.parametrize("view", [False, True])
    def test_view_flag_present_iff_requested(self, view: bool) -> None:
        """`--view` is added for the MuJoCo view and only then."""
        argv = build_argv(request(view=view), Path("runs/r1"))

        assert ("--view" in argv) is view

    def test_every_flag_is_one_the_cli_declares(self) -> None:
        """Guards against a spelling drift between runner and `cli.main`."""
        declared = set(
            re.findall(r'add_argument\(\s*"(--[a-z-]+)"', CLI_SOURCE.read_text())
        )
        argv = build_argv(request(view=True), Path("runs/r1"))

        used = {arg for arg in argv if arg.startswith("--")}
        assert used <= declared

    @pytest.mark.parametrize("drones", [0, -1])
    def test_fewer_than_one_drone_is_rejected(self, drones: int) -> None:
        """A swarm has at least one drone."""
        with pytest.raises(ValueError, match="drones"):
            request(drones=drones)


class TestRunDirectory:
    """C3: named by time, scenario, variant and swarm size; never reused."""

    def test_name_format(self) -> None:
        """`YYYYmmdd-HHMMSS_<scenario>_<variant>_<N>d`, `A+B` as `AB`."""
        assert run_dir_name(request(), NOW) == "20260921-143005_small_indoor_AB_3d"

    def test_plain_variants_are_kept_as_is(self) -> None:
        """Only `+` is rewritten; other labels appear verbatim."""
        name = run_dir_name(request(variant="baseline", drones=1), NOW)

        assert name == "20260921-143005_small_indoor_baseline_1d"

    def test_creates_the_directory_and_its_root(self, tmp_path: Path) -> None:
        """A missing runs root is created with the run's directory."""
        root = tmp_path / "runs"

        created = new_run_dir(root, request(), NOW)

        assert created == root / "20260921-143005_small_indoor_AB_3d"
        assert created.is_dir()

    def test_collisions_get_numbered_suffixes(self, tmp_path: Path) -> None:
        """Two runs in the same second never share a directory."""
        first = new_run_dir(tmp_path, request(), NOW)
        second = new_run_dir(tmp_path, request(), NOW)
        third = new_run_dir(tmp_path, request(), NOW)

        assert second.name == f"{first.name}-2"
        assert third.name == f"{first.name}-3"


class TestRoots:
    """C6: roots come from the environment, with the documented defaults."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without overrides: `runs/`, `scenarios/` and the bundled assets."""
        for name in ("SWARM_RUNS_ROOT", "SWARM_SCENARIOS_ROOT", "SWARM_ASSETS_DIR"):
            monkeypatch.delenv(name, raising=False)

        assert runner.runs_root() == Path("runs")
        assert runner.scenarios_root() == Path("scenarios")
        assert runner.assets_dir() == (
            REPO_ROOT / "src" / "swarm_mapping" / "simulation" / "assets"
        )

    def test_environment_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Each root follows its environment variable."""
        monkeypatch.setenv("SWARM_RUNS_ROOT", str(tmp_path / "r"))
        monkeypatch.setenv("SWARM_SCENARIOS_ROOT", str(tmp_path / "s"))
        monkeypatch.setenv("SWARM_ASSETS_DIR", str(tmp_path / "a"))

        assert runner.runs_root() == tmp_path / "r"
        assert runner.scenarios_root() == tmp_path / "s"
        assert runner.assets_dir() == tmp_path / "a"


class TestScenarioChoices:
    """The Run page's menu: every scenario, valid or not, with the reason."""

    def test_lists_valid_and_invalid_scenarios_by_name(self, tmp_path: Path) -> None:
        """Invalid scenarios are listed with their problems, not dropped."""
        good = tmp_path / "b_good"
        good.mkdir()
        (good / "config.yaml").write_text(SMALL_INDOOR.read_text())
        bad = tmp_path / "a_bad"
        bad.mkdir()
        raw = yaml.safe_load(SMALL_INDOOR.read_text())
        raw["scene"]["path"] = "no_such_scene.xml"
        (bad / "config.yaml").write_text(yaml.safe_dump(raw))
        (tmp_path / "not_a_scenario").mkdir()

        choices = scenario_choices(tmp_path)

        assert [c.name for c in choices] == ["a_bad", "b_good"]
        assert choices[0].problems and "no_such_scene.xml" in choices[0].problems[0]
        assert choices[1].problems == []
        assert choices[1].spawns == 3
        assert choices[1].config_path == good / "config.yaml"

    def test_unparseable_config_has_no_spawn_count(self, tmp_path: Path) -> None:
        """A config that does not load has problems and no count."""
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "config.yaml").write_text("drones: [")

        (choice,) = scenario_choices(tmp_path)

        assert choice.spawns is None
        assert choice.problems

    def test_missing_root_is_empty(self, tmp_path: Path) -> None:
        """No scenarios root means no scenarios, not an error."""
        assert scenario_choices(tmp_path / "nowhere") == []


class TestLaunch:
    """The subprocess is real; only its command is swapped for a harmless one."""

    def test_output_goes_to_console_txt_in_the_new_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Both output streams land in `console.txt`."""
        monkeypatch.setattr(
            runner,
            "build_argv",
            lambda _request, _dir: [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ],
        )

        process, run_dir = runner.launch(request(), tmp_path, now=NOW)
        assert process.wait(timeout=30) == 0

        assert run_dir == tmp_path / "20260921-143005_small_indoor_AB_3d"
        captured = (run_dir / "console.txt").read_text().split()
        assert sorted(captured) == ["err", "out"]


class TestNoStreamlitInCore:
    """C2: the core and the console's logic import without streamlit."""

    def test_imports_with_streamlit_blocked(self) -> None:
        """A fresh interpreter that cannot import streamlit still can these."""
        script = textwrap.dedent(
            """
            import sys

            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "streamlit" or name.startswith("streamlit."):
                        raise ImportError("streamlit is blocked")
                    return None

            sys.meta_path.insert(0, Block())
            import swarm_mapping
            import swarm_mapping.app.history
            import swarm_mapping.app.runner
            import swarm_mapping.app.validation
            import swarm_mapping.cli
            assert "streamlit" not in sys.modules
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
