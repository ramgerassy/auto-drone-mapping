"""A real run writes its records: `log.jsonl`, `run.json`, paths and routes.

End to end through `run_pipeline` on `small_indoor`, capped at a few dozen
ticks so the whole module stays in seconds. The record's *contracts* (schema,
parsing, history listing) are unit-tested in `tests/unit/test_records.py`;
this module checks that the CLI fills them with the run it actually did.
"""

from __future__ import annotations

import collections
import copy
import json
import re
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from swarm_mapping.cli import MissionResult, describe_inputs, run_pipeline
from swarm_mapping.config.loader import load_config
from swarm_mapping.records import LOG_FILE, RUN_FILE, load_run

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_INDOOR = REPO_ROOT / "scenarios" / "small_indoor" / "config.yaml"

# Short: every test that needs a real two-drone run shares one this long.
CAPPED_TICKS = 10

# What an event name looks like: a stable machine name, never prose.
EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def capped_config(directory: Path, max_ticks: int = CAPPED_TICKS) -> Path:
    """small_indoor, capped so a run takes a second or two."""
    raw: dict[str, Any] = copy.deepcopy(yaml.safe_load(SMALL_INDOOR.read_text()))
    raw["coordination"]["max_ticks"] = max_ticks
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def read_log(directory: Path) -> list[dict[str, Any]]:
    """The run's log, one parsed object per line."""
    text = (directory / LOG_FILE).read_text()
    return [json.loads(line) for line in text.splitlines()]


def run_json(directory: Path) -> dict[str, Any]:
    """The raw `run.json` document."""
    data: dict[str, Any] = json.loads((directory / RUN_FILE).read_text())
    return data


def without_wall_clock(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the two fields that measure the wall clock rather than the run."""
    out = copy.deepcopy(data)
    del out["started_at"]
    del out["outputs"]["wall_seconds"]
    return out


def assert_event_names(directory: Path) -> None:
    """Every log event and every `event_counts` key is a machine name."""
    events = [line["event"] for line in read_log(directory)]
    keys = list(run_json(directory)["outputs"]["event_counts"])
    for name in events + keys:
        assert EVENT_NAME.match(name), f"not an event name: {name!r}"


def room_config(directory: Path, scene_xml: str, max_ticks: int) -> Path:
    """A one-drone config on a 10 m inline scene, for tests needing many ticks.

    small_indoor's 200x200 grid costs about 0.1 s a tick; this 100x100 room
    costs roughly a tenth of that.
    """
    scene = directory / "room.xml"
    scene.write_text(scene_xml)
    raw: dict[str, Any] = yaml.safe_load(SMALL_INDOOR.read_text())
    raw["scene"]["path"] = str(scene)
    raw["drones"]["start_positions"] = [[-3.0, 0.0, 1.0]]
    raw["map"].update(
        {"origin_x": -5.0, "origin_y": -5.0, "grid_width": 100, "grid_height": 100}
    )
    raw["coordination"]["max_ticks"] = max_ticks
    path = directory / "room.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def snapshot_to_yaml(snapshot: dict[str, Any], directory: Path) -> Path:
    """Turn a run.json config snapshot back into a loadable scenario file.

    The snapshot is `dataclasses.asdict(ScenarioConfig)`, whose sections are the
    YAML sections field for field — only `scene_path` is spelled `scene.path`
    in YAML.
    """
    document = copy.deepcopy(snapshot)
    document["scene"] = {"path": document.pop("scene_path")}
    path = directory / "replay.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture(scope="module")
def two_drone_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, MissionResult]:
    """One capped two-drone run, shared by the read-only tests below."""
    workdir = tmp_path_factory.mktemp("two_drone")
    config = capped_config(workdir)
    output = workdir / "run"
    result = run_pipeline(config, output, drones=2)
    return config, output, result


class TestEveryRunIsRecorded:
    """Test case 1: each run writes both files into its own directory."""

    def test_both_files_are_written(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """run.json and log.jsonl land in `--output`."""
        _, output, _ = two_drone_run
        assert (output / RUN_FILE).is_file()
        assert (output / LOG_FILE).is_file()

    def test_files_lists_exactly_what_the_run_wrote(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """`files` names every file in the run directory, and only those."""
        _, output, _ = two_drone_run
        listed = run_json(output)["outputs"]["files"]
        assert listed == sorted(p.name for p in output.iterdir())

    def test_a_second_run_leaves_the_first_record_untouched(
        self, two_drone_run: tuple[Path, Path, MissionResult], tmp_path: Path
    ) -> None:
        """Runs in different directories never touch each other's records."""
        _, first, _ = two_drone_run
        before = (first / RUN_FILE).read_bytes(), (first / LOG_FILE).read_bytes()

        config = capped_config(tmp_path, max_ticks=5)
        run_pipeline(config, tmp_path / "second", drones=1)

        after = (first / RUN_FILE).read_bytes(), (first / LOG_FILE).read_bytes()
        assert after == before
        assert load_run(tmp_path / "second").inputs.drones == 1

    def test_rerunning_into_the_same_directory_replaces_the_log(
        self, tmp_path: Path
    ) -> None:
        """Same `--output` twice: files are replaced, as they always were (R1).

        The log is replaced too, never spliced onto the previous run's.
        """
        config = capped_config(tmp_path, max_ticks=5)
        output = tmp_path / "run"
        run_pipeline(config, output, drones=1)
        first = read_log(output)
        run_pipeline(config, output, drones=1)

        assert read_log(output) == first

    @pytest.mark.parametrize("fault", ["bad_config", "too_many_drones"])
    def test_a_run_that_fails_validation_leaves_nothing(
        self, tmp_path: Path, fault: str
    ) -> None:
        """A bad config or `--drones` override fails before DIR is touched."""
        config = capped_config(tmp_path, max_ticks=5)
        drones = 2
        if fault == "bad_config":
            raw = yaml.safe_load(config.read_text())
            del raw["sensor"]
            config.write_text(yaml.safe_dump(raw))
        else:
            drones = 9
        output = tmp_path / "run"

        with pytest.raises(ValueError):
            run_pipeline(config, output, drones=drones)

        assert not output.exists()


class TestRunJson:
    """Test case 2 and R4: run.json holds the inputs and the outcome."""

    def test_outcome_matches_the_mission_result(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """The outputs are the MissionResult the caller received."""
        _, output, result = two_drone_run
        outputs = run_json(output)["outputs"]

        assert outputs["ticks"] == result.ticks == CAPPED_TICKS
        assert outputs["coverage"] == result.coverage
        assert outputs["blocked"] == result.blocked
        assert outputs["unreachable_frontiers"] == result.unreachable_frontiers
        assert outputs["tick_capped"] == result.tick_capped
        assert outputs["succeeded"] == result.succeeded
        assert outputs["wall_seconds"] > 0

    def test_inputs_record_what_was_run(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """Config path, snapshot, swarm size, allocation and variant."""
        config, output, _ = two_drone_run
        inputs = run_json(output)["inputs"]

        assert inputs["config_path"] == str(config.resolve())
        assert inputs["drones"] == 2
        # small_indoor ships A+B: global allocation, 3-cell tolerance.
        assert inputs["assignment"] == "global"
        assert inputs["target_tolerance_cells"] == 3
        assert inputs["variant"] == "A+B"
        assert inputs["view"] is False
        assert inputs["config"] == json.loads(json.dumps(asdict(load_config(config))))

    def test_overrides_are_in_the_record_and_the_snapshot(self, tmp_path: Path) -> None:
        """A CLI override is part of the run.

        The snapshot is the config *as run*, not as written in the file.
        """
        output = tmp_path / "run"
        run_pipeline(
            capped_config(tmp_path, max_ticks=5),
            output,
            drones=1,
            assignment="greedy",
            target_tolerance=0,
        )
        inputs = run_json(output)["inputs"]

        assert inputs["variant"] == "baseline"
        assert inputs["config"]["coordination"]["assignment"] == "greedy"
        assert inputs["config"]["coordination"]["target_tolerance_cells"] == 0

    def test_per_drone_path_stats(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """One PathLog summary per drone flown."""
        _, output, result = two_drone_run
        paths = run_json(output)["outputs"]["paths"]

        assert sorted(paths) == ["0", "1"]
        for stats in paths.values():
            assert stats["ticks"] == result.ticks
            assert 1 <= stats["distinct_cells"] <= stats["route_length"]

    def test_event_counts_match_the_log(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """`event_counts` agrees with the log file it summarises."""
        _, output, _ = two_drone_run
        counted = collections.Counter(line["event"] for line in read_log(output))

        assert run_json(output)["outputs"]["event_counts"] == dict(counted)

    def test_rerunning_the_record_reproduces_the_run(
        self, two_drone_run: tuple[Path, Path, MissionResult], tmp_path: Path
    ) -> None:
        """Re-running from run.json alone reproduces the run exactly.

        Test case 2 and R8 in one re-run: the config snapshot plus the drone
        count is the whole recipe, and a second headless run of the same inputs
        gives the same map bytes, the same log bytes, and the same record except
        for the wall clock. `config_path` differs only because the replay is
        read from a different file.
        """
        _, output, _ = two_drone_run
        original = run_json(output)

        replay = tmp_path / "replay"
        run_pipeline(
            snapshot_to_yaml(original["inputs"]["config"], tmp_path),
            replay,
            drones=original["inputs"]["drones"],
        )
        replayed = run_json(replay)

        assert (replay / "map.npz").read_bytes() == (output / "map.npz").read_bytes()
        assert (replay / LOG_FILE).read_bytes() == (output / LOG_FILE).read_bytes()
        expected = without_wall_clock(original)
        actual = without_wall_clock(replayed)
        assert actual["outputs"] == expected["outputs"]
        del expected["inputs"]["config_path"], actual["inputs"]["config_path"]
        assert actual == expected


class TestLogFile:
    """Test case 3: every line is JSON with `event` and `tick`.

    Of the failure half, `failure_injected` (Feature 9) is checked here;
    `drone_failed` is Feature 10's and is checked where it lands.
    """

    def test_every_line_has_event_and_tick(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """No line is missing either key, whatever its call site passed."""
        _, output, _ = two_drone_run
        lines = read_log(output)

        assert lines
        for line in lines:
            assert isinstance(line["event"], str)
            assert isinstance(line["tick"], int)

    def test_events_are_machine_names(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """Prose messages carry an `event` name, so no key embeds a number."""
        _, output, _ = two_drone_run
        assert_event_names(output)
        assert "mission_started" in run_json(output)["outputs"]["event_counts"]

    def test_ticks_follow_the_mission(self, tmp_path: Path) -> None:
        """Lines carry the ticks completed when they were written.

        The start line precedes the first tick (0); the progress line the CLI
        logs after tick 50 carries 50 — a stamp, since that call passes no
        tick of its own. Run on a small inline room to reach 50 ticks cheaply.
        """
        output = tmp_path / "run"
        result = run_pipeline(room_config(tmp_path, OPEN_SCENE, 50), output)
        lines = read_log(output)
        ticks = [line["tick"] for line in lines]

        assert result.ticks == 50
        assert ticks == sorted(ticks)
        assert lines[0]["event"] == "mission_started"
        assert lines[0]["tick"] == 0
        progress = [line for line in lines if line["event"] == "mission_progress"]
        assert [line["tick"] for line in progress] == [50]
        assert 0.0 < progress[0]["coverage"] <= 1.0
        assert_event_names(output)

    def test_structured_events_keep_their_fields(self, tmp_path: Path) -> None:
        """The master's own events arrive with their `extra=` payload.

        Uses a room split by a gap too narrow to fly — it ends `blocked` in
        well under a second, which guarantees a `mission_blocked` event.
        """
        config = room_config(tmp_path, BLOCKED_SCENE, 400)
        result = run_pipeline(config, tmp_path / "run")

        blocked = [
            line
            for line in read_log(tmp_path / "run")
            if line["event"] == "mission_blocked"
        ]
        assert blocked
        # The master logs this during its final tick, before counting it done:
        # `tick` is ticks completed when the line was written.
        assert blocked[-1]["tick"] == result.ticks - 1
        assert blocked[-1]["unreachable_frontiers"] == result.unreachable_frontiers
        assert blocked[-1]["level"] == "WARNING"
        assert_event_names(tmp_path / "run")

    def test_an_injected_failure_reaches_the_log(self, tmp_path: Path) -> None:
        """Test case 3: `failure_injected` and `drone_failed` land together.

        Both are in a failure run's log and both counted in `run.json`.
        `failure_injected` is Feature 9's (the simulator's ground truth);
        `drone_failed` is Feature 10's (the master's detection, from missed
        heartbeats — small_indoor's `heartbeat_timeout_ticks` is 3, so a
        drone that goes silent at tick 5 is declared lost at tick 7). Both
        are asserted here so this test closes the Feature 10 x Feature 12
        gap: a failure run's *detection* event, not just its injection, is
        on record.
        """
        raw: dict[str, Any] = yaml.safe_load(SMALL_INDOOR.read_text())
        raw["coordination"]["max_ticks"] = 10
        raw["failures"] = [{"drone": 1, "tick": 5, "mode": "silent"}]
        config = tmp_path / "failure.yaml"
        config.write_text(yaml.safe_dump(raw))
        output = tmp_path / "run"

        run_pipeline(config, output, drones=2)

        injected = [
            line for line in read_log(output) if line["event"] == "failure_injected"
        ]
        assert injected == [
            {
                "event": "failure_injected",
                "tick": 5,
                "level": "INFO",
                "drone_id": 1,
                "mode": "silent",
            }
        ]
        failed = [line for line in read_log(output) if line["event"] == "drone_failed"]
        assert len(failed) == 1
        assert failed[0]["tick"] == 7
        assert failed[0]["drone_id"] == 1
        assert failed[0]["health"] == "lost"
        assert isinstance(failed[0]["released"], list)
        record = run_json(output)
        assert record["outputs"]["event_counts"]["failure_injected"] == 1
        assert record["outputs"]["event_counts"]["drone_failed"] == 1
        assert record["inputs"]["config"]["failures"] == [
            {"drone_id": 1, "tick": 5, "mode": "silent"}
        ]
        assert_event_names(output)


class TestPathsAlwaysRecorded:
    """R5: paths.json and one route PNG per drone, with or without a flag."""

    def test_written_without_any_flag(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """A plain run writes its paths; heatmaps still need their flag."""
        _, output, _ = two_drone_run
        assert (output / "paths.json").is_file()
        assert (output / "route_drone_0.png").is_file()
        assert (output / "route_drone_1.png").is_file()
        assert not list(output.glob("visits_drone_*.png")), (
            "visit heatmaps stay behind --visit-heatmaps"
        )

    def test_heatmaps_still_come_with_their_flag(self, tmp_path: Path) -> None:
        """`--visit-heatmaps` adds the heatmaps and they are listed in files."""
        output = tmp_path / "run"
        run_pipeline(
            capped_config(tmp_path, max_ticks=5),
            output,
            drones=1,
            visit_heatmaps=True,
        )

        assert (output / "visits_drone_0.png").is_file()
        assert (output / "route_drone_0.png").is_file()
        assert "visits_drone_0.png" in run_json(output)["outputs"]["files"]


class TestViewIsViewOnly:
    """R8, replacing test case 5 — a viewer cannot open headless.

    That two headless runs of the same inputs agree on everything but the
    wall clock is `TestRunJson::test_rerunning_the_record_reproduces_the_run`.
    Here: `view` changes nothing in the record's inputs but `view`.
    """

    def test_view_is_the_only_input_it_changes(
        self, two_drone_run: tuple[Path, Path, MissionResult]
    ) -> None:
        """Recording `view=True` changes `view` and nothing else."""
        config, _, _ = two_drone_run
        scenario = load_config(config)

        headless = describe_inputs(config, scenario, drones=2, view=False)
        viewed = describe_inputs(config, scenario, drones=2, view=True)

        assert viewed == replace(headless, view=True)
        assert headless.view is False


class TestStderrIsUnchanged:
    """The file log captures INFO; the terminal still shows only what it did."""

    def test_quiet_cli_prints_no_info_but_logs_it(self, tmp_path: Path) -> None:
        """Without --verbose, INFO reaches log.jsonl but not the terminal."""
        config = capped_config(tmp_path, max_ticks=3)
        output = tmp_path / "run"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "swarm_mapping.cli",
                "--config",
                str(config),
                "--output",
                str(output),
                "--drones",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Capped at 3 ticks: the mission did not converge, exit code 1 as ever.
        assert completed.returncode == 1, completed.stderr
        assert "Starting exploration" not in completed.stderr
        assert "INFO" not in completed.stderr
        events = [line["event"] for line in read_log(output)]
        assert "mission_started" in events


class TestSwarmLostFailsTheRun:
    """A mission that loses every drone did not succeed.

    Even without a tick cap: `CentralizedMaster.is_complete` goes True with
    nothing left to assign, so `tick_capped` alone would miss it
    (`MissionResult.succeeded`).
    """

    def test_every_drone_failing_reports_unsucceeded(self, tmp_path: Path) -> None:
        """Both drones fail silently, early: `run.json` records the failure."""
        raw: dict[str, Any] = yaml.safe_load(SMALL_INDOOR.read_text())
        raw["coordination"]["max_ticks"] = 10
        raw["failures"] = [
            {"drone": 0, "tick": 1, "mode": "silent"},
            {"drone": 1, "tick": 1, "mode": "silent"},
        ]
        config = tmp_path / "swarm_lost.yaml"
        config.write_text(yaml.safe_dump(raw))
        output = tmp_path / "run"

        result = run_pipeline(config, output, drones=2)

        assert result.swarm_lost is True
        assert result.tick_capped is False  # not what stopped this mission
        assert result.succeeded is False
        record = run_json(output)
        assert record["outputs"]["succeeded"] is False
        assert record["outputs"]["tick_capped"] is False

    def test_every_drone_failing_exits_non_zero_through_the_cli(
        self, tmp_path: Path
    ) -> None:
        """The CLI subprocess itself exits non-zero, not just `run_pipeline`."""
        raw: dict[str, Any] = yaml.safe_load(SMALL_INDOOR.read_text())
        raw["coordination"]["max_ticks"] = 10
        raw["failures"] = [
            {"drone": 0, "tick": 1, "mode": "silent"},
            {"drone": 1, "tick": 1, "mode": "silent"},
        ]
        config = tmp_path / "swarm_lost.yaml"
        config.write_text(yaml.safe_dump(raw))
        output = tmp_path / "run"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "swarm_mapping.cli",
                "--config",
                str(config),
                "--output",
                str(output),
                "--drones",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 1, completed.stderr
        assert (
            json.loads((output / RUN_FILE).read_text())["outputs"]["succeeded"] is False
        )

    def test_a_healthy_short_run_is_unaffected(self, tmp_path: Path) -> None:
        """`swarm_lost` stays False and doesn't touch a normal `succeeded`.

        No failures scheduled, and the mission didn't hit the tick cap.
        Uses the small inline `OPEN_SCENE` room (one drone, finishes in well
        under 100 ticks) rather than `small_indoor`, so this stays fast.
        """
        output = tmp_path / "run"

        result = run_pipeline(room_config(tmp_path, OPEN_SCENE, 500), output)

        assert result.swarm_lost is False
        assert result.tick_capped is False
        assert result.succeeded is True
        assert run_json(output)["outputs"]["succeeded"] is True


OPEN_SCENE = """<mujoco model="open">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    <geom name="wall_east" type="box" pos="5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_west" type="box" pos="-5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_north" type="box" pos="0 5 1.5" size="5 0.1 1.5"/>
    <geom name="wall_south" type="box" pos="0 -5 1.5" size="5 0.1 1.5"/>
  </worldbody>
</mujoco>
"""

BLOCKED_SCENE = """<mujoco model="blocked">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    <geom name="wall_east" type="box" pos="5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_west" type="box" pos="-5 0 1.5" size="0.1 5 1.5"/>
    <geom name="wall_north" type="box" pos="0 5 1.5" size="5 0.1 1.5"/>
    <geom name="wall_south" type="box" pos="0 -5 1.5" size="5 0.1 1.5"/>
    <geom name="divider_north" type="box" pos="0 2.55 1.5" size="0.1 2.45 1.5"/>
    <geom name="divider_south" type="box" pos="0 -2.55 1.5" size="0.1 2.45 1.5"/>
  </worldbody>
</mujoco>
"""
