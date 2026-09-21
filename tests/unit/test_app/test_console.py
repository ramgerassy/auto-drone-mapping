"""Streamlit `AppTest` tests for the operator console.

Skipped when the `ui` extra is not installed (CI syncs without it). Every root
the console reads or writes is pointed at a tmp dir through its environment
variable, and no mission is ever started: `runner.launch` — our own function,
not MuJoCo — is replaced to capture the request the Run page builds.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")

import streamlit  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from swarm_mapping.app import runner  # noqa: E402

from .conftest import make_run  # noqa: E402

pytestmark = pytest.mark.sprint(3)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSOLE = REPO_ROOT / "src" / "swarm_mapping" / "app" / "console.py"
SMALL_INDOOR = REPO_ROOT / "scenarios" / "small_indoor" / "config.yaml"
SMALL_SCENE = REPO_ROOT / "src" / "swarm_mapping" / "simulation" / "assets"
TIMEOUT_S = 60


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Empty runs, scenarios and assets roots, wired in through the env."""
    paths = {name: tmp_path / name for name in ("runs", "scenarios", "assets")}
    for path in paths.values():
        path.mkdir()
    monkeypatch.setenv(runner.RUNS_ROOT_ENV, str(paths["runs"]))
    monkeypatch.setenv(runner.SCENARIOS_ROOT_ENV, str(paths["scenarios"]))
    monkeypatch.setenv(runner.ASSETS_DIR_ENV, str(paths["assets"]))
    return paths


def with_small_indoor(roots: dict[str, Path]) -> Path:
    """Install a copy of small_indoor (bundled scene) in the tmp scenarios root."""
    directory = roots["scenarios"] / "small_indoor"
    directory.mkdir()
    config = directory / "config.yaml"
    config.write_text(SMALL_INDOOR.read_text())
    return config


def open_page(page: str) -> AppTest:
    """Run the console and switch the sidebar to `page`."""
    app = AppTest.from_file(str(CONSOLE), default_timeout=TIMEOUT_S)
    app.run()
    app.sidebar.radio(key="page").set_value(page).run()
    return app


def no_exception(app: AppTest) -> None:
    """Fail with the app's own traceback if a page raised."""
    assert not app.exception, [e.value for e in app.exception]


class FinishedProcess:
    """Stands in for the CLI subprocess: already exited with code 0."""

    returncode = 0

    def poll(self) -> int:
        """Exited."""
        return 0


class RunningProcess:
    """Stands in for a CLI subprocess still in flight."""

    returncode = None

    def poll(self) -> None:
        """Still running."""
        return None


class TestPagesRender:
    """Sprint test 12: every page renders on an empty run history."""

    @pytest.mark.parametrize("page", ["Run", "Scenarios", "History"])
    def test_page_renders_without_exception(
        self, roots: dict[str, Path], page: str
    ) -> None:
        """Empty roots: each page explains there is nothing yet, and no error."""
        app = open_page(page)

        no_exception(app)
        assert app.info  # "no scenarios" / "no runs yet"


class TestRunPage:
    """The Run page builds the request; `runner` turns it into the CLI."""

    def test_run_builds_the_argv_a_user_would_type(
        self, roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sprint test 14: 2 drones, variant B, with the view — as typed by hand."""
        config = with_small_indoor(roots)
        captured: list[runner.RunRequest] = []
        run_dir = roots["runs"] / "fake_run"

        def fake_launch(request: runner.RunRequest, root: Path) -> tuple[Any, Path]:
            captured.append(request)
            assert root == roots["runs"]
            run_dir.mkdir()
            return FinishedProcess(), run_dir

        monkeypatch.setattr(runner, "launch", fake_launch)
        app = open_page("Run")
        app.slider(key="drones::small_indoor").set_value(2)
        app.radio(key="variant::small_indoor").set_value("B")
        app.radio(key="run_mode").set_value("With MuJoCo view")
        app.button(key="run_button").click().run()

        no_exception(app)
        (request,) = captured
        assert runner.build_argv(request, run_dir) == [
            sys.executable,
            "-m",
            "swarm_mapping.cli",
            "--config",
            str(config),
            "--output",
            str(run_dir),
            "--drones",
            "2",
            "--assignment",
            "greedy",
            "--target-tolerance",
            "3",
            "--view",
        ]

    def test_defaults_are_the_scenarios_own_settings(
        self, roots: dict[str, Path]
    ) -> None:
        """All start positions, and the variant the config itself uses (A+B)."""
        with_small_indoor(roots)

        app = open_page("Run")

        no_exception(app)
        assert app.slider(key="drones::small_indoor").value == 3
        assert app.radio(key="variant::small_indoor").value == "A+B"
        assert app.radio(key="run_mode").value == "Headless"

    def test_button_is_disabled_while_a_run_is_in_flight(
        self, roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One run at a time."""
        with_small_indoor(roots)
        run_dir = roots["runs"] / "fake_run"
        run_dir.mkdir()
        monkeypatch.setattr(
            runner, "launch", lambda _request, _root: (RunningProcess(), run_dir)
        )
        app = open_page("Run")
        assert not app.button(key="run_button").disabled

        app.button(key="run_button").click().run()

        no_exception(app)
        assert app.button(key="run_button").disabled

    def test_invalid_scenarios_are_listed_not_offered(
        self, roots: dict[str, Path]
    ) -> None:
        """A broken scenario shows its problem and is not in the selector."""
        with_small_indoor(roots)
        broken = roots["scenarios"] / "broken"
        broken.mkdir()
        (broken / "config.yaml").write_text("drones: [")

        app = open_page("Run")

        no_exception(app)
        assert app.selectbox(key="run_scenario").options == ["small_indoor"]
        assert any("broken" in e.value or "YAML" in e.value for e in app.error)


class TestUpload:
    """The Scenarios page's upload form, through `install_scenario`.

    `AppTest` cannot drive `st.file_uploader`, so the uploader is replaced by a
    stand-in that returns the uploaded bytes by widget key. Everything after
    that — the form, the validator, the messages — is the real console.
    """

    @staticmethod
    def upload(
        monkeypatch: pytest.MonkeyPatch, config_text: str, scene_text: str
    ) -> None:
        """Make the two uploaders return these files."""
        files = {
            "upload_config": io.BytesIO(config_text.encode()),
            "upload_scene": io.BytesIO(scene_text.encode()),
        }
        monkeypatch.setattr(
            streamlit,
            "file_uploader",
            lambda *_args, key, **_kwargs: files[key],
        )

    def test_invalid_upload_shows_problems_and_writes_nothing(
        self, roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sprint test 15: three spawns is refused, with the validator's words."""
        self.upload(
            monkeypatch,
            SMALL_INDOOR.read_text(),  # declares 3 start positions, not 5
            (SMALL_SCENE / "small_indoor.xml").read_text(),
        )
        app = open_page("Scenarios")
        app.text_input(key="upload_name").set_value("my_room")
        app.button[0].click().run()

        no_exception(app)
        messages = [e.value for e in app.error]
        assert any("3 start position(s)" in m for m in messages), messages
        assert list(roots["scenarios"].iterdir()) == []
        assert list(roots["assets"].iterdir()) == []

    def test_valid_upload_is_installed_and_listed(
        self, roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A five-spawn room is installed under both tmp roots and listed."""
        self.upload(
            monkeypatch,
            (REPO_ROOT / "scenarios" / "large_indoor" / "config.yaml").read_text(),
            (SMALL_SCENE / "large_indoor.xml").read_text(),
        )
        app = open_page("Scenarios")
        app.text_input(key="upload_name").set_value("new_room")
        app.button[0].click().run()

        no_exception(app)
        assert (roots["scenarios"] / "new_room" / "config.yaml").is_file()
        assert (roots["assets"] / "new_room.xml").is_file()
        assert any("new_room" in s.value for s in app.success)


class TestHistoryPage:
    """A hand-written run history renders, in detail and compared."""

    def test_runs_table_details_and_compare(self, roots: dict[str, Path]) -> None:
        """Two runs and one broken directory."""
        make_run(roots["runs"], "older", started_at="2026-09-21T09:00:00+00:00")
        make_run(roots["runs"], "newer", started_at="2026-09-21T10:00:00+00:00")
        (roots["runs"] / "crashed").mkdir()

        app = open_page("History")

        no_exception(app)
        table = app.dataframe[0].value
        assert list(table["run"]) == ["newer", "older"]
        assert list(table["failures detected"]) == [1, 1]
        assert any("crashed" in w.value for w in app.warning)
        assert [tab.label for tab in app.tabs] == ["Drone 0", "Drone 1"]
        assert app.selectbox(key="compare_a").value == "newer"
        assert app.selectbox(key="compare_b").value == "older"

    def test_log_filter_narrows_the_log(self, roots: dict[str, Path]) -> None:
        """Filtering on one event name shows only its lines."""
        make_run(roots["runs"], "only")
        app = open_page("History")

        app.multiselect(key="log_events").set_value(["drone_failed"]).run()

        no_exception(app)
        assert any("1 of 5 lines" in c.value for c in app.caption)
