"""Tests for `swarm-console`: it starts streamlit, or says how to install it.

Never starts streamlit: `subprocess.call` is replaced to capture the command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from swarm_mapping.app import launch

pytestmark = pytest.mark.sprint(3)


class TestLauncher:
    """`swarm-console` without the extra says how to get it."""

    def test_missing_streamlit_prints_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without the extra: the hint on stderr, exit code 1."""
        monkeypatch.setattr(launch.importlib.util, "find_spec", lambda _name: None)

        with pytest.raises(SystemExit) as exit_info:
            launch.main()

        assert exit_info.value.code == 1
        assert "install the console with: uv sync --extra ui" in capsys.readouterr().err

    def test_runs_streamlit_on_the_console_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the extra: `streamlit run console.py`, extra args passed on."""
        calls: list[list[str]] = []
        monkeypatch.setattr(launch.importlib.util, "find_spec", lambda _name: object())
        monkeypatch.setattr(
            launch.subprocess, "call", lambda argv: calls.append(argv) or 0
        )
        monkeypatch.setattr(sys, "argv", ["swarm-console", "--server.port", "9000"])

        with pytest.raises(SystemExit) as exit_info:
            launch.main()

        assert exit_info.value.code == 0
        (argv,) = calls
        assert argv[:4] == [sys.executable, "-m", "streamlit", "run"]
        assert Path(argv[4]).name == "console.py"
        assert Path(argv[4]).is_file()
        assert argv[5:] == ["--server.port", "9000"]
