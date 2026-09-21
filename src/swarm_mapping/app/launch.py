"""The `swarm-console` entry point: `streamlit run` on the console script.

Streamlit is the optional `ui` extra, so it is looked up here rather than
imported at module level: without it, the command says how to install it
instead of failing with a traceback.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = "install the console with: uv sync --extra ui"
"""Printed when the `ui` extra is missing."""

_CONSOLE = Path(__file__).with_name("console.py")


def main() -> None:
    """Start the console; extra arguments go to `streamlit run`.

    Raises:
        SystemExit: Always — with 1 if streamlit is missing, else with
            streamlit's exit code.
    """
    if importlib.util.find_spec("streamlit") is None:
        print(INSTALL_HINT, file=sys.stderr)
        sys.exit(1)
    argv = [sys.executable, "-m", "streamlit", "run", str(_CONSOLE), *sys.argv[1:]]
    sys.exit(subprocess.call(argv))
