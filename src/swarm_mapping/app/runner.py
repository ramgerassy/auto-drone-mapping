"""Turn an operator's choices into a CLI run, and start it.

The console never runs a mission itself. It builds the exact command a user
would type — `python -m swarm_mapping.cli --config ... --output ...` — and
starts it as a subprocess (sprint plan, cross-cutting constraint 7). So there is
one way a mission executes, a console run is byte-identical to the same CLI run,
the MuJoCo viewer gets its own process, and the UI stays responsive.

Pure Python, no streamlit: everything the console decides is here, where it can
be tested without a browser.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from swarm_mapping.app.validation import validate_scenario
from swarm_mapping.config.loader import load_config
from swarm_mapping.records import VARIANTS

RUNS_ROOT_ENV = "SWARM_RUNS_ROOT"
"""Overrides the directory holding one sub-directory per console run."""

SCENARIOS_ROOT_ENV = "SWARM_SCENARIOS_ROOT"
"""Overrides the directory holding one sub-directory per scenario."""

ASSETS_DIR_ENV = "SWARM_ASSETS_DIR"
"""Overrides where uploaded scenes are installed."""

CONSOLE_FILE = "console.txt"
"""The subprocess's stdout and stderr, inside its run directory."""

# The scenes that relative `scene.path` values resolve against (`cli`'s
# `_ASSETS_DIR`). Recomputed here rather than imported: it is private to cli.
_BUNDLED_ASSETS = Path(__file__).resolve().parents[1] / "simulation" / "assets"


def runs_root() -> Path:
    """Where console runs are written: `$SWARM_RUNS_ROOT`, else `runs/`.

    Relative to the working directory, which is the repo root in normal use.
    """
    return Path(os.environ.get(RUNS_ROOT_ENV, "runs"))


def scenarios_root() -> Path:
    """Where scenarios are listed from: `$SWARM_SCENARIOS_ROOT`, else `scenarios/`."""
    return Path(os.environ.get(SCENARIOS_ROOT_ENV, "scenarios"))


def assets_dir() -> Path:
    """Where uploaded scenes go: `$SWARM_ASSETS_DIR`, else the bundled assets."""
    override = os.environ.get(ASSETS_DIR_ENV)
    return Path(override) if override else _BUNDLED_ASSETS


@dataclass(frozen=True)
class RunRequest:
    """One run, as the operator chose it.

    Attributes:
        config_path: The scenario YAML, passed to `--config` as given.
        scenario: The scenario's name, for the run directory's name.
        drones: Swarm size, passed to `--drones`.
        variant: Allocation variant label, a key of `records.VARIANTS`.
        view: Open the live MuJoCo viewer (`--view`).
        min_drones: The fewest drones this scenario can run with — see
            `ScenarioChoice.min_drones`. Defaults to 1 (no failure schedule),
            so callers outside the console need not set it.
    """

    config_path: Path
    scenario: str
    drones: int
    variant: str
    view: bool
    min_drones: int = 1

    def __post_init__(self) -> None:
        """Reject a request the CLI could not run as labelled."""
        if self.variant not in VARIANTS:
            msg = f"unknown variant {self.variant!r}; expected one of {list(VARIANTS)}"
            raise ValueError(msg)
        if self.drones < 1:
            msg = f"drones must be at least 1, got {self.drones}"
            raise ValueError(msg)
        if self.drones < self.min_drones:
            msg = (
                f"drones must be at least {self.min_drones} for {self.scenario!r} "
                f"(its failure schedule names a drone that would be missing), "
                f"got {self.drones}"
            )
            raise ValueError(msg)


def build_argv(request: RunRequest, output_dir: Path) -> list[str]:
    """The CLI command for a request — what a user would type.

    The variant is always passed explicitly, even when it matches the config:
    the run then does what its label says whatever the file currently holds.

    Args:
        request: The run.
        output_dir: The run's directory, passed to `--output`.

    Returns:
        The argument list, starting with this interpreter.
    """
    assignment, tolerance = VARIANTS[request.variant]
    argv = [
        sys.executable,
        "-m",
        "swarm_mapping.cli",
        "--config",
        str(request.config_path),
        "--output",
        str(output_dir),
        "--drones",
        str(request.drones),
        "--assignment",
        assignment,
        "--target-tolerance",
        str(tolerance),
    ]
    if request.view:
        argv.append("--view")
    return argv


def run_dir_name(request: RunRequest, now: datetime) -> str:
    """Name a run's directory: `YYYYmmdd-HHMMSS_<scenario>_<variant>_<N>d`.

    The wall-clock time is only ever part of this name — never a mission
    input — so it cannot perturb a run. `+` is dropped from the variant
    (`A+B` → `AB`) to keep the name shell-friendly.

    Args:
        request: The run.
        now: The time to stamp.

    Returns:
        The directory name, without a collision suffix.
    """
    variant = request.variant.replace("+", "")
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{request.scenario}_{variant}_{request.drones}d"


def new_run_dir(root: Path, request: RunRequest, now: datetime) -> Path:
    """Create a fresh directory for a run, never reusing an existing one.

    Two runs started in the same second get `-2`, `-3`, ... Creating with
    `exist_ok=False` makes the existence check and the claim a single step.

    Args:
        root: The runs root; created if missing.
        request: The run.
        now: The time to stamp.

    Returns:
        The created directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    base = run_dir_name(request, now)
    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def launch(
    request: RunRequest, root: Path, now: datetime | None = None
) -> tuple[subprocess.Popen[bytes], Path]:
    """Start a run as a CLI subprocess in a new run directory.

    Args:
        request: The run.
        root: The runs root.
        now: The time for the directory name; the current local time if None.

    Returns:
        The running process and its run directory. stdout and stderr go to
        `<run dir>/console.txt`.
    """
    run_dir = new_run_dir(root, request, now or datetime.now())
    with (run_dir / CONSOLE_FILE).open("wb") as console:
        # Popen duplicates the handle into the child, so closing ours when the
        # `with` exits does not cut the child's output off.
        process = subprocess.Popen(
            build_argv(request, run_dir),
            stdout=console,
            stderr=subprocess.STDOUT,
        )
    return process, run_dir


@dataclass(frozen=True)
class ScenarioChoice:
    """A scenario on disk and whether it can run.

    Attributes:
        name: The scenario's directory name.
        config_path: Its `config.yaml`.
        spawns: Start positions it declares; None if the config does not load.
        min_drones: The fewest drones a run can use. 1, unless the config has
            a `failures:` schedule, in which case it is one more than the
            highest drone id the schedule names — a run with fewer drones
            than that would drop a scheduled drone, and `FailureInjector`
            raises `ValueError` at startup rather than run with a schedule it
            cannot honour.
        problems: `validate_scenario`'s findings under the default rules;
            empty means it can be run.
    """

    name: str
    config_path: Path
    spawns: int | None
    min_drones: int
    problems: list[str]


def _min_drones(config_path: Path) -> int:
    """The fewest drones this config's failure schedule needs, else 1.

    Args:
        config_path: The scenario YAML.

    Returns:
        1 if the config does not load or declares no failures; otherwise one
        more than the highest `drone_id` any scheduled failure names.
    """
    try:
        failures = load_config(config_path).failures
    except Exception:  # already reported in `problems`, in the validator's words
        return 1
    if not failures:
        return 1
    return max(failure.drone_id for failure in failures) + 1


def scenario_choices(root: Path) -> list[ScenarioChoice]:
    """Every `<root>/*/config.yaml`, validated, sorted by name.

    Args:
        root: The scenarios root. Need not exist.

    Returns:
        One entry per scenario directory holding a `config.yaml`.
    """
    if not root.is_dir():
        return []
    choices = []
    for config_path in sorted(root.glob("*/config.yaml")):
        problems = validate_scenario(config_path)
        try:
            spawns: int | None = load_config(config_path).drones.count
        except Exception:  # already reported in `problems`, in the validator's words
            spawns = None
        choices.append(
            ScenarioChoice(
                name=config_path.parent.name,
                config_path=config_path,
                spawns=spawns,
                min_drones=_min_drones(config_path),
                problems=problems,
            )
        )
    return choices
