"""Scenario validation: can this config and scene run, for any swarm size?

The rules are the sprint plan's D2 reading of "can spawn up to 5 drones": the
config declares its start positions and **every one is usable** — inside the
grid, clear of the scene's geometry, and far enough from the others — so any
drone count that includes every drone the failure schedule names works.

Each rule is checked by the component that already owns it rather than
re-implemented here: `config` parses and bounds-checks, MuJoCo decides what
touches what, and `build_mission` (via `CentralizedMaster`) enforces spawn
separation. A second copy of any of those checks would be a second thing to
keep in step, and the copy would drift.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from swarm_mapping.cli import build_mission, resolve_scene_path
from swarm_mapping.config.loader import load_config
from swarm_mapping.simulation.engine import SimulationEngine

UPLOAD_DRONES = 5
"""Start positions an uploaded room must declare: "up to 5 drones" means all
five usable (CLAUDE.md scopes the system at 1-5 drones)."""

SCENARIO_NAME = re.compile(r"[a-z][a-z0-9_]{0,40}")
"""What an uploaded room may be called (matched in full). No `/`, no `.`: the
name becomes a directory and a file name, so it must not be able to climb out
of either root or pick up a second extension."""

# Placeholder for the staging directory in problems reported back to the user,
# who never saw that temporary path.
_UPLOAD = "<upload>"


def validate_scenario(
    config_path: str | Path, *, require_five: bool = False
) -> list[str]:
    """List everything that stops a scenario from running.

    Checks run in order, and stop where a later check would be meaningless: a
    config that does not parse has no scene to look for, and a missing scene is
    reported before MuJoCo is touched.

    Args:
        config_path: The scenario YAML.
        require_five: Also require exactly `UPLOAD_DRONES` start positions.
            Off by default because two shipped scenarios (`small_indoor`,
            `comb_indoor`) deliberately declare three; on for uploads.

    Returns:
        Human-readable problems, each naming what is wrong. Empty means valid.
        Never raises on bad input — a broken upload is a list of reasons, not a
        crash.
    """
    path = Path(config_path)
    try:
        config = load_config(path)
    except OSError as exc:
        return [f"cannot read config {path}: {exc}"]
    except yaml.YAMLError as exc:
        return [f"{path}: not valid YAML: {exc}"]
    except ValueError as exc:
        return [str(exc)]  # already names the file and the key

    scene = resolve_scene_path(config.scene_path)
    if not scene.is_file():
        return [f"scene.path '{config.scene_path}' does not exist (looked for {scene})"]

    problems: list[str] = []
    count = config.drones.count
    if require_five and count != UPLOAD_DRONES:
        problems.append(
            f"drones.start_positions declares {count} start position(s); an "
            f"uploaded room must declare exactly {UPLOAD_DRONES}, so that every "
            f"run of 1-{UPLOAD_DRONES} drones is valid"
        )

    positions = config.drones.start_positions
    try:
        engine = SimulationEngine(
            scene,
            {
                index: np.array(position, dtype=np.float64)
                for index, position in enumerate(positions)
            },
        )
    except Exception as exc:  # MuJoCo raises several types on bad XML
        return [*problems, f"scene {scene} failed to load: {exc}"]
    problems.extend(_spawns_in_geometry(engine, positions))

    # Every declared spawn at once, so separation is checked between all of
    # them — not just the default swarm's.
    try:
        build_mission(config, drones=count)
    except Exception as exc:  # the contract: problems, never raises
        problems.append(str(exc))
    return problems


def install_scenario(
    name: str,
    config_text: str,
    scene_text: str,
    scenarios_root: Path,
    assets_dir: Path,
) -> list[str]:
    """Validate an uploaded room and, only if it is clean, install it.

    The room is staged in a temporary directory and validated there with the
    upload rules (`require_five=True`); nothing is written under either root
    unless validation passes. Nothing is ever overwritten: a name already used
    by a scenario directory or a scene file is refused, and the final writes
    use exclusive creation so even a race cannot replace a file.

    Installed exactly like a shipped scenario: the scene goes to
    `assets_dir/<name>.xml` and the config to `scenarios_root/<name>/config.yaml`
    with `scene.path: <name>.xml`, which resolves under the bundled assets.
    The config is rewritten through YAML, so comments in the upload are not
    kept.

    Args:
        name: The room's name; must match `SCENARIO_NAME`.
        config_text: The uploaded scenario YAML.
        scene_text: The uploaded MJCF scene.
        scenarios_root: Directory holding one sub-directory per scenario.
        assets_dir: Directory holding the scene files that relative
            `scene.path` values resolve against.

    Returns:
        Problems that stopped the install. Empty means it was installed.
    """
    if not SCENARIO_NAME.fullmatch(name):
        return [
            f"scenario name {name!r} is not allowed: use 1-41 characters, a "
            f"lowercase letter first, then lowercase letters, digits or '_'"
        ]
    scenario_dir = scenarios_root / name
    scene_file = assets_dir / f"{name}.xml"
    clashes = [
        f"{path} already exists; installing never overwrites"
        for path in (scenario_dir, scene_file)
        if path.exists()
    ]
    if clashes:
        return clashes

    raw = _scenario_document(config_text)
    with tempfile.TemporaryDirectory(prefix="swarm_upload_") as staging_name:
        staging = Path(staging_name)
        staged_scene = staging / f"{name}.xml"
        staged_scene.write_text(scene_text)
        staged_config = staging / "config.yaml"
        if raw is None:
            # Unusable as a scenario document: stage it as uploaded and let
            # the validator say what is wrong, in the same words it would use
            # for a file on disk.
            staged_config.write_text(config_text)
        else:
            raw["scene"]["path"] = str(staged_scene)
            staged_config.write_text(yaml.safe_dump(raw, sort_keys=False))
        problems = validate_scenario(staged_config, require_five=True)
    if problems:
        return [problem.replace(str(staging), _UPLOAD) for problem in problems]
    assert raw is not None  # a document without a scene section cannot validate

    raw["scene"]["path"] = scene_file.name
    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        scenarios_root.mkdir(parents=True, exist_ok=True)
        with scene_file.open("x") as handle:
            handle.write(scene_text)
    except FileExistsError as exc:
        return [f"{exc.filename} already exists; installing never overwrites"]
    created_dir = False
    try:
        scenario_dir.mkdir()
        created_dir = True
        with (scenario_dir / "config.yaml").open("x") as handle:
            handle.write(yaml.safe_dump(raw, sort_keys=False))
    except BaseException as exc:
        # Never leave half a room behind: a scene file with no scenario would
        # block the name forever and list nowhere. Only what this call created
        # is removed — the scene was written above with exclusive create, and
        # the directory only if our own mkdir made it.
        scene_file.unlink(missing_ok=True)
        if created_dir:
            for leftover in scenario_dir.iterdir():
                leftover.unlink()
            scenario_dir.rmdir()
        if isinstance(exc, FileExistsError):
            return [f"{exc.filename} already exists; installing never overwrites"]
        raise
    return []


def _scenario_document(config_text: str) -> dict[str, Any] | None:
    """Parse an uploaded config far enough to rewrite its `scene.path`.

    Returns:
        The document, or None if it is not YAML, not a mapping, or has no
        `scene` mapping — the validator then reports why.
    """
    try:
        raw = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("scene"), dict):
        return None
    return raw


def _spawns_in_geometry(
    engine: SimulationEngine,
    positions: tuple[tuple[float, float, float], ...],
) -> list[str]:
    """Name every start position that intersects the scene's geometry.

    Asks MuJoCo rather than reading the XML: a geom's world position depends
    on its parent bodies' frames, default classes and includes, which only the
    compiled model resolves. The engine's constructor runs `mj_forward`, which
    fills `data.contact` with every overlapping pair.

    Drone-to-drone contacts are ignored here — spacing is `build_mission`'s
    check, against the configured `min_separation` rather than bare overlap.
    Geoms with `contype=0` and `conaffinity=0` never produce contacts, so a
    spawn inside one is not seen here (no shipped scene has one).

    Args:
        engine: An engine built with one drone per start position, drone id =
            spawn index.
        positions: The start positions, for the messages.

    Returns:
        One problem per offending spawn, in spawn order.
    """
    model, data = engine.model, engine.data
    spawn_of_body = {engine.get_body_id(index): index for index in engine.drone_ids}
    touching: dict[int, set[str]] = {}
    for contact in data.contact[: data.ncon]:
        pair = (int(contact.geom1), int(contact.geom2))
        for geom, other in (pair, pair[::-1]):
            spawn = spawn_of_body.get(int(model.geom_bodyid[geom]))
            other_is_drone = int(model.geom_bodyid[other]) in spawn_of_body
            if spawn is None or other_is_drone:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            touching.setdefault(spawn, set()).add(name or f"geom #{other}")

    problems = []
    for spawn in sorted(touching):
        x, y, z = positions[spawn]
        # Names sorted so the message is identical run to run.
        geoms = ", ".join(sorted(touching[spawn]))
        problems.append(
            f"drones.start_positions[{spawn}] at ({x}, {y}, {z}) is inside or "
            f"touching scene geometry: {geoms}"
        )
    return problems
