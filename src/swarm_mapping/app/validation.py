"""Scenario validation: can this config and scene run, for any swarm size?

The rules are the sprint plan's D2 reading of "can spawn up to 5 drones": the
config declares its start positions and **every one is usable** — inside the
grid, clear of the scene's geometry, and far enough from the others — so any
run from one drone up to all of them works.

Each rule is checked by the component that already owns it rather than
re-implemented here: `config` parses and bounds-checks, MuJoCo decides what
touches what, and `build_mission` (via `CentralizedMaster`) enforces spawn
separation. A second copy of any of those checks would be a second thing to
keep in step, and the copy would drift.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from swarm_mapping.cli import build_mission, resolve_scene_path
from swarm_mapping.config.loader import load_config
from swarm_mapping.simulation.engine import SimulationEngine

UPLOAD_DRONES = 5
"""Start positions an uploaded room must declare: "up to 5 drones" means all
five usable (CLAUDE.md scopes the system at 1-5 drones)."""


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
