"""Main simulation engine wrapping MuJoCo."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from swarm_mapping.simulation.localizer import GroundTruthLocalizer
from swarm_mapping.simulation.raycaster import MjRayCaster
from swarm_mapping.simulation.types import Pose, RayHit

# Drone colors cycle for visual distinction in multi-drone scenarios
_DRONE_COLORS = [
    "0.2 0.6 1.0 1.0",  # blue
    "1.0 0.4 0.2 1.0",  # orange
    "0.2 0.8 0.4 1.0",  # green
    "0.8 0.2 0.8 1.0",  # purple
    "1.0 0.8 0.2 1.0",  # yellow
]


def _build_drone_xml(drone_id: int, position: NDArray[np.float64]) -> str:
    """Generate MJCF XML fragment for a single drone body.

    Args:
        drone_id: Integer identifier used in naming.
        position: Initial (x, y, z) position, shape (3,).

    Returns:
        XML string for the drone body element.
    """
    name = f"drone_{drone_id}"
    color = _DRONE_COLORS[drone_id % len(_DRONE_COLORS)]
    pos = f"{position[0]} {position[1]} {position[2]}"

    return f"""\
    <body name="{name}" pos="{pos}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_body" type="box" size="0.15 0.15 0.05"
            mass="0.5" rgba="{color}"/>
      <geom name="{name}_rotor_fl" type="cylinder" size="0.08 0.01"
            pos="0.15 0.15 0.05" rgba="0.3 0.3 0.3 0.5"
            contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_fr" type="cylinder" size="0.08 0.01"
            pos="0.15 -0.15 0.05" rgba="0.3 0.3 0.3 0.5"
            contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_bl" type="cylinder" size="0.08 0.01"
            pos="-0.15 0.15 0.05" rgba="0.3 0.3 0.3 0.5"
            contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_br" type="cylinder" size="0.08 0.01"
            pos="-0.15 -0.15 0.05" rgba="0.3 0.3 0.3 0.5"
            contype="0" conaffinity="0"/>
    </body>"""


def _inject_drones(
    scene_xml: str,
    drone_positions: dict[int, NDArray[np.float64]],
) -> str:
    """Inject drone bodies into a scene XML string.

    Args:
        scene_xml: The environment MJCF XML (no drones).
        drone_positions: Mapping from drone_id to initial position.

    Returns:
        Complete MJCF XML string with drones injected.

    Raises:
        ValueError: If the scene XML has no </worldbody> tag.
    """
    drone_fragments = [
        _build_drone_xml(drone_id, pos)
        for drone_id, pos in sorted(drone_positions.items())
    ]
    injection = "\n\n".join(drone_fragments)

    if "</worldbody>" not in scene_xml:
        msg = "Scene XML missing </worldbody> tag"
        raise ValueError(msg)

    return scene_xml.replace("</worldbody>", f"\n{injection}\n  </worldbody>")


class SimulationEngine:
    """High-level wrapper around a MuJoCo simulation.

    Loads a scene XML, injects drone bodies at specified positions,
    and provides access to drone poses and ray-casting through the
    Localizer and RayCaster interfaces.

    Args:
        scene_path: Path to the MJCF scene XML file (environment only,
            no drone bodies — they are injected programmatically).
        drone_positions: Mapping from drone_id to initial (x, y, z)
            position. The number of entries determines drone count.
    """

    def __init__(
        self,
        scene_path: str | Path,
        drone_positions: dict[int, NDArray[np.float64]],
    ) -> None:
        scene_xml = Path(scene_path).read_text()
        full_xml = _inject_drones(scene_xml, drone_positions)

        self._model = mujoco.MjModel.from_xml_string(full_xml)
        self._data = mujoco.MjData(self._model)

        # Build drone name mapping from the IDs
        drone_names = {drone_id: f"drone_{drone_id}" for drone_id in drone_positions}

        self._localizer = GroundTruthLocalizer(self._model, self._data, drone_names)
        self._raycaster = MjRayCaster(self._model, self._data)

        # Cache joint info for position control
        self._joint_qpos_adr: dict[int, int] = {}
        for drone_id, body_name in drone_names.items():
            joint_name = f"{body_name}_joint"
            joint_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id == -1:
                msg = f"Joint '{joint_name}' for drone {drone_id} not found in model"
                raise KeyError(msg)
            self._joint_qpos_adr[drone_id] = int(self._model.jnt_qposadr[joint_id])

        # Store initial qpos for reset
        self._initial_qpos = self._data.qpos.copy()

        # Compute forward kinematics for initial state
        mujoco.mj_forward(self._model, self._data)

    @property
    def model(self) -> mujoco.MjModel:
        """The MuJoCo model."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """The MuJoCo data."""
        return self._data

    @property
    def timestep(self) -> float:
        """Simulation timestep in seconds."""
        return float(self._model.opt.timestep)

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return float(self._data.time)

    def reset(self) -> None:
        """Reset simulation to initial state.

        Restores qpos to initial values, zeros qvel, and recomputes
        forward kinematics. Simulation time is reset to zero.
        """
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = self._initial_qpos
        mujoco.mj_forward(self._model, self._data)

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self._model, self._data)

    def get_pose(self, drone_id: int) -> Pose:
        """Return the current pose of a drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            The drone's current pose in world coordinates.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        return self._localizer.get_pose(drone_id)

    def get_body_id(self, drone_id: int) -> int:
        """Return the MuJoCo body ID for a drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            The MuJoCo body ID.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        return self._localizer.get_body_id(drone_id)

    def cast_rays(
        self,
        drone_id: int,
        directions: NDArray[np.float64],
    ) -> list[RayHit | None]:
        """Cast rays from a drone's position.

        Args:
            drone_id: Integer identifier for the drone.
            directions: Array of unit direction vectors, shape (N, 3).

        Returns:
            List of RayHit for hits, None for misses.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        pose = self._localizer.get_pose(drone_id)
        body_id = self._localizer.get_body_id(drone_id)
        return self._raycaster.cast_rays(pose.position, directions, body_id)

    def set_drone_position(
        self,
        drone_id: int,
        position: NDArray[np.float64],
    ) -> None:
        """Teleport a drone to a new position.

        Directly sets qpos for the drone's freejoint and recomputes
        forward kinematics. Zeros velocity to prevent drift. This is
        the Sprint 1 movement model — no physics-based control.

        Args:
            drone_id: Integer identifier for the drone.
            position: Target position as (x, y, z), shape (3,).

        Raises:
            KeyError: If drone_id is not recognized.
        """
        if drone_id not in self._joint_qpos_adr:
            msg = f"Unknown drone_id: {drone_id}"
            raise KeyError(msg)

        qpos_adr = self._joint_qpos_adr[drone_id]

        # Set position (first 3 of 7 qpos values for freejoint)
        self._data.qpos[qpos_adr : qpos_adr + 3] = position

        # Reset quaternion to identity (upright)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]

        # Zero out velocity to prevent drift
        self._data.qvel[:] = 0

        # Recompute derived quantities
        mujoco.mj_forward(self._model, self._data)
