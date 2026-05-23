"""Main simulation engine wrapping MuJoCo."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from swarm_mapping.simulation.localizer import GroundTruthLocalizer
from swarm_mapping.simulation.raycaster import MjRayCaster
from swarm_mapping.simulation.types import Pose, RayHit


class SimulationEngine:
    """High-level wrapper around a MuJoCo simulation.

    Manages the MuJoCo model and data, provides access to drone poses
    and ray-casting through the Localizer and RayCaster interfaces,
    and supports position-controlled drone movement for Sprint 1.

    Args:
        scene_path: Path to the MJCF scene XML file.
        drone_names: Mapping from drone_id to MuJoCo body name.
    """

    def __init__(
        self,
        scene_path: str | Path,
        drone_names: dict[int, str],
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)
        self._drone_names = dict(drone_names)

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
