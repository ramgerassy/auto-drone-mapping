"""Ground-truth localizer that reads poses directly from MuJoCo state."""

from __future__ import annotations

import mujoco

from swarm_mapping.simulation.types import Pose


class GroundTruthLocalizer:
    """Reads drone poses from MuJoCo's computed body positions.

    This is the only Localizer implementation. It returns exact poses
    from the simulator state — no estimation, no noise. A SLAM-based
    localizer would be a second implementation of the Localizer
    protocol, but that is out of scope.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data (must stay in sync with model).
        drone_body_names: Mapping from drone_id to MuJoCo body name.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        drone_body_names: dict[int, str],
    ) -> None:
        self._model = model
        self._data = data
        self._body_ids: dict[int, int] = {}

        for drone_id, body_name in drone_body_names.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                msg = f"Body '{body_name}' for drone {drone_id} not found in model"
                raise KeyError(msg)
            self._body_ids[drone_id] = body_id

    def get_pose(self, drone_id: int) -> Pose:
        """Return the current pose of the given drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            The drone's current pose in world coordinates.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        if drone_id not in self._body_ids:
            msg = f"Unknown drone_id: {drone_id}"
            raise KeyError(msg)

        body_id = self._body_ids[drone_id]
        position = self._data.xpos[body_id].copy()
        quaternion = self._data.xquat[body_id].copy()

        return Pose(position=position, quaternion=quaternion)

    def get_body_id(self, drone_id: int) -> int:
        """Return the MuJoCo body ID for a drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            The MuJoCo body ID.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        if drone_id not in self._body_ids:
            msg = f"Unknown drone_id: {drone_id}"
            raise KeyError(msg)
        return self._body_ids[drone_id]
