"""Protocol definitions for simulation module interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.simulation.types import Pose, RayHit


class Localizer(Protocol):
    """Interface for obtaining drone poses.

    The only implementation is GroundTruthLocalizer (reads from MuJoCo
    state). A SLAM-based localizer would be a second implementation,
    but that is explicitly out of scope.
    """

    def get_pose(self, drone_id: int) -> Pose:
        """Return the current pose of the given drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            The drone's current pose in world coordinates.

        Raises:
            KeyError: If drone_id is not recognized.
        """
        ...


class RayCaster(Protocol):
    """Interface for casting sensor rays into the environment.

    The only implementation is MjRayCaster (wraps mujoco.mj_ray).
    A depth-camera or stereo-pair sensor would be a second
    implementation.
    """

    def cast_rays(
        self,
        origin: NDArray[np.float64],
        directions: NDArray[np.float64],
        body_exclude: int,
    ) -> list[RayHit | None]:
        """Cast rays from a point and return hit results.

        Args:
            origin: The 3D point to cast from, shape (3,).
            directions: Array of unit direction vectors, shape (N, 3).
            body_exclude: MuJoCo body ID to exclude from hits (the
                drone's own body, so rays don't self-intersect).

        Returns:
            List of length N. Each element is a RayHit if the ray hit
            a geometry, or None if it missed everything.
        """
        ...
