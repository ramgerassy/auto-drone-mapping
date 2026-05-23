"""Horizontal-plane rangefinder sensor implementation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.perception.types import RayObservation, ScanResult
from swarm_mapping.simulation.engine import SimulationEngine


def _rotate_vectors_by_quaternion(
    vectors: NDArray[np.float64],
    quat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate vectors by a quaternion using q * v * q_conj.

    Args:
        vectors: Array of 3D vectors, shape (N, 3).
        quat: Unit quaternion (w, x, y, z), shape (4,).
            Follows MuJoCo convention where w is first.

    Returns:
        Rotated vectors, shape (N, 3).
    """
    w, x, y, z = quat

    # Rotation matrix from quaternion (avoids per-vector quaternion multiply)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    rotation_matrix = np.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ]
    )

    return (rotation_matrix @ vectors.T).T


class Rangefinder:
    """Horizontal-plane rangefinder sensor.

    Casts evenly-spaced rays around the drone's horizontal plane,
    converting raw ray-cast results into RayObservation objects that
    the mapper can consume.

    Args:
        engine: The simulation engine providing poses and ray-casting.
        num_rays: Number of rays per scan. Default 36 (10° spacing).
        max_range: Maximum sensor range in meters. Default 10.0.
        angular_range: Angular coverage in radians. Default 2π
            (full 360° sweep). Rays are centered on the drone's
            forward (+x) direction.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        num_rays: int = 36,
        max_range: float = 10.0,
        angular_range: float = 2 * math.pi,
    ) -> None:
        self._engine = engine
        self._num_rays = num_rays
        self._max_range = max_range
        self._angular_range = angular_range

        # Pre-compute body-frame ray directions (XY plane)
        self._body_directions = self._compute_directions()

    def _compute_directions(self) -> NDArray[np.float64]:
        """Compute evenly-spaced ray directions in the body frame.

        Returns:
            Array of unit direction vectors, shape (num_rays, 3).
            Rays are in the XY plane, centered on +x axis.
        """
        start_angle = -self._angular_range / 2
        angles = np.linspace(
            start_angle,
            start_angle + self._angular_range,
            self._num_rays,
            endpoint=False,
        )

        directions = np.zeros((self._num_rays, 3))
        directions[:, 0] = np.cos(angles)
        directions[:, 1] = np.sin(angles)
        # z = 0: rays are horizontal

        return directions

    def scan(self, drone_id: int) -> ScanResult:
        """Perform a rangefinder scan for the given drone.

        Casts rays from the drone's current position in evenly-spaced
        directions (rotated by the drone's orientation), and packages
        the results as RayObservation objects.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            A ScanResult containing all ray observations.
        """
        pose = self._engine.get_pose(drone_id)

        # Rotate body-frame directions to world frame
        world_directions = _rotate_vectors_by_quaternion(
            self._body_directions, pose.quaternion
        )

        # Cast rays through the engine (handles body exclusion)
        hits = self._engine.cast_rays(drone_id, world_directions)

        # Build observations
        observations: list[RayObservation] = []
        for i, hit in enumerate(hits):
            direction = world_directions[i]

            if hit is not None and hit.distance <= self._max_range:
                obs = RayObservation(
                    origin=pose.position.copy(),
                    direction=direction.copy(),
                    max_range=self._max_range,
                    distance=hit.distance,
                    hit_point=hit.hit_point.copy(),
                )
            else:
                # Miss or beyond max range
                obs = RayObservation(
                    origin=pose.position.copy(),
                    direction=direction.copy(),
                    max_range=self._max_range,
                    distance=None,
                    hit_point=None,
                )

            observations.append(obs)

        return ScanResult(
            drone_id=drone_id,
            pose=pose,
            observations=observations,
            timestamp=self._engine.time,
        )
