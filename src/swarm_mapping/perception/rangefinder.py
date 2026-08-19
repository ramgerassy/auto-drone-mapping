"""Horizontal-plane rangefinder sensor implementation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.perception.types import RayObservation, ScanResult
from swarm_mapping.simulation.engine import (
    DRONE_EXCLUSION_RADIUS,
    SimulationEngine,
)


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
        exclusion_radius: Radius in metres around a teammate's centre within
            which a hit is attributed to that teammate rather than to the
            environment, and re-encoded as a MISS. Pass 0.0 to disable
            filtering entirely.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        num_rays: int = 36,
        max_range: float = 10.0,
        angular_range: float = 2 * math.pi,
        exclusion_radius: float = DRONE_EXCLUSION_RADIUS,
    ) -> None:
        self._engine = engine
        self._num_rays = num_rays
        self._max_range = max_range
        self._angular_range = angular_range
        # Compared against squared distances, so the sqrt never runs.
        self._exclusion_radius_sq = exclusion_radius * exclusion_radius

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

        Hits landing on a teammate are re-encoded as misses with the range
        shortened to the teammate's distance — see `_is_teammate`.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            A ScanResult containing all ray observations.
        """
        pose = self._engine.get_pose(drone_id)
        teammates = self._teammate_positions(drone_id)

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
                if self._is_teammate(hit.hit_point, teammates):
                    # A teammate is not part of the map. Emit a MISS that
                    # stops where the teammate is: the ray genuinely travelled
                    # that far unobstructed, so those cells are free, but
                    # nothing is claimed at or beyond the teammate. That keeps
                    # the occlusion shadow a real LiDAR would also have.
                    obs = RayObservation(
                        origin=pose.position.copy(),
                        direction=direction.copy(),
                        max_range=hit.distance,
                        distance=None,
                        hit_point=None,
                    )
                else:
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

    def _teammate_positions(self, drone_id: int) -> list[NDArray[np.float64]]:
        """World positions of every drone in the swarm except this one.

        Ground-truth poses stand in for the shared telemetry a real swarm would
        broadcast, so the radius below absorbs body extent only — never
        localization error.

        Args:
            drone_id: The sensing drone, excluded from the result.

        Returns:
            Positions in ascending drone-id order.
        """
        return [
            self._engine.get_pose(other).position
            for other in sorted(self._engine.drone_ids)
            if other != drone_id
        ]

    def _is_teammate(
        self,
        hit_point: NDArray[np.float64],
        teammates: list[NDArray[np.float64]],
    ) -> bool:
        """Whether a hit point falls on one of the teammates.

        A radial test in 3D. 3D rather than 2D costs nothing while every drone
        shares one altitude, and stays correct if they are ever separated
        vertically — where a 2D test would discard a wall merely sharing an
        (x, y) with a drone flying above it.

        Iteration is in sorted id order for legibility only; this is a boolean
        any-match, so order cannot change the result.

        Args:
            hit_point: The world-coordinate hit point.
            teammates: Teammate positions from `_teammate_positions`.

        Returns:
            True if the hit should be discarded as a teammate.
        """
        for position in teammates:
            delta = hit_point - position
            if float(delta @ delta) < self._exclusion_radius_sq:
                return True
        return False
