"""Horizontal-plane rangefinder sensor implementation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.perception.types import RayObservation, ScanResult
from swarm_mapping.simulation.engine import (
    DRONE_EXCLUSION_RADIUS,
    DRONE_HALF_EXTENT,
    SimulationEngine,
)
from swarm_mapping.simulation.types import RayHit


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
            environment, and re-encoded as a MISS. Must be 0.0 (filtering
            disabled) or at least the body's corner radius; anything between
            would let corner-on hits through and re-admit the very bug this
            filter exists to fix.

    Raises:
        ValueError: If exclusion_radius is negative, or is positive but below
            the body corner radius.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        num_rays: int = 36,
        max_range: float = 10.0,
        angular_range: float = 2 * math.pi,
        exclusion_radius: float = DRONE_EXCLUSION_RADIUS,
    ) -> None:
        # Squaring destroys the sign, so a negative radius would silently
        # behave like its absolute value. Reject it before that can happen.
        corner_radius = DRONE_HALF_EXTENT * math.sqrt(2)
        if exclusion_radius < 0.0:
            msg = f"exclusion_radius must be >= 0, got {exclusion_radius}"
            raise ValueError(msg)
        if 0.0 < exclusion_radius < corner_radius:
            msg = (
                f"exclusion_radius {exclusion_radius} is below the body corner "
                f"radius {corner_radius:.4f} m, so corner-on hits would be "
                "mapped as obstacles. Pass exactly 0.0 to disable filtering."
            )
            raise ValueError(msg)

        self._engine = engine
        self._num_rays = num_rays
        self._max_range = max_range
        self._angular_range = angular_range
        self._exclusion_radius = exclusion_radius
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
                stop = self._teammate_stop_distance(
                    pose.position, direction, hit, teammates
                )
                if stop is not None:
                    # A teammate is not part of the map. Emit a MISS that stops
                    # where the ray ENTERS the teammate's exclusion sphere —
                    # not at the hit. Everything up to that entry point was
                    # provably traversed unobstructed, so it is free; nothing
                    # inside the sphere or beyond it is claimed, which keeps
                    # both the occlusion shadow a real LiDAR would have and any
                    # real geometry that happens to sit near the teammate.
                    obs = RayObservation(
                        origin=pose.position.copy(),
                        direction=direction.copy(),
                        max_range=stop,
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

    def _teammate_stop_distance(
        self,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        hit: RayHit,
        teammates: list[NDArray[np.float64]],
    ) -> float | None:
        """How far to trace free space for a hit that landed on a teammate.

        A hit is attributed to a teammate when it falls inside that teammate's
        exclusion sphere. The returned distance is where the ray *entered* the
        sphere, not where it hit — the distinction matters because the mapper
        marks a MISS free all the way to its endpoint, endpoint included. Using
        the hit distance would write "free" over whatever the ray actually
        struck, and when a filter is wrong that is real geometry: a wall within
        the radius of a teammate would be eroded to free in two observations
        and could then be planned through.

        The sphere entry point is the last position on the ray that is provably
        unobstructed, so it is the furthest we may honestly claim.

        A radial test in 3D. 3D rather than 2D costs nothing while every drone
        shares one altitude, and stays correct if they are ever separated
        vertically — where a 2D test would discard a wall merely sharing an
        (x, y) with a drone flying above it.

        Args:
            origin: Ray origin in world coordinates.
            direction: Unit direction vector.
            hit: The ray-cast result being classified.
            teammates: Teammate positions from `_teammate_positions`.

        Returns:
            The distance at which to stop tracing free space, or None if the
            hit is real geometry and should be kept.
        """
        stop: float | None = None
        for position in teammates:
            offset = hit.hit_point - position
            if float(offset @ offset) >= self._exclusion_radius_sq:
                continue

            # Ray-sphere entry: |origin + t*direction - position| = radius,
            # nearest root. The hit point is inside the sphere, so a real root
            # exists; max() guards only against floating-point noise.
            to_centre = origin - position
            b = float(to_centre @ direction)
            c = float(to_centre @ to_centre) - self._exclusion_radius_sq
            entry = -b - math.sqrt(max(b * b - c, 0.0))
            # Clamp: the origin itself can sit inside the sphere if drones are
            # closer than the radius, in which case nothing is claimed at all.
            entry = max(0.0, entry)

            # Nearest entry wins when spheres overlap; a min is order
            # independent, so teammate ordering cannot affect the result.
            if stop is None or entry < stop:
                stop = entry
        return stop
