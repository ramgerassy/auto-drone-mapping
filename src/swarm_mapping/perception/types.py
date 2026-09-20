"""Data types for the perception module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.simulation.types import Pose


@dataclass(frozen=True)
class RayObservation:
    """A single ray observation in world coordinates.

    Carries everything the mapper needs to update its grid: the ray
    geometry (origin, direction, max_range) and the result (distance
    and hit point for hits, None for misses). The mapper traces the
    ray through its grid — perception stays grid-ignorant.

    Attributes:
        origin: Ray origin in world coordinates, shape (3,).
        direction: Unit direction vector, shape (3,).
        max_range: On a MISS, how far to trace free space along this ray, in
            metres — the sensor's rating normally, or the shorter distance at
            which a teammate-filtered ray stops. On a HIT this is the sensor's
            rating and the mapper does not read it, tracing to `hit_point`
            instead.
        distance: Distance to hit point, or None if the ray missed (or hit
            beyond the sensor's rated range).
        hit_point: World-coordinate hit point, shape (3,), or None
            if the ray missed.
        navigation_plane: Whether this ray travelled in the drone's horizontal
            plane, and may therefore claim the space it crossed is free.

            An elevated ray carries no information about what lies *beneath*
            it. Letting one write free space erases the obstacles the sweep
            exists to find: a ray passing over a 0.8 m crate and striking a
            wall ten metres beyond would mark the crate's own cell free, and
            with one occupied update (+0.847) against four free ones (-1.62)
            per scan, the crate loses.
    """

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]
    max_range: float
    distance: float | None
    hit_point: NDArray[np.float64] | None
    navigation_plane: bool = True


@dataclass(frozen=True)
class ScanResult:
    """Complete scan result from one sensor reading for one drone.

    Groups all ray observations from a single scan with metadata
    about when and where the scan was taken.

    Attributes:
        drone_id: Which drone produced this scan.
        pose: The drone's pose at the time of the scan.
        observations: List of individual ray observations.
        timestamp: Simulation time when the scan was taken.
    """

    drone_id: int
    pose: Pose
    observations: list[RayObservation]
    timestamp: float
