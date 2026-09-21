"""Shared data types for the simulation module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pose:
    """6-DOF pose of a body in world coordinates.

    Attributes:
        position: World position as (x, y, z), shape (3,).
        quaternion: Orientation as (w, x, y, z) quaternion, shape (4,).
            Follows MuJoCo convention where w is first.
    """

    position: NDArray[np.float64]
    quaternion: NDArray[np.float64]


@dataclass(frozen=True)
class RayHit:
    """Result of a single ray-cast that hit a geometry.

    Attributes:
        distance: Distance from ray origin to hit point.
        hit_point: World-coordinate hit point, shape (3,).
        geom_id: MuJoCo geom ID of the hit geometry.
    """

    distance: float
    hit_point: NDArray[np.float64]
    geom_id: int


class FailureMode(Enum):
    """How a drone fails.

    Both modes ignore motion commands. They differ in what the drone still
    reports, which is what the coordinator can observe — and so decides which
    detector catches it.

    Attributes:
        SILENT: Crash or comms loss. No heartbeat, no scan, no motion.
        STUCK: Motor fault. Heartbeat and sensor continue; commanded moves do
            not happen.
    """

    SILENT = "silent"
    STUCK = "stuck"
