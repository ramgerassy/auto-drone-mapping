"""Shared test fixtures and fakes.

FakeEngine provides a lightweight stub matching SimulationEngine's
interface for unit tests that don't need MuJoCo. Update this single
class if SimulationEngine's API changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_mapping.simulation.types import Pose, RayHit


class FakeEngine:
    """Lightweight stub matching SimulationEngine's public interface.

    Use this in unit tests for modules that depend on SimulationEngine
    (perception, mapping, coordination) without needing MuJoCo.

    Args:
        pose: The pose returned for any drone_id.
        ray_results: The ray-cast results returned for any call.
        time: The simulation time to report.
    """

    def __init__(
        self,
        pose: Pose,
        ray_results: list[RayHit | None],
        time: float = 0.0,
    ) -> None:
        self._pose = pose
        self._ray_results = ray_results
        self._time = time

    def get_pose(self, drone_id: int) -> Pose:
        """Return the configured pose."""
        return self._pose

    def cast_rays(
        self,
        drone_id: int,
        directions: NDArray[np.float64],
    ) -> list[RayHit | None]:
        """Return the configured ray results."""
        return self._ray_results

    @property
    def time(self) -> float:
        """Return the configured simulation time."""
        return self._time


IDENTITY_POSE = Pose(
    position=np.array([0.0, 0.0, 1.0]),
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
)


@pytest.fixture
def identity_pose() -> Pose:
    """A pose at (0, 0, 1) with identity rotation."""
    return IDENTITY_POSE
