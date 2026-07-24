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

# --- Sprint-derived test classification -------------------------------------
#
# Every test declares the sprint it was introduced in via
# ``pytestmark = pytest.mark.sprint(N)`` at the top of its module. From that
# single tag we derive two selectable groups, so promotion is automatic:
#
#   * regression  — tests from sprints BEFORE the current one. They guard
#                   already-shipped behaviour and must not break (PR gate).
#   * progression — tests from the CURRENT sprint; the work in flight
#                   (per-commit gate).
#
# When a sprint closes, bump CURRENT_SPRINT by one: last sprint's progression
# tests become regression with no re-tagging. ``sanity`` is an independent,
# hand-curated marker for a small/fast health-check subset.
#
# Run: ``pytest -m sanity`` / ``-m progression`` / ``-m regression``.
CURRENT_SPRINT = 2


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag each test as regression or progression based on its sprint marker."""
    unmarked: list[str] = []
    for item in items:
        marker = item.get_closest_marker("sprint")
        if marker is None or not marker.args:
            unmarked.append(item.nodeid)
            continue
        sprint = marker.args[0]
        if sprint < CURRENT_SPRINT:
            item.add_marker(pytest.mark.regression)
        else:
            item.add_marker(pytest.mark.progression)

    if unmarked:
        import warnings

        warnings.warn(
            "Tests without a sprint(N) marker are neither regression nor "
            "progression and escape both gates: " + ", ".join(unmarked),
            stacklevel=1,
        )


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
