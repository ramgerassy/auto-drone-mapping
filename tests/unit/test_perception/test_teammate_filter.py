"""Tests for the teammate filter in Rangefinder (Sprint 2, Feature 4b).

A drone must never map its teammates as obstacles. `mujoco.mj_ray` can exclude
only one body — the sensing drone's own — so every other drone comes back as a
solid hit. The filter compares each hit point against teammates' ground-truth
positions and re-encodes a teammate hit as a MISS with the range shortened to
the teammate distance: free space up to it, nothing occupied, nothing claimed
beyond.

Uses FakeEngine — no MuJoCo. The pipeline consequences are covered in
tests/integration/test_teammate_filter.py.

Geometry note: with `num_rays=1` the single ray points along **-x** (body-frame
angle -pi), so teammates in these tests sit on the negative x axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.simulation.types import Pose, RayHit
from tests.conftest import IDENTITY_POSE, FakeEngine

pytestmark = pytest.mark.sprint(2)  # Feature 4b — perception teammate filter

# The drone body geom is a box of half-extent 0.15, so a head-on ray stops
# 0.15 m short of the teammate's centre.
HALF_EXTENT = 0.15
TEAMMATE_X = -2.0
FACE_DISTANCE = abs(TEAMMATE_X) - HALF_EXTENT  # 1.85


def pose_at(x: float, y: float = 0.0, z: float = 1.0) -> Pose:
    """A pose at (x, y, z) with identity orientation."""
    return Pose(
        position=np.array([x, y, z]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def sensor_with(
    hit: RayHit | None,
    teammates: dict[int, Pose] | None = None,
    **kwargs: float,
) -> Rangefinder:
    """A single-ray Rangefinder for drone 0, with optional teammates."""
    poses = {0: IDENTITY_POSE}
    if teammates:
        poses.update(teammates)
    engine = FakeEngine(IDENTITY_POSE, [hit], poses=poses)
    return Rangefinder(engine, num_rays=1, **kwargs)  # type: ignore[arg-type]


def head_on_hit() -> RayHit:
    """A ray landing dead centre on the near face of the teammate at -2 m."""
    return RayHit(
        distance=FACE_DISTANCE,
        hit_point=np.array([TEAMMATE_X + HALF_EXTENT, 0.0, 1.0]),
        geom_id=7,
    )


class TestTeammateHitsAreFiltered:
    """A hit on a teammate becomes a shortened MISS."""

    @pytest.mark.sanity
    def test_teammate_hit_yields_no_hit_point(self) -> None:
        """Case 1: the observation carries no distance and no hit point."""
        sensor = sensor_with(head_on_hit(), {1: pose_at(TEAMMATE_X)})

        obs = sensor.scan(0).observations[0]

        assert obs.distance is None
        assert obs.hit_point is None

    def test_range_is_shortened_to_the_teammate(self) -> None:
        """Case 2: max_range becomes the hit distance, not the sensor rating.

        This is what preserves the occlusion shadow — the mapper traces a MISS
        to `max_range`, so it marks free space up to the teammate and makes no
        claim about the space behind it.
        """
        sensor = sensor_with(head_on_hit(), {1: pose_at(TEAMMATE_X)}, max_range=10.0)

        obs = sensor.scan(0).observations[0]

        assert obs.max_range == pytest.approx(FACE_DISTANCE)

    def test_corner_on_hit_is_still_filtered(self) -> None:
        """Case 5: a hit at the box corner, 0.212 m off-centre, is caught.

        Pins the exclusion radius above the box circumradius
        (0.15 * sqrt(2) = 0.2121). A radius tuned to the face distance alone
        would let corner-on hits through.
        """
        corner = np.array([TEAMMATE_X + HALF_EXTENT, HALF_EXTENT, 1.0])
        offset = float(np.linalg.norm(corner - np.array([TEAMMATE_X, 0.0, 1.0])))
        assert offset == pytest.approx(0.2121, abs=1e-3)  # the case is what we think

        hit = RayHit(
            distance=float(np.linalg.norm(corner)), hit_point=corner, geom_id=7
        )
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X)})

        assert sensor.scan(0).observations[0].distance is None


class TestRealObstaclesSurvive:
    """The filter must not swallow genuine geometry."""

    def test_wall_hit_is_untouched(self) -> None:
        """Case 3: a wall far from any teammate passes through unchanged."""
        wall = np.array([-5.0, 0.0, 1.0])
        hit = RayHit(distance=5.0, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X)}, max_range=10.0)

        obs = sensor.scan(0).observations[0]

        assert obs.distance == pytest.approx(5.0)
        assert obs.hit_point is not None
        np.testing.assert_allclose(obs.hit_point, wall)
        assert obs.max_range == pytest.approx(10.0)

    def test_single_drone_swarm_filters_nothing(self) -> None:
        """Case 4: with no teammates, the same hit is kept."""
        sensor = sensor_with(head_on_hit())  # drone_ids defaults to [0]

        obs = sensor.scan(0).observations[0]

        assert obs.distance == pytest.approx(FACE_DISTANCE)
        assert obs.hit_point is not None


class TestBoundaries:
    """Edges where the filter interacts with existing behaviour."""

    def test_teammate_beyond_max_range_stays_a_full_range_miss(self) -> None:
        """Case 6: out-of-range hits keep today's semantics.

        A hit past `max_range` is already reported as a full-range MISS; a
        teammate out there is the same case. The distinguishing assertion is
        `max_range` — the filter would have shortened it to 15.0.
        """
        far = np.array([-15.0, 0.0, 1.0])
        hit = RayHit(distance=15.0, hit_point=far, geom_id=7)
        sensor = sensor_with(hit, {1: pose_at(-15.0)}, max_range=10.0)

        obs = sensor.scan(0).observations[0]

        assert obs.distance is None
        assert obs.max_range == pytest.approx(10.0)

    def test_wall_inside_the_exclusion_radius_is_discarded(self) -> None:
        """Case 7: the accepted cost, pinned as known behaviour.

        A real wall 0.25 m behind a teammate's centre is inside the 0.30 m
        radius and is discarded with it. That cell stays unknown for this tick
        and is mapped later from another vantage. Documented in
        docs/progress.md as the price of a radial filter.
        """
        wall = np.array([TEAMMATE_X - 0.25, 0.0, 1.0])
        hit = RayHit(distance=2.25, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X)})

        obs = sensor.scan(0).observations[0]

        assert obs.distance is None
        assert obs.max_range == pytest.approx(2.25)


class TestDeterminism:
    """Same scene, same scan."""

    def test_repeated_scans_identical(self) -> None:
        """Case 8: filtering is a pure function of poses and hits."""
        sensor = sensor_with(head_on_hit(), {1: pose_at(TEAMMATE_X)})

        first = sensor.scan(0).observations[0]
        second = sensor.scan(0).observations[0]

        assert first.distance == second.distance
        assert first.max_range == second.max_range
        np.testing.assert_array_equal(first.direction, second.direction)
        np.testing.assert_array_equal(first.origin, second.origin)
