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
RADIUS = 0.30  # DRONE_EXCLUSION_RADIUS
TEAMMATE_X = -2.0
FACE_DISTANCE = abs(TEAMMATE_X) - HALF_EXTENT  # 1.85, where the ray stops
ENTRY_DISTANCE = abs(TEAMMATE_X) - RADIUS  # 1.70, where the free trace stops


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

    def test_range_is_shortened_to_the_sphere_entry(self) -> None:
        """Case 2: max_range is the exclusion-sphere entry, not the hit.

        This is what preserves the occlusion shadow — the mapper traces a MISS
        to `max_range` and marks every cell free *including the endpoint*. So
        the stop distance must be the last point provably traversed, which is
        where the ray entered the sphere (1.70 m), not where it struck the body
        (1.85 m). Stopping at the hit would write "free" over whatever was
        actually there.
        """
        sensor = sensor_with(head_on_hit(), {1: pose_at(TEAMMATE_X)}, max_range=10.0)

        obs = sensor.scan(0).observations[0]

        assert obs.max_range == pytest.approx(ENTRY_DISTANCE)
        assert obs.max_range < FACE_DISTANCE

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

    def test_wall_inside_the_exclusion_radius_is_discarded_not_overwritten(
        self,
    ) -> None:
        """Case 7: the accepted cost, and the bound on how far it goes.

        A real wall 0.25 m from a teammate's centre is inside the 0.30 m radius
        and is discarded with it — that much is the accepted price of a radial
        filter. What must NOT happen is the free trace running out to the wall:
        the mapper marks a MISS free through its endpoint, so a stop distance of
        2.25 would erode a real wall to free in two observations, and A* would
        then plan through it.

        The trace therefore stops at the sphere entry (1.70 m), well short of
        the wall, which stays unknown until another vantage sees it.
        """
        wall = np.array([TEAMMATE_X - 0.25, 0.0, 1.0])
        hit = RayHit(distance=2.25, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X)})

        obs = sensor.scan(0).observations[0]

        assert obs.distance is None
        assert obs.max_range == pytest.approx(ENTRY_DISTANCE)
        assert obs.max_range < 2.25 - RADIUS  # never reaches the wall cell

    def test_hit_just_outside_the_radius_survives(self) -> None:
        """The radius is pinned from ABOVE as well as below.

        `test_corner_on_hit_is_still_filtered` stops the radius shrinking;
        without this one it could be widened arbitrarily (0.30 -> 0.50 passed
        the whole suite) and silently discard every wall hit near any drone.
        """
        wall = np.array([TEAMMATE_X - 0.31, 0.0, 1.0])
        hit = RayHit(distance=2.31, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X)}, max_range=10.0)

        assert sensor.scan(0).observations[0].distance is not None

    def test_sensing_drone_does_not_filter_itself(self) -> None:
        """Only teammates are excluded — never the drone doing the scanning.

        Dropping the `other != drone_id` guard passed the whole suite. The
        consequence would be a drone blinded to everything within 0.30 m of its
        own centre, which is exactly the near-wall geometry where it most needs
        to see.
        """
        wall = np.array([-0.2, 0.0, 1.0])
        hit = RayHit(distance=0.2, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(-5.0)}, max_range=10.0)

        assert sensor.scan(0).observations[0].distance is not None

    def test_every_teammate_is_checked_not_just_the_first(self) -> None:
        """Three drones, and the hit is on the LAST teammate.

        Truncating the loop to `teammates[:1]` was caught only by a coordination
        test, leaving a perception invariant guarded from another module.
        """
        sensor = sensor_with(
            head_on_hit(),
            {1: pose_at(5.0), 2: pose_at(7.0), 3: pose_at(TEAMMATE_X)},
        )

        assert sensor.scan(0).observations[0].distance is None

    def test_teammate_directly_above_does_not_filter_a_floor_hit(self) -> None:
        """The 3D test earns its keep: a 2D one would discard this wall.

        The teammate shares the hit's (x, y) but flies 0.6 m higher, so it is
        0.6 m away in 3D and must not shadow the hit. Slicing the distance to
        (x, y) passed the whole suite before this test existed.
        """
        wall = np.array([TEAMMATE_X, 0.0, 1.0])
        hit = RayHit(distance=2.0, hit_point=wall, geom_id=3)
        sensor = sensor_with(hit, {1: pose_at(TEAMMATE_X, z=1.6)}, max_range=10.0)

        assert sensor.scan(0).observations[0].distance is not None


class TestRadiusValidation:
    """A mis-set radius must fail loudly, not silently."""

    def test_negative_radius_is_rejected(self) -> None:
        """Squaring would turn -0.30 into +0.30 and look like it worked."""
        with pytest.raises(ValueError, match="must be >= 0"):
            sensor_with(None, exclusion_radius=-0.30)

    def test_radius_below_the_corner_radius_is_rejected(self) -> None:
        """0.20 clears the face distance but not the 0.212 corner."""
        with pytest.raises(ValueError, match="below the body corner radius"):
            sensor_with(None, exclusion_radius=0.20)

    def test_zero_is_accepted_as_a_deliberate_disable(self) -> None:
        """Exactly 0.0 is the greppable opt-out the tests rely on."""
        sensor = sensor_with(
            head_on_hit(), {1: pose_at(TEAMMATE_X)}, exclusion_radius=0.0
        )

        assert sensor.scan(0).observations[0].distance is not None


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
