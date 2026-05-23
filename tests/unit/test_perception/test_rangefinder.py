"""Tests for Rangefinder sensor.

Uses FakeEngine from tests/conftest.py — no MuJoCo needed.
Tests verify ray direction generation, quaternion rotation,
hit/miss classification, and scan result metadata.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from swarm_mapping.perception.rangefinder import (
    Rangefinder,
    _rotate_vectors_by_quaternion,
)
from swarm_mapping.simulation.types import Pose, RayHit
from tests.conftest import IDENTITY_POSE, FakeEngine


class TestRangefinder:
    """Tests for the Rangefinder sensor."""

    def test_scan_returns_correct_observation_count(self) -> None:
        """Scan with num_rays=8 returns 8 observations."""
        num_rays = 8
        hits: list[RayHit | None] = [None] * num_rays
        engine = FakeEngine(IDENTITY_POSE, hits)
        sensor = Rangefinder(engine, num_rays=num_rays)  # type: ignore[arg-type]

        result = sensor.scan(0)

        assert len(result.observations) == num_rays

    def test_ray_directions_evenly_spaced(self) -> None:
        """With num_rays=4, directions are at 0, 90, 180, 270 degrees."""
        num_rays = 4
        hits: list[RayHit | None] = [None] * num_rays
        engine = FakeEngine(IDENTITY_POSE, hits)
        sensor = Rangefinder(engine, num_rays=num_rays)  # type: ignore[arg-type]

        result = sensor.scan(0)

        # With identity quaternion, body-frame == world-frame
        directions = np.array([obs.direction for obs in result.observations])

        # angular_range=2pi, start_angle=-pi, angles: -pi, -pi/2, 0, pi/2
        expected = np.array(
            [
                [-1.0, 0.0, 0.0],  # -180°
                [0.0, -1.0, 0.0],  # -90°
                [1.0, 0.0, 0.0],  # 0°
                [0.0, 1.0, 0.0],  # 90°
            ]
        )
        np.testing.assert_allclose(directions, expected, atol=1e-10)

    def test_hit_observation_has_distance_and_point(self) -> None:
        """A ray that hits returns distance and hit_point."""
        hit_point = np.array([2.0, 0.0, 1.0])
        hit = RayHit(distance=2.0, hit_point=hit_point, geom_id=1)
        engine = FakeEngine(IDENTITY_POSE, [hit])
        sensor = Rangefinder(engine, num_rays=1)  # type: ignore[arg-type]

        result = sensor.scan(0)
        obs = result.observations[0]

        assert obs.distance == pytest.approx(2.0)
        assert obs.hit_point is not None
        np.testing.assert_allclose(obs.hit_point, hit_point)

    def test_miss_observation_has_none(self) -> None:
        """A ray that misses returns None distance and hit_point."""
        engine = FakeEngine(IDENTITY_POSE, [None])
        sensor = Rangefinder(engine, num_rays=1)  # type: ignore[arg-type]

        result = sensor.scan(0)
        obs = result.observations[0]

        assert obs.distance is None
        assert obs.hit_point is None

    def test_hit_beyond_max_range_is_miss(self) -> None:
        """A hit beyond max_range is treated as a miss."""
        far_hit = RayHit(
            distance=15.0,
            hit_point=np.array([15.0, 0.0, 1.0]),
            geom_id=1,
        )
        engine = FakeEngine(IDENTITY_POSE, [far_hit])
        sensor = Rangefinder(
            engine,
            num_rays=1,
            max_range=10.0,  # type: ignore[arg-type]
        )

        result = sensor.scan(0)
        obs = result.observations[0]

        assert obs.distance is None
        assert obs.hit_point is None
        assert obs.max_range == pytest.approx(10.0)

    def test_scan_result_metadata(self) -> None:
        """ScanResult has correct drone_id, pose, and timestamp."""
        engine = FakeEngine(IDENTITY_POSE, [None], time=1.5)
        sensor = Rangefinder(engine, num_rays=1)  # type: ignore[arg-type]

        result = sensor.scan(42)

        assert result.drone_id == 42
        np.testing.assert_allclose(result.pose.position, IDENTITY_POSE.position)
        assert result.timestamp == pytest.approx(1.5)

    def test_directions_rotated_by_quaternion(self) -> None:
        """90-degree yaw rotation transforms +x to +y."""
        yaw_90 = Pose(
            position=np.array([0.0, 0.0, 1.0]),
            quaternion=np.array(
                [
                    math.sqrt(2) / 2,
                    0.0,
                    0.0,
                    math.sqrt(2) / 2,
                ]
            ),
        )
        engine = FakeEngine(yaw_90, [None])
        sensor = Rangefinder(engine, num_rays=1)  # type: ignore[arg-type]

        result = sensor.scan(0)
        direction = result.observations[0].direction

        # With 1 ray, body-frame direction is at angle=-pi
        # which is (-1, 0, 0). Rotated 90° around z → (0, -1, 0).
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(direction, expected, atol=1e-10)

    def test_default_parameters(self) -> None:
        """Default Rangefinder has 36 rays and 10m max range."""
        hits: list[RayHit | None] = [None] * 36
        engine = FakeEngine(IDENTITY_POSE, hits)
        sensor = Rangefinder(engine)  # type: ignore[arg-type]

        result = sensor.scan(0)

        assert len(result.observations) == 36
        assert result.observations[0].max_range == pytest.approx(10.0)

    def test_observation_origin_matches_drone_position(self) -> None:
        """Each observation's origin is the drone's position."""
        engine = FakeEngine(IDENTITY_POSE, [None])
        sensor = Rangefinder(engine, num_rays=1)  # type: ignore[arg-type]

        result = sensor.scan(0)

        np.testing.assert_allclose(
            result.observations[0].origin, IDENTITY_POSE.position
        )


class TestQuaternionRotation:
    """Tests for the quaternion rotation helper."""

    def test_identity_quaternion_no_rotation(self) -> None:
        """Identity quaternion leaves vectors unchanged."""
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        result = _rotate_vectors_by_quaternion(vectors, quat)

        np.testing.assert_allclose(result, vectors, atol=1e-10)

    def test_90_degree_yaw(self) -> None:
        """90° yaw (around z) rotates +x to +y."""
        vectors = np.array([[1.0, 0.0, 0.0]])
        quat = np.array([math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2])

        result = _rotate_vectors_by_quaternion(vectors, quat)

        np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_180_degree_yaw(self) -> None:
        """180° yaw rotates +x to -x."""
        vectors = np.array([[1.0, 0.0, 0.0]])
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        result = _rotate_vectors_by_quaternion(vectors, quat)

        np.testing.assert_allclose(result, [[-1.0, 0.0, 0.0]], atol=1e-10)
