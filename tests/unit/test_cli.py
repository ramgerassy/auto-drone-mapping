"""Unit tests for CLI helpers.

Only the pure logic is tested here. The live MuJoCo viewer requires a
display and is exercised manually / via the integration run, not in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm_mapping.cli import interpolate_segment

pytestmark = pytest.mark.sprint(1)  # tests introduced in Sprint 1


@pytest.mark.sanity
def test_interpolate_segment_ends_at_target() -> None:
    """The final emitted point must equal the target exactly."""
    start = np.array([0.0, 0.0, 1.0])
    end = np.array([1.0, 0.0, 1.0])

    points = list(interpolate_segment(start, end, max_step=0.15))

    assert np.allclose(points[-1], end)


def test_interpolate_segment_excludes_start() -> None:
    """The start point is not emitted (motion begins after it)."""
    start = np.array([0.0, 0.0, 1.0])
    end = np.array([2.0, 0.0, 1.0])

    points = list(interpolate_segment(start, end, max_step=0.5))

    assert not np.allclose(points[0], start)


def test_interpolate_segment_respects_max_step() -> None:
    """No consecutive hop (including from start) exceeds max_step."""
    start = np.array([0.0, 0.0, 1.0])
    end = np.array([1.0, 1.0, 1.0])
    max_step = 0.15

    points = [start, *interpolate_segment(start, end, max_step)]

    for a, b in zip(points[:-1], points[1:], strict=True):
        assert float(np.linalg.norm(b - a)) <= max_step + 1e-9


def test_interpolate_segment_uniform_spacing() -> None:
    """Points are evenly spaced along the segment."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.0, 0.0])

    points = list(interpolate_segment(start, end, max_step=0.25))

    # 1.0 / 0.25 = 4 hops → 4 points at 0.25, 0.5, 0.75, 1.0
    expected = [
        np.array([0.25, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.75, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    assert len(points) == len(expected)
    for got, exp in zip(points, expected, strict=True):
        assert np.allclose(got, exp)


def test_interpolate_segment_zero_distance() -> None:
    """A zero-length segment still emits exactly one point at the target."""
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([1.0, 2.0, 3.0])

    points = list(interpolate_segment(start, end, max_step=0.15))

    assert len(points) == 1
    assert np.allclose(points[0], end)
