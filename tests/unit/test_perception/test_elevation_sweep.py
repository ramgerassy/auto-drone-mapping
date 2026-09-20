"""Tests for the elevation sweep (Sprint 3, Feature 8).

A flat horizontal sweep records the flight altitude into every occupied cell
and nothing else, so the map's height channel carried one value. These pin the
geometry that fixes it, and the rule that stops it erasing obstacles.

Uses FakeEngine — no MuJoCo.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.simulation.types import RayHit
from tests.conftest import IDENTITY_POSE, FakeEngine

pytestmark = pytest.mark.sprint(2)  # Feature 8 — elevation sweep


def sensor(**kwargs: float | int) -> Rangefinder:
    """A Rangefinder over a stub engine with no teammates."""
    rays = int(kwargs.pop("num_rays", 4))
    layers = int(kwargs.pop("elevation_layers", 1))
    hits: list[RayHit | None] = [None] * (rays * layers)
    engine = FakeEngine(IDENTITY_POSE, hits)
    return Rangefinder(
        engine,  # type: ignore[arg-type]
        num_rays=rays,
        elevation_layers=layers,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSweepGeometry:
    """Where the rays point."""

    def test_one_layer_is_the_flat_sweep(self) -> None:
        """A single band must reproduce the horizontal sweep exactly."""
        directions = sensor(num_rays=8, elevation_layers=1)._body_directions

        assert directions.shape == (8, 3)
        assert np.allclose(directions[:, 2], 0.0)

    @pytest.mark.sanity
    def test_bands_span_zero_to_the_maximum_and_never_point_down(self) -> None:
        """Upward only.

        A downward ray strikes the floor, which the mapper would record as an
        obstacle — a ring of phantom walls around every drone. Flying low is
        what makes an upward-only fan sufficient.
        """
        directions = sensor(
            num_rays=6, elevation_layers=4, elevation_max_deg=30.0
        )._body_directions

        assert directions.shape == (24, 3)
        assert (directions[:, 2] >= -1e-12).all()
        assert directions[:, 2].max() == pytest.approx(math.sin(math.radians(30.0)))
        assert np.allclose(directions[:6, 2], 0.0)  # first band is horizontal

    def test_every_direction_is_a_unit_vector(self) -> None:
        """Distances are read straight off `mj_ray`, which assumes unit length."""
        directions = sensor(
            num_rays=7, elevation_layers=3, elevation_max_deg=25.0
        )._body_directions

        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_a_downward_maximum_is_rejected(self) -> None:
        """Fail at construction rather than map the floor as a wall."""
        with pytest.raises(ValueError, match="upward only"):
            sensor(elevation_max_deg=-5.0)

    def test_zero_layers_is_rejected(self) -> None:
        """A sweep with no bands would scan nothing."""
        with pytest.raises(ValueError, match="elevation_layers"):
            sensor(elevation_layers=0)


class TestNavigationPlaneFlag:
    """Which rays may claim free space."""

    def test_only_the_horizontal_band_may_write_free_space(self) -> None:
        """The rule that keeps the sweep from erasing what it finds.

        An elevated ray passes *over* everything between it and its endpoint,
        so it knows nothing about that ground. Letting it mark the ground free
        loses the obstacle: one occupied update (+0.847) against four free ones
        (-1.62) per scan and the obstacle is gone.
        """
        rays, layers = 4, 3
        scan = sensor(
            num_rays=rays, elevation_layers=layers, elevation_max_deg=20.0
        ).scan(0)

        flags = [obs.navigation_plane for obs in scan.observations]

        assert flags[:rays] == [True] * rays
        assert flags[rays:] == [False] * (rays * (layers - 1))

    def test_a_flat_sweep_marks_every_ray_navigational(self) -> None:
        """With one band nothing changes for existing behaviour."""
        scan = sensor(num_rays=5, elevation_layers=1).scan(0)

        assert all(obs.navigation_plane for obs in scan.observations)
