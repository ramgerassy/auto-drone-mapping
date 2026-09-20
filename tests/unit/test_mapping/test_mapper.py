"""Tests for Mapper scan integration."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.types import RayObservation, ScanResult
from swarm_mapping.simulation.types import Pose

pytestmark = pytest.mark.sprint(1)  # tests introduced in Sprint 1

# Shared config: 10x10 grid, 1m resolution, origin at (0, 0)
TEST_CONFIG = MapConfig(
    resolution=1.0,
    origin_x=0.0,
    origin_y=0.0,
    grid_width=10,
    grid_height=10,
)

DUMMY_POSE = Pose(
    position=np.array([5.0, 5.0, 1.0]),
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
)


def _make_scan(observations: list[RayObservation]) -> ScanResult:
    """Helper to build a ScanResult with the given observations."""
    return ScanResult(
        drone_id=0,
        pose=DUMMY_POSE,
        observations=observations,
        timestamp=0.0,
    )


class TestMapper:
    """Tests for Mapper."""

    @pytest.mark.sanity
    def test_single_hit_marks_free_and_occupied(self) -> None:
        """A hit ray marks cells along path as free and endpoint as occupied."""
        mapper = Mapper(TEST_CONFIG)

        # Ray from (1.5, 5.5) to (4.5, 5.5) — horizontal, hits at x=4.5
        obs = RayObservation(
            origin=np.array([1.5, 5.5, 1.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_range=10.0,
            distance=3.0,
            hit_point=np.array([4.5, 5.5, 1.0]),
        )
        mapper.integrate_scan(_make_scan([obs]))

        grid = mapper.grid
        # Cells (1,5), (2,5), (3,5) should be free (log-odds < 0)
        assert grid.log_odds[5, 1] < 0.0
        assert grid.log_odds[5, 2] < 0.0
        assert grid.log_odds[5, 3] < 0.0
        # Cell (4,5) should be occupied (log-odds > 0)
        assert grid.log_odds[5, 4] > 0.0

    def test_single_miss_marks_all_free(self) -> None:
        """A miss ray marks all cells along path as free."""
        mapper = Mapper(TEST_CONFIG)

        obs = RayObservation(
            origin=np.array([1.5, 5.5, 1.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_range=5.0,
            distance=None,
            hit_point=None,
        )
        mapper.integrate_scan(_make_scan([obs]))

        grid = mapper.grid
        # All cells along path should be free
        for col in range(1, 7):
            assert grid.log_odds[5, col] < 0.0

    def test_height_updated_on_hit(self) -> None:
        """Hit updates the height of the endpoint cell."""
        mapper = Mapper(TEST_CONFIG)

        obs = RayObservation(
            origin=np.array([1.5, 5.5, 1.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_range=10.0,
            distance=3.0,
            hit_point=np.array([4.5, 5.5, 2.5]),
        )
        mapper.integrate_scan(_make_scan([obs]))

        assert mapper.grid.height[5, 4] == pytest.approx(2.5)

    def test_height_tracks_maximum(self) -> None:
        """Multiple hits on same cell keep the maximum height."""
        mapper = Mapper(TEST_CONFIG)

        for z in [1.5, 3.0, 2.0]:
            obs = RayObservation(
                origin=np.array([3.5, 5.5, 1.0]),
                direction=np.array([1.0, 0.0, 0.0]),
                max_range=10.0,
                distance=1.0,
                hit_point=np.array([4.5, 5.5, z]),
            )
            mapper.integrate_scan(_make_scan([obs]))

        assert mapper.grid.height[5, 4] == pytest.approx(3.0)

    def test_multiple_scans_accumulate(self) -> None:
        """Multiple scans on same cell accumulate log-odds."""
        mapper = Mapper(TEST_CONFIG)

        obs = RayObservation(
            origin=np.array([3.5, 5.5, 1.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_range=10.0,
            distance=1.0,
            hit_point=np.array([4.5, 5.5, 1.0]),
        )

        mapper.integrate_scan(_make_scan([obs]))
        first_odds = mapper.grid.log_odds[5, 4]

        mapper.integrate_scan(_make_scan([obs]))
        second_odds = mapper.grid.log_odds[5, 4]

        assert second_odds > first_odds

    def test_out_of_bounds_ray_no_crash(self) -> None:
        """Ray ending outside grid does not raise, updates in-bounds cells."""
        mapper = Mapper(TEST_CONFIG)

        # Ray starts in-bounds, ends outside grid
        obs = RayObservation(
            origin=np.array([8.5, 5.5, 1.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_range=10.0,
            distance=None,
            hit_point=None,
        )
        mapper.integrate_scan(_make_scan([obs]))

        # Cell (8, 5) and (9, 5) should be free, no crash
        assert mapper.grid.log_odds[5, 8] < 0.0
        assert mapper.grid.log_odds[5, 9] < 0.0

    def test_determinism(self) -> None:
        """Same observations produce identical grid state."""
        obs_list = [
            RayObservation(
                origin=np.array([1.5, 5.5, 1.0]),
                direction=np.array([1.0, 0.0, 0.0]),
                max_range=10.0,
                distance=3.0,
                hit_point=np.array([4.5, 5.5, 1.0]),
            ),
            RayObservation(
                origin=np.array([5.5, 1.5, 1.0]),
                direction=np.array([0.0, 1.0, 0.0]),
                max_range=10.0,
                distance=None,
                hit_point=None,
            ),
        ]

        mapper_a = Mapper(TEST_CONFIG)
        mapper_a.integrate_scan(_make_scan(obs_list))

        mapper_b = Mapper(TEST_CONFIG)
        mapper_b.integrate_scan(_make_scan(obs_list))

        np.testing.assert_array_equal(mapper_a.grid.log_odds, mapper_b.grid.log_odds)
        np.testing.assert_array_equal(mapper_a.grid.height, mapper_b.grid.height)

    def test_empty_scan_no_change(self) -> None:
        """Empty scan does not modify the grid."""
        mapper = Mapper(TEST_CONFIG)
        mapper.integrate_scan(_make_scan([]))
        assert np.all(mapper.grid.log_odds == 0.0)


class TestElevatedRaysDoNotEraseObstacles:
    """Sprint 3, Feature 8 — only navigation-plane rays write free space."""

    def grid_config(self) -> MapConfig:
        """A 1 m-cell grid big enough for a ten-cell ray."""
        return MapConfig(
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            grid_width=12,
            grid_height=12,
        )

    def test_an_elevated_ray_does_not_mark_the_ground_beneath_it_free(self) -> None:
        """The rule this feature turns on.

        An upward ray passing over a short obstacle and striking a wall beyond
        would, under plain 2D projection, mark the obstacle's own cell free.
        With one occupied update against four free ones per scan the obstacle
        loses — so the sweep would erase exactly what it was added to find.
        """
        mapper = Mapper(self.grid_config())
        # A navigation-plane ray finds the obstacle at (3, 0).
        mapper.integrate_scan(
            ScanResult(
                drone_id=0,
                pose=Pose(
                    position=np.array([0.5, 0.5, 0.3]),
                    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
                observations=[
                    RayObservation(
                        origin=np.array([0.5, 0.5, 0.3]),
                        direction=np.array([1.0, 0.0, 0.0]),
                        max_range=10.0,
                        distance=3.0,
                        hit_point=np.array([3.5, 0.5, 0.3]),
                        navigation_plane=True,
                    )
                ],
                timestamp=0.0,
            )
        )
        before = mapper.grid.probability()[0, 3]

        # Four elevated rays pass over it and strike a wall at x = 9.5.
        mapper.integrate_scan(
            ScanResult(
                drone_id=0,
                pose=Pose(
                    position=np.array([0.5, 0.5, 0.3]),
                    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
                observations=[
                    RayObservation(
                        origin=np.array([0.5, 0.5, 0.3]),
                        direction=np.array([0.97, 0.0, 0.26]),
                        max_range=10.0,
                        distance=9.0,
                        hit_point=np.array([9.5, 0.5, 2.6]),
                        navigation_plane=False,
                    )
                ]
                * 4,
                timestamp=1.0,
            )
        )

        assert mapper.grid.probability()[0, 3] == pytest.approx(before)

    def test_an_elevated_ray_still_records_occupancy_and_height(self) -> None:
        """It reports what it struck — that is the whole point of the band."""
        mapper = Mapper(self.grid_config())
        mapper.integrate_scan(
            ScanResult(
                drone_id=0,
                pose=Pose(
                    position=np.array([0.5, 0.5, 0.3]),
                    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
                observations=[
                    RayObservation(
                        origin=np.array([0.5, 0.5, 0.3]),
                        direction=np.array([0.97, 0.0, 0.26]),
                        max_range=10.0,
                        distance=9.0,
                        hit_point=np.array([9.5, 0.5, 2.6]),
                        navigation_plane=False,
                    )
                ],
                timestamp=0.0,
            )
        )

        assert mapper.grid.probability()[0, 9] > 0.5
        assert mapper.grid.height[0, 9] == pytest.approx(2.6)
