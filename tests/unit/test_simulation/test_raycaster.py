"""Tests for MjRayCaster."""

from __future__ import annotations

import mujoco
import numpy as np

from swarm_mapping.simulation.raycaster import MjRayCaster


class TestMjRayCaster:
    """Tests for MjRayCaster."""

    def test_ray_hits_wall(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Ray toward east wall returns correct distance.

        Drone is at (0, 0, 1). East wall is at x=3 with half-extent
        0.1, so inner face is at x=2.9. Expected distance ~2.9.
        """
        caster = MjRayCaster(mj_model, mj_data)
        drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone_0")
        origin = mj_data.xpos[drone_body_id].copy()
        directions = np.array([[1.0, 0.0, 0.0]])

        results = caster.cast_rays(origin, directions, drone_body_id)

        assert len(results) == 1
        hit = results[0]
        assert hit is not None
        np.testing.assert_allclose(hit.distance, 2.9, atol=0.05)

    def test_ray_hits_obstacle(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Ray toward obstacle at (1, 1, 0.5) from origin (0, 0, 1).

        Obstacle has half-extent 0.5, so its west face is at x=0.5.
        A ray in the +x direction from (0, 0, 1) should hit the wall
        first at x=2.9, not the obstacle (which is at y=1, not y=0).
        A ray aimed at (1, 1, 0.5) should hit the obstacle.
        """
        caster = MjRayCaster(mj_model, mj_data)
        drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone_0")
        origin = mj_data.xpos[drone_body_id].copy()

        # Direction toward obstacle center
        target = np.array([1.0, 1.0, 0.5])
        direction = target - origin
        direction = direction / np.linalg.norm(direction)
        directions = direction.reshape(1, 3)

        results = caster.cast_rays(origin, directions, drone_body_id)

        assert len(results) == 1
        hit = results[0]
        assert hit is not None
        # Should hit the obstacle, which is closer than the wall
        assert hit.distance < 2.0

    def test_multiple_rays(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Casting multiple rays returns correct number of results."""
        caster = MjRayCaster(mj_model, mj_data)
        drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone_0")
        origin = mj_data.xpos[drone_body_id].copy()
        directions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        results = caster.cast_rays(origin, directions, drone_body_id)

        assert len(results) == 4
        # All four cardinal directions should hit walls in the 6m room
        for hit in results:
            assert hit is not None
            assert 0.0 < hit.distance < 4.0

    def test_ray_excludes_drone_body(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Rays originating from the drone don't hit the drone's own geom."""
        caster = MjRayCaster(mj_model, mj_data)
        drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone_0")
        drone_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, "drone_0_body"
        )
        origin = mj_data.xpos[drone_body_id].copy()
        directions = np.array([[1.0, 0.0, 0.0]])

        results = caster.cast_rays(origin, directions, drone_body_id)

        hit = results[0]
        assert hit is not None
        assert hit.geom_id != drone_geom_id

    def test_hit_point_is_consistent(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Hit point equals origin + direction * distance."""
        caster = MjRayCaster(mj_model, mj_data)
        drone_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone_0")
        origin = mj_data.xpos[drone_body_id].copy()
        direction = np.array([1.0, 0.0, 0.0])
        directions = direction.reshape(1, 3)

        results = caster.cast_rays(origin, directions, drone_body_id)

        hit = results[0]
        assert hit is not None
        expected_point = origin + direction * hit.distance
        np.testing.assert_allclose(hit.hit_point, expected_point)
