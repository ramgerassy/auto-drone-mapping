"""Tests for GroundTruthLocalizer."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from swarm_mapping.simulation.localizer import GroundTruthLocalizer


class TestGroundTruthLocalizer:
    """Tests for GroundTruthLocalizer."""

    def test_initial_pose_matches_mjcf(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Drone pose matches the position defined in MJCF."""
        localizer = GroundTruthLocalizer(mj_model, mj_data, {0: "drone_0"})
        pose = localizer.get_pose(0)

        np.testing.assert_allclose(pose.position, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(pose.quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_pose_updates_after_qpos_change(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Pose reflects manual changes to qpos after mj_forward."""
        localizer = GroundTruthLocalizer(mj_model, mj_data, {0: "drone_0"})

        joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, "drone_0_joint"
        )
        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr : qpos_adr + 3] = [2.0, 3.0, 1.5]
        mujoco.mj_forward(mj_model, mj_data)

        pose = localizer.get_pose(0)
        np.testing.assert_allclose(pose.position, [2.0, 3.0, 1.5])

    def test_pose_returns_copy(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Returned pose arrays are copies, not views into MuJoCo data."""
        localizer = GroundTruthLocalizer(mj_model, mj_data, {0: "drone_0"})
        pose = localizer.get_pose(0)
        pose.position[0] = 999.0  # type: ignore[index]

        fresh = localizer.get_pose(0)
        assert fresh.position[0] != 999.0

    def test_unknown_drone_id_raises(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Requesting a pose for an unknown drone_id raises KeyError."""
        localizer = GroundTruthLocalizer(mj_model, mj_data, {0: "drone_0"})
        with pytest.raises(KeyError, match="Unknown drone_id: 99"):
            localizer.get_pose(99)

    def test_invalid_body_name_raises(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Constructing with an invalid body name raises KeyError."""
        with pytest.raises(KeyError, match="not found in model"):
            GroundTruthLocalizer(mj_model, mj_data, {0: "nonexistent_body"})
