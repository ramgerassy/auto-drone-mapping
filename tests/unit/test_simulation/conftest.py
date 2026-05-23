"""Shared fixtures for simulation module tests.

Uses a tiny inline MJCF environment — a 6m x 6m box room with one
obstacle and a freejoint drone. All tests run headless.
"""

from __future__ import annotations

import mujoco
import pytest

TINY_ROOM_XML = """\
<mujoco model="test_room">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <geom name="wall_east" type="box" pos="3 0 1" size="0.1 3 1"/>
    <geom name="wall_west" type="box" pos="-3 0 1" size="0.1 3 1"/>
    <geom name="wall_north" type="box" pos="0 3 1" size="3 0.1 1"/>
    <geom name="wall_south" type="box" pos="0 -3 1" size="3 0.1 1"/>
    <geom name="obstacle" type="box" pos="1 1 0.5" size="0.5 0.5 0.5"/>
    <body name="drone_0" pos="0 0 1">
      <freejoint name="drone_0_joint"/>
      <geom name="drone_0_body" type="box" size="0.1 0.1 0.05" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def mj_model() -> mujoco.MjModel:
    """Return a MuJoCo model loaded from the tiny room XML."""
    return mujoco.MjModel.from_xml_string(TINY_ROOM_XML)


@pytest.fixture
def mj_data(mj_model: mujoco.MjModel) -> mujoco.MjData:
    """Return MuJoCo data with forward kinematics computed."""
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    return data
