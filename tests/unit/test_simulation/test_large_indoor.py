"""Geometry invariants for the large indoor scenario (Sprint 2, Feature 5).

Every assertion runs against the **parsed MuJoCo model**, never the file text —
a test that greps XML proves nothing about what MuJoCo loaded.

The scene's dimensions are not free choices. Feature 4c measured that a gap
must clear `2r + 1` cells plus one cell of discretization slop, because a wall
face landing on a cell boundary has its ray hit point attributed to the far
cell. These tests re-assert that the built scene still honours it.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.path_planner import AStarPlanner
from swarm_mapping.simulation.engine import SimulationEngine

pytestmark = pytest.mark.sprint(2)  # Feature 5 — large indoor scenario

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "src" / "swarm_mapping" / "simulation" / "assets" / "large_indoor.xml"
CONFIG = ROOT / "scenarios" / "large_indoor" / "config.yaml"

LATTICE = 0.2
RESOLUTION = 0.2
ORIGIN = -25.0
CELLS = 250
CLEARANCE = 0.20
SPAWN = (0.0, 0.0)

# One probe per room, well inside it.
ROOMS = {
    "NE-near": (12.0, 7.0),
    "NE-far": (12.0, 19.0),
    "NW-near": (-12.0, 7.0),
    "NW-far": (-12.0, 19.0),
    "SE-near": (12.0, -7.0),
    "SE-far": (12.0, -19.0),
    "SW-near": (-12.0, -7.0),
    "SW-far": (-12.0, -19.0),
}


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    """The parsed scene."""
    return mujoco.MjModel.from_xml_path(str(SCENE))


def box_geoms(model: mujoco.MjModel) -> list[tuple[str, float, float, float, float]]:
    """Every box geom as (name, pos_x, pos_y, half_x, half_y)."""
    out = []
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        pos, size = model.geom_pos[i], model.geom_size[i]
        out.append((name, float(pos[0]), float(pos[1]), float(size[0]), float(size[1])))
    return out


def truth_grid(model: mujoco.MjModel) -> OccupancyGrid:
    """A ground-truth occupancy grid built from the parsed geometry.

    A cell is occupied when its extent overlaps a box geom's — the same
    conservative rule the mapper lands on when a ray endpoint falls inside a
    cell, without needing to run a full scan.
    """
    grid = OccupancyGrid(
        MapConfig(
            resolution=RESOLUTION,
            origin_x=ORIGIN,
            origin_y=ORIGIN,
            grid_width=CELLS,
            grid_height=CELLS,
        )
    )
    grid.log_odds[:] = -2.0  # free
    half = RESOLUTION / 2
    for _, px, py, sx, sy in box_geoms(model):
        c0 = max(0, int(np.floor((px - sx - ORIGIN) / RESOLUTION)))
        c1 = min(CELLS - 1, int(np.ceil((px + sx - ORIGIN) / RESOLUTION)))
        r0 = max(0, int(np.floor((py - sy - ORIGIN) / RESOLUTION)))
        r1 = min(CELLS - 1, int(np.ceil((py + sy - ORIGIN) / RESOLUTION)))
        for col in range(c0, c1 + 1):
            for row in range(r0, r1 + 1):
                cx, cy = grid.grid_to_world(col, row)
                if abs(cx - px) < sx + half and abs(cy - py) < sy + half:
                    grid.log_odds[row, col] = 2.0  # occupied
    return grid


class TestSceneLoads:
    """The scene is a scene."""

    @pytest.mark.sanity
    def test_engine_injects_drones_into_it(self) -> None:
        """The scene carries no drone bodies; the engine adds them."""
        engine = SimulationEngine(
            SCENE,
            {
                0: np.array([0.0, 0.0, 1.0]),
                1: np.array([0.0, -1.0, 1.0]),
            },
        )

        assert engine.drone_ids == [0, 1]

    def test_spawn_point_is_clear_of_every_geom(self, model: mujoco.MjModel) -> None:
        """A drone spawned inside a wall would start the mission embedded."""
        half = 0.15  # body half-extent
        for name, px, py, sx, sy in box_geoms(model):
            clear_x = abs(SPAWN[0] - px) >= sx + half
            clear_y = abs(SPAWN[1] - py) >= sy + half
            assert clear_x or clear_y, f"spawn overlaps {name}"


class TestLatticeAlignment:
    """Every wall face on a cell boundary — the invariant behind the widths."""

    def test_every_wall_face_lies_on_the_lattice(self, model: mujoco.MjModel) -> None:
        """A face at a half-cell offset costs a cell of usable gap.

        This is precisely what turned a 3-cell doorway into 2 free cells in the
        Feature 4c measurement, so it is asserted rather than trusted. 0.2 m
        lattice points are also 0.1 m lattice points, which is what lets the
        resolution drop without moving a wall.
        """
        off_lattice = []
        for name, px, py, sx, sy in box_geoms(model):
            for face in (px - sx, px + sx, py - sy, py + sy):
                if abs(face / LATTICE - round(face / LATTICE)) > 1e-9:
                    off_lattice.append((name, face))

        assert off_lattice == []


class TestNavigability:
    """The scene has to be explorable by a drone with a body."""

    @pytest.mark.sanity
    def test_every_room_is_reachable_from_the_spawn(
        self, model: mujoco.MjModel
    ) -> None:
        """The test this scene exists for.

        A doorway one cell too narrow leaves a room permanently unexplorable,
        and the mission would report completion having never entered it. That
        is invisible in the MJCF and obvious here.
        """
        grid = truth_grid(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*SPAWN)

        unreachable = [
            name
            for name, (x, y) in ROOMS.items()
            if planner.plan(grid, start, grid.world_to_grid(x, y)) is None
        ]

        assert unreachable == []

    def test_a_narrowed_doorway_makes_its_rooms_unreachable(
        self, model: mujoco.MjModel
    ) -> None:
        """Proves the test above is not vacuous.

        Filling four cells of one jamb takes the NE doorway from 6 cells to 2 —
        under `2r + 1` — and both rooms behind it drop off the map while the
        rest of the plan is untouched.
        """
        grid = truth_grid(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*SPAWN)

        # The NE doorway is the gap in the north corridor wall (centre y = 1.8)
        # spanning x in [7.0, 8.2].
        _, wall_row = grid.world_to_grid(0.0, 1.8)
        for x in (7.1, 7.3, 7.9, 8.1):
            col, _ = grid.world_to_grid(x, 0.0)
            grid.log_odds[wall_row, col] = 2.0

        def reachable(name: str) -> bool:
            x, y = ROOMS[name]
            return planner.plan(grid, start, grid.world_to_grid(x, y)) is not None

        assert not reachable("NE-near")
        assert not reachable("NE-far")
        assert reachable("NW-near")  # the rest of the plan is unaffected

    def test_doorways_are_at_least_the_measured_floor(
        self, model: mujoco.MjModel
    ) -> None:
        """Each doorway measured on the grid, not read off the file.

        The floor is `2r + 1` cells plus one cell of slop = 4 at r = 1. These
        are 6, so the scene tolerates clearance_radius rising to r = 2.
        """
        grid = truth_grid(model)
        prob = grid.probability()
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        radius = planner._inflation_cells(RESOLUTION)
        floor = 2 * radius + 1 + 1

        # Doorways in the two corridor walls (y = +-1.8) and the four quadrant
        # dividers (y = +-13.0), all spanning x in [7.0, 8.2] or its mirror.
        for wall_y in (1.8, -1.8, 13.0, -13.0):
            _, row = grid.world_to_grid(0.0, wall_y)
            for sign in (1, -1):
                cols = [
                    col
                    for col in range(CELLS)
                    if prob[row, col] < 0.4
                    and 6.0 < sign * grid.grid_to_world(col, row)[0] < 9.2
                ]
                assert len(cols) >= floor, (
                    f"doorway at y={wall_y}, x sign {sign} is {len(cols)} cells, "
                    f"under the {floor}-cell floor"
                )


class TestConfig:
    """The scenario config matches the scene it points at."""

    def test_config_extents_cover_the_whole_floor(self) -> None:
        """Grid extent, origin and the scene's 50 m floor must agree."""
        config = load_config(CONFIG)
        map_cfg = config["map"]

        assert map_cfg["grid_width"] * map_cfg["resolution"] == 50.0
        assert map_cfg["grid_height"] * map_cfg["resolution"] == 50.0
        assert map_cfg["origin_x"] == ORIGIN
        assert map_cfg["origin_y"] == ORIGIN
        assert config["scene"]["path"] == "large_indoor.xml"

    def test_config_clearance_matches_what_the_doorways_were_sized_for(self) -> None:
        """The scene's widths are only correct for this clearance value."""
        config = load_config(CONFIG)

        assert config["planning"]["clearance_radius"] == CLEARANCE
