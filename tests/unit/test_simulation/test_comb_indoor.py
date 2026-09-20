"""Geometry invariants for the comb indoor scenario (Sprint 2).

Every assertion runs against the **parsed MuJoCo model**, never the file text —
a test that greps XML proves nothing about what MuJoCo loaded.

The scene is an adversarial benchmark: nine dead-end ribs off a central spine.
A rib that a drone cannot reach, or cannot reach without clipping a jamb, would
not fail loudly — the mission would simply report completion having never
entered it, and the strategy comparison the scene exists for would be measured
against a smaller map than intended. These tests are what makes that visible.

The dimensions are not free choices. Feature 4c measured that a gap must clear
`2r + 1` cells plus one cell of discretization slop, because a wall face
landing on a cell boundary has its ray hit point attributed to the far cell.
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
from tests.scene_truth import box_geoms, truth_grid

pytestmark = pytest.mark.sprint(2)  # comb indoor — strategy benchmark scene

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "src" / "swarm_mapping" / "simulation" / "assets" / "comb_indoor.xml"
CONFIG = ROOT / "scenarios" / "comb_indoor" / "config.yaml"

LATTICE = 0.2
RESOLUTION = 0.2
ORIGIN = -20.0
CELLS = 200
CLEARANCE = 0.20
SPAWN = (0.0, 0.0)

# The spine's interior is y in [-SPINE_HALF, +SPINE_HALF]: 3.2 m / 16 cells.
SPINE_HALF = 1.6
# Rib interior half-width: 1.2 m / 6 cells across.
RIB_HALF = 0.6

# name -> (x centre, depth from the spine's inner face, +1 north / -1 south).
RIBS = {
    "N-16": (-16.0, 18.2, 1),
    "S-12": (-12.0, 5.2, -1),
    "N-8": (-8.0, 13.2, 1),
    "S-4": (-4.0, 18.2, -1),
    "N0": (0.0, 9.2, 1),
    "S4": (4.0, 13.2, -1),
    "N8": (8.0, 5.2, 1),
    "S12": (12.0, 18.2, -1),
    "N16": (16.0, 9.2, 1),
}

# How far short of the cap face a tip probe sits. The cell touching the cap is
# free but inflated away by the clearance mask, so a probe placed there would
# be unreachable for a reason that says nothing about the scene. 0.5 m is 2.5
# cells: clear of the mask at r = 1, and still inside the rib at r = 2.
TIP_INSET = 0.5

# Half-width of the window a mouth is counted in. The nearest same-side rib is
# 8 m away, so anything under 4 m isolates one mouth; 1.0 m keeps it obvious.
MOUTH_WINDOW = 1.0


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    """The parsed scene."""
    return mujoco.MjModel.from_xml_path(str(SCENE))


def scene_truth(model: mujoco.MjModel) -> OccupancyGrid:
    """Ground truth for this scene at the scenario's grid geometry."""
    return truth_grid(
        model,
        MapConfig(
            resolution=RESOLUTION,
            origin_x=ORIGIN,
            origin_y=ORIGIN,
            grid_width=CELLS,
            grid_height=CELLS,
        ),
    )


def tip_of(name: str) -> tuple[float, float]:
    """World point just short of a rib's dead end."""
    x, depth, side = RIBS[name]
    return x, side * (SPINE_HALF + depth - TIP_INSET)


def mouth_row(grid: OccupancyGrid, name: str) -> int:
    """Grid row of a rib's mouth — the first row past the spine's wall face.

    This row is the rib's only connection to the rest of the scene, which is
    what makes it both the width to measure and the place to pinch.
    """
    _, _, side = RIBS[name]
    _, row = grid.world_to_grid(0.0, side * (SPINE_HALF + RESOLUTION / 2))
    return row


def mouth_cells(grid: OccupancyGrid, name: str) -> list[int]:
    """Free columns across a rib's mouth, measured on the grid."""
    x, _, _ = RIBS[name]
    prob = grid.probability()
    row = mouth_row(grid, name)
    return [
        col
        for col in range(CELLS)
        if prob[row, col] < 0.4
        and abs(grid.grid_to_world(col, row)[0] - x) < MOUTH_WINDOW
    ]


class TestSceneLoads:
    """The scene is a scene."""

    @pytest.mark.sanity
    def test_engine_injects_drones_into_it(self) -> None:
        """The scene carries no drone bodies; the engine adds them."""
        engine = SimulationEngine(
            SCENE,
            {
                0: np.array([0.0, 0.0, 1.0]),
                1: np.array([-13.0, 0.0, 1.0]),
            },
        )

        assert engine.drone_ids == [0, 1]

    def test_spawn_point_is_clear_of_every_geom(self, model: mujoco.MjModel) -> None:
        """A drone spawned inside a wall would start the mission embedded.

        The spine is the only place in this scene with room to spawn: every
        rib is 1.2 m wide and everything else is solid.
        """
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
        Feature 4c measurement, so it is asserted rather than trusted. The
        geometry was generated from exact rationals; this proves the generated
        numbers survived the round trip through the MJCF and MuJoCo's parser.
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
    def test_every_rib_tip_is_reachable_from_the_spawn(
        self, model: mujoco.MjModel
    ) -> None:
        """The test this scene exists for.

        A rib is a dead end: its tip is reachable through its own mouth or not
        at all, with no second route to cover a mouth one cell too narrow. A
        stranded rib would leave the benchmark quietly measuring strategies
        against a smaller map than the one it documents.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*SPAWN)

        unreachable = [
            name
            for name in RIBS
            if planner.plan(grid, start, grid.world_to_grid(*tip_of(name))) is None
        ]

        assert unreachable == []

    def test_a_narrowed_rib_mouth_strands_its_tip(self, model: mujoco.MjModel) -> None:
        """Proves the test above is not vacuous.

        Filling four cells of the N0 mouth takes it from 6 cells to 2 — under
        `2r + 1` — and its tip drops off the map while its neighbours are
        untouched. Nothing else can reach that tip, which is the property the
        dead ends give the scene.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*SPAWN)

        x, _, _ = RIBS["N0"]
        row = mouth_row(grid, "N0")
        for offset in (-0.5, -0.3, 0.3, 0.5):
            col, _ = grid.world_to_grid(x + offset, 0.0)
            grid.log_odds[row, col] = 2.0

        def reachable(name: str) -> bool:
            return (
                planner.plan(grid, start, grid.world_to_grid(*tip_of(name))) is not None
            )

        assert not reachable("N0")
        assert reachable("N8")  # the rest of the comb is unaffected
        assert reachable("S-12")

    def test_rib_mouths_are_at_least_the_measured_floor(
        self, model: mujoco.MjModel
    ) -> None:
        """Each mouth measured on the grid, not read off the file.

        The floor is `2r + 1` cells plus one cell of slop = 4 at r = 1. The
        mouths are 1.2 m / 6 cells and measure 5 or 6 here: their faces land
        exactly on cell boundaries, and float rounding decides which side of
        the tie a boundary cell falls on. That single cell is the slop the
        floor budgets for — which is why 6 was built for a floor of 4.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        radius = planner._inflation_cells(RESOLUTION)
        floor = 2 * radius + 1 + 1

        for name in RIBS:
            cols = mouth_cells(grid, name)
            assert len(cols) >= floor, (
                f"rib {name} mouth is {len(cols)} cells, under the {floor}-cell floor"
            )
            assert cols == list(range(cols[0], cols[-1] + 1)), (
                f"rib {name} mouth is not contiguous: {cols}"
            )


class TestConfig:
    """The scenario config matches the scene it points at."""

    def test_config_extents_cover_the_whole_floor(self) -> None:
        """Grid extent, origin and the scene's 40 m floor must agree."""
        config = load_config(CONFIG)
        map_cfg = config.map

        assert map_cfg.grid_width * map_cfg.resolution == 40.0
        assert map_cfg.grid_height * map_cfg.resolution == 40.0
        assert map_cfg.origin_x == ORIGIN
        assert map_cfg.origin_y == ORIGIN
        assert config.scene_path == "comb_indoor.xml"

    def test_config_clearance_matches_what_the_mouths_were_sized_for(self) -> None:
        """The scene's widths are only correct for this clearance value."""
        config = load_config(CONFIG)

        assert config.planning.clearance_radius == CLEARANCE

    def test_every_start_position_sits_in_the_spine(self) -> None:
        """The spine is the only region wide enough to spawn a swarm in.

        A start position in a rib would put a drone in a 1.2 m dead end with a
        teammate's exclusion radius across it, and one outside the free space
        entirely would start the mission embedded in fill.
        """
        config = load_config(CONFIG)

        for index, (x, y, _z) in enumerate(config.drones.start_positions):
            assert abs(y) < SPINE_HALF, f"start_positions[{index}] is off the spine"
            assert abs(x) < 20.0 - LATTICE, f"start_positions[{index}] is outside"
