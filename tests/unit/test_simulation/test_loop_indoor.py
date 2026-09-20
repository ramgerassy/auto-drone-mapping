"""Geometry invariants for the loop indoor scenario (Sprint 2, Feature 5).

Every assertion runs against the **parsed MuJoCo model**, never the file text —
a test that greps XML proves nothing about what MuJoCo loaded.

The scene's dimensions are not free choices. Feature 4c measured that a gap
must clear `2r + 1` cells plus one cell of discretization slop, because a wall
face landing on a cell boundary has its ray hit point attributed to the far
cell. These tests re-assert that the built scene still honours it.

Beyond that, this scene exists for one property — **loop connectivity** — and
`TestLoopConnectivity` is the reason the file is here. large_indoor is a
corridor cross with a single junction, so a drone exploring two opposite
quadrants has no choice but to cross the junction twice; its measured long-gap
revisits may therefore be forced by the floor plan rather than chosen by the
frontier strategy. This scene removes the chokepoint, and the test below is
what proves the chokepoint is actually gone rather than merely rearranged.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.planning.path_planner import AStarPlanner, Cell
from swarm_mapping.simulation.engine import SimulationEngine
from tests.scene_truth import box_geoms, truth_grid

pytestmark = pytest.mark.sprint(2)  # Feature 5 — loop indoor scenario

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "src" / "swarm_mapping" / "simulation" / "assets"
SCENE = ASSETS / "loop_indoor.xml"
CROSS_SCENE = ASSETS / "large_indoor.xml"  # the single-chokepoint control
CONFIG = ROOT / "scenarios" / "loop_indoor" / "config.yaml"

LATTICE = 0.2
RESOLUTION = 0.2
ORIGIN = -25.0
CELLS = 250
CLEARANCE = 0.20
WALL_TOP = 3.0  # walls reach the full height; obstacles are everything shorter

# The three ring-corridor spawns, from the scenario config.
# All five, so the reachability and clearance checks below cover every spawn
# the config can actually use. The last two were added to take the scaling KPI
# to five drones; they sit on the south leg and the north-east corner, keeping
# the set spread around the racetrack rather than clustered.
SPAWNS = (
    (0.0, 14.4),
    (14.4, 0.0),
    (-14.4, 0.0),
    (0.0, -14.4),
    (14.4, 14.4),
)

# One probe per room, well inside it and clear of every obstacle.
ROOMS = {
    "N-outer": (0.0, 20.6),
    "S-outer": (0.0, -20.6),
    "E-outer": (20.6, 0.0),
    "W-outer": (-20.6, 0.0),
    "NE-corner": (20.6, 20.6),
    "NW-corner": (-20.6, 20.6),
    "SE-corner": (20.6, -20.6),
    "SW-corner": (-20.6, -20.6),
    "NE-core": (7.2, 7.2),
    "NW-core": (-7.2, 7.2),
    "SE-core": (7.2, -7.2),
    "SW-core": (-7.2, -7.2),
}

# Every gap in the plan, as (label, wall axis, wall coordinate, gap centre,
# nominal width). "x" names a wall whose face is at that x, so its gap is
# measured along y, and vice versa. Half-widths of the measuring window are
# derived from the nominal width, so a window never reaches a neighbouring gap.
DOORWAY = 1.2
CORRIDOR = 3.2
_OUTER_RING_GAPS = (-20.6, -8.0, 8.0, 20.6)
_INNER_RING_GAPS = ((-7.2, DOORWAY), (0.0, CORRIDOR), (7.2, DOORWAY))
GAPS: list[tuple[str, str, float, float, float]] = []
for _line in (16.2, -16.2):
    for _c in _OUTER_RING_GAPS:
        GAPS.append((f"outer ring y={_line} at x={_c}", "y", _line, _c, DOORWAY))
        GAPS.append((f"outer ring x={_line} at y={_c}", "x", _line, _c, DOORWAY))
for _line in (12.6, -12.6):
    for _c, _w in _INNER_RING_GAPS:
        GAPS.append((f"inner ring y={_line} at x={_c}", "y", _line, _c, _w))
    GAPS.append((f"inner ring x={_line} at y=0.0", "x", _line, 0.0, CORRIDOR))
for _line in (1.8, -1.8):
    for _c in (-7.2, 7.2):
        GAPS.append((f"cross wall y={_line} at x={_c}", "y", _line, _c, DOORWAY))

# Region pairs that must have two disjoint routes. Each pair is chosen so the
# short route uses a different part of the plan from the long one: through the
# cross-corridors versus around the ring.
DISJOINT_PAIRS = (
    ("ring north leg", (0.0, 14.4), "ring south leg", (0.0, -14.4)),
    ("E-outer room", (20.6, 0.0), "W-outer room", (-20.6, 0.0)),
    ("NE-corner room", (20.6, 20.6), "SW-corner room", (-20.6, -20.6)),
)

# How much of a route to wall off, and how far to spread the block sideways.
#
# The dilation is deliberately larger than the widest passage in the scene
# (3.2 m, 16 cells): a 33-cell-wide block centred on a path cell severs
# whatever the path was travelling through, whichever side of it the path was
# hugging. A smaller dilation would let the "second" route slide past the
# blockage in the same corridor, which would make the test pass without
# proving anything.
BLOCK_DILATION = 16
BLOCK_FRACTION = 0.6


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


def obstacle_tops(model: mujoco.MjModel) -> list[float]:
    """Top height in metres of every box geom shorter than a wall.

    Identified by height rather than by name so renaming a crate cannot
    silently drop it from the check.
    """
    tops = []
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        top = float(model.geom_pos[i][2]) + float(model.geom_size[i][2])
        if top < WALL_TOP:
            tops.append(top)
    return sorted(tops)


def block_middle(grid: OccupancyGrid, path: list[Cell]) -> OccupancyGrid:
    """Return a copy of `grid` with the middle of `path` walled off.

    Blocks `BLOCK_FRACTION` of the route, centred, dilating each blocked cell
    by `BLOCK_DILATION` so the passage the route used is severed rather than
    merely narrowed. The ends are left open so the start and goal stay legal
    cells to plan from.

    Args:
        grid: The grid to copy. Not modified.
        path: A planned route, start to goal.

    Returns:
        A new grid with the middle of the route marked occupied.
    """
    blocked = OccupancyGrid(grid.config)
    blocked.log_odds[:] = grid.log_odds
    margin = int(len(path) * (1.0 - BLOCK_FRACTION) / 2)
    for col, row in path[margin : len(path) - margin]:
        row_lo = max(0, row - BLOCK_DILATION)
        col_lo = max(0, col - BLOCK_DILATION)
        blocked.log_odds[
            row_lo : row + BLOCK_DILATION + 1, col_lo : col + BLOCK_DILATION + 1
        ] = 2.0
    return blocked


class TestSceneLoads:
    """The scene is a scene."""

    @pytest.mark.sanity
    def test_engine_injects_drones_into_it(self) -> None:
        """The scene carries no drone bodies; the engine adds them."""
        engine = SimulationEngine(
            SCENE,
            {
                0: np.array([0.0, 14.4, 1.0]),
                1: np.array([14.4, 0.0, 1.0]),
                2: np.array([-14.4, 0.0, 1.0]),
            },
        )

        assert engine.drone_ids == [0, 1, 2]

    def test_every_spawn_is_clear_of_every_geom(self, model: mujoco.MjModel) -> None:
        """A drone spawned inside a wall would start the mission embedded.

        Three spawns rather than one, because the scene deliberately starts the
        swarm apart — see the config. Each has to be checked.
        """
        half = 0.15  # body half-extent
        for spawn_x, spawn_y in SPAWNS:
            for name, px, py, sx, sy in box_geoms(model):
                clear_x = abs(spawn_x - px) >= sx + half
                clear_y = abs(spawn_y - py) >= sy + half
                assert clear_x or clear_y, f"spawn {(spawn_x, spawn_y)} in {name}"


class TestLatticeAlignment:
    """Every wall face on a cell boundary — the invariant behind the widths."""

    def test_every_wall_face_lies_on_the_lattice(self, model: mujoco.MjModel) -> None:
        """A face at a half-cell offset costs a cell of usable gap.

        This is precisely what turned a 3-cell doorway into 2 free cells in the
        Feature 4c measurement, so it is asserted rather than trusted. The
        geometry was generated from `fractions.Fraction`, which makes the
        property true by construction; this proves the emitted file kept it.
        """
        off_lattice = []
        for name, px, py, sx, sy in box_geoms(model):
            for face in (px - sx, px + sx, py - sy, py + sy):
                if abs(face / LATTICE - round(face / LATTICE)) > 1e-9:
                    off_lattice.append((name, face))

        assert off_lattice == []


class TestObstacleHeights:
    """Height is information the 2.5D map records, so the scene varies it."""

    def test_obstacles_span_the_flight_altitude_and_reach_near_wall_height(
        self, model: mujoco.MjModel
    ) -> None:
        """large_indoor's crates are all 0.8-1.2 m and barely exercise height.

        The drone flies at 0.3 m with an upward elevation fan, so an obstacle
        below that altitude is invisible to it and one near wall height fills
        the fan. Both ends have to be present for the height channel to be
        worth measuring at all.
        """
        tops = obstacle_tops(model)

        assert len(tops) >= 4, f"only {len(tops)} obstacles"
        assert tops[0] < 0.3, f"nothing below the 0.3 m flight altitude: {tops}"
        assert tops[-1] > 2.0, f"nothing above 2 m: {tops}"
        assert len(set(tops)) == len(tops), f"heights are not distinct: {tops}"


class TestNavigability:
    """The scene has to be explorable by a drone with a body."""

    @pytest.mark.sanity
    def test_every_room_is_reachable_from_every_spawn(
        self, model: mujoco.MjModel
    ) -> None:
        """A doorway one cell too narrow leaves a room permanently unexplorable.

        The mission would report completion having never entered it. That is
        invisible in the MJCF and obvious here. Checked from all three spawns,
        since the swarm starts spread around the ring.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)

        unreachable = [
            (spawn, name)
            for spawn in SPAWNS
            for name, (x, y) in ROOMS.items()
            if planner.plan(grid, grid.world_to_grid(*spawn), grid.world_to_grid(x, y))
            is None
        ]

        assert unreachable == []

    def test_a_narrowed_doorway_makes_its_room_unreachable(
        self, model: mujoco.MjModel
    ) -> None:
        """Proves the test above is not vacuous.

        The NE core room has exactly two doors, both 6 cells wide and both at
        x = 7.2 — one north onto the ring, one south onto the cross-corridor.
        Filling four cells of each jamb takes them under `2r + 1` and the room
        drops off the map, while every other room is untouched.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*SPAWNS[0])

        for wall_y in (12.6, 1.8):
            _, wall_row = grid.world_to_grid(0.0, wall_y)
            for x in (6.7, 6.9, 7.5, 7.7):
                col, _ = grid.world_to_grid(x, 0.0)
                grid.log_odds[wall_row, col] = 2.0

        def reachable(name: str) -> bool:
            x, y = ROOMS[name]
            return planner.plan(grid, start, grid.world_to_grid(x, y)) is not None

        assert not reachable("NE-core")
        assert reachable("NW-core")  # the rest of the plan is unaffected
        assert reachable("SE-core")

    def test_every_gap_is_at_least_the_measured_floor(
        self, model: mujoco.MjModel
    ) -> None:
        """Each doorway and corridor mouth measured on the grid, not the file.

        The floor is `2r + 1` cells plus one cell of slop = 4 at r = 1. Wall
        faces sit exactly half a cell from the nearest cell centre, so a
        nominally 6-cell doorway measures 5 or 6 depending on which side of the
        knife edge the float comparison in `truth_grid` lands — which is why
        the assertion is a floor rather than an equality, and why the nominal
        widths carry two cells of headroom over it.
        """
        grid = scene_truth(model)
        prob = grid.probability()
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        floor = 2 * planner._inflation_cells(RESOLUTION) + 1 + 1

        narrow = []
        for label, axis, line, centre, width in GAPS:
            half = width / 2 + 0.2  # one cell of slack, never a neighbour's gap
            if axis == "y":
                _, row = grid.world_to_grid(0.0, line)
                free = [
                    col
                    for col in range(CELLS)
                    if prob[row, col] < 0.4
                    and abs(grid.grid_to_world(col, row)[0] - centre) < half
                ]
            else:
                col, _ = grid.world_to_grid(line, 0.0)
                free = [
                    row
                    for row in range(CELLS)
                    if prob[row, col] < 0.4
                    and abs(grid.grid_to_world(col, row)[1] - centre) < half
                ]
            if len(free) < floor:
                narrow.append((label, len(free)))

        assert narrow == [], f"gaps under the {floor}-cell floor: {narrow}"


class TestLoopConnectivity:
    """The property the whole scenario exists for.

    Two genuinely distinct routes between any two regions, and no single cell
    whose removal disconnects the map. Without this the scene is just a
    rearrangement of large_indoor and answers nothing about whether the
    measured long-gap revisits are structural or strategic.
    """

    @pytest.mark.parametrize(("name_a", "point_a", "name_b", "point_b"), DISJOINT_PAIRS)
    def test_two_disjoint_routes_exist(
        self,
        model: mujoco.MjModel,
        name_a: str,
        point_a: tuple[float, float],
        name_b: str,
        point_b: tuple[float, float],
    ) -> None:
        """Plan a route, sever it, and plan again.

        The second plan cannot reuse any part of the severed middle — the block
        is wider than the widest passage in the scene — so a second route being
        found is proof that the two are disjoint, not that one squeezed past
        the other. The overlap bound then pins down *how* disjoint: the two
        routes may share their approach to the endpoints and nothing else.
        """
        grid = scene_truth(model)
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(*point_a)
        goal = grid.world_to_grid(*point_b)

        first = planner.plan(grid, start, goal)
        assert first is not None, f"no route at all from {name_a} to {name_b}"

        second = planner.plan(block_middle(grid, first), start, goal)
        assert second is not None, (
            f"severing the middle of the {name_a} -> {name_b} route "
            f"disconnects them: that is a chokepoint, not a loop"
        )

        shared = set(first) & set(second)
        assert len(shared) < 0.25 * len(first), (
            f"the two {name_a} -> {name_b} routes share {len(shared)} of "
            f"{len(first)} cells — they are not distinct routes"
        )

    def test_the_corridor_cross_scene_fails_the_same_check(self) -> None:
        """Proves the test above is not vacuous — and states the diagnosis.

        large_indoor is the scene under suspicion. Run the identical procedure
        on it and the second plan comes back empty: with one junction there is
        no second route, which is exactly the structural explanation for its
        long-gap revisits that this scenario was built to test against.
        """
        cross = mujoco.MjModel.from_xml_path(str(CROSS_SCENE))
        grid = truth_grid(
            cross,
            MapConfig(
                resolution=RESOLUTION,
                origin_x=ORIGIN,
                origin_y=ORIGIN,
                grid_width=CELLS,
                grid_height=CELLS,
            ),
        )
        planner = AStarPlanner(clearance_radius=CLEARANCE)
        start = grid.world_to_grid(12.0, 19.0)  # NE-far room
        goal = grid.world_to_grid(-12.0, -19.0)  # SW-far room

        first = planner.plan(grid, start, goal)
        assert first is not None

        assert planner.plan(block_middle(grid, first), start, goal) is None


class TestConfig:
    """The scenario config matches the scene it points at."""

    def test_config_extents_cover_the_whole_floor(self) -> None:
        """Grid extent, origin and the scene's 50 m floor must agree.

        Deliberately identical to large_indoor's, so coverage and timing
        figures from the two scenes are directly comparable.
        """
        config = load_config(CONFIG)
        map_cfg = config.map

        assert map_cfg.grid_width * map_cfg.resolution == 50.0
        assert map_cfg.grid_height * map_cfg.resolution == 50.0
        assert map_cfg.origin_x == ORIGIN
        assert map_cfg.origin_y == ORIGIN
        assert config.scene_path == "loop_indoor.xml"

    def test_config_clearance_matches_what_the_doorways_were_sized_for(self) -> None:
        """The scene's widths are only correct for this clearance value."""
        config = load_config(CONFIG)

        assert config.planning.clearance_radius == CLEARANCE

    def test_config_spawns_are_the_ones_the_tests_check(self) -> None:
        """The spawns the other tests use are the mission's real ones.

        The reachability and clearance checks above are only worth anything if
        they run on the positions the mission actually launches from.
        """
        config = load_config(CONFIG)

        assert tuple((x, y) for x, y, _z in config.drones.start_positions) == SPAWNS
