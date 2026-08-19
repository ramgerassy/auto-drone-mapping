"""Pipeline tests for the teammate filter (Sprint 2, Feature 4b).

Real MuJoCo, real ray-casts, real mapper — the unit tests in
tests/unit/test_perception/test_teammate_filter.py pin the filter's contract;
these pin what it means for the map.

Two drones 2.0 m apart in line of sight. A head-on ray stops on the near face
of the teammate, 0.15 m short of its centre, so the assertions look at a
neighbourhood around the teammate rather than its centre cell alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.simulation.engine import SimulationEngine

pytestmark = pytest.mark.sprint(2)  # Feature 4b — perception teammate filter

ROOM_XML = """\
<mujoco model="teammate_filter_room">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <geom name="wall_east" type="box" pos="3 0 1" size="0.1 3 1"/>
    <geom name="wall_west" type="box" pos="-3 0 1" size="0.1 3 1"/>
    <geom name="wall_north" type="box" pos="0 3 1" size="3 0.1 1"/>
    <geom name="wall_south" type="box" pos="0 -3 1" size="3 0.1 1"/>
  </worldbody>
</mujoco>
"""

ALTITUDE = 1.0
SENSING = (-1.0, 0.0)
TEAMMATE = (1.0, 0.0)
RESOLUTION = 0.1
# Wide enough to cover the near face (0.15 m short of centre) and the corners.
NEIGHBOURHOOD = 0.3


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """Write the inline room to a file the engine can load."""
    path = tmp_path / "room.xml"
    path.write_text(ROOM_XML)
    return path


def map_one_scan(scene: Path, exclusion_radius: float | None = None) -> Mapper:
    """Scan once with drone 0 while drone 1 sits 2 m away, and map the result.

    Args:
        scene: The room MJCF.
        exclusion_radius: Passed through to Rangefinder; None uses the default.
            0.0 disables filtering, which is how the un-filtered case is built.
    """
    positions = {
        0: np.array([*SENSING, ALTITUDE]),
        1: np.array([*TEAMMATE, ALTITUDE]),
    }
    engine = SimulationEngine(scene, positions)
    kwargs = {} if exclusion_radius is None else {"exclusion_radius": exclusion_radius}
    sensor = Rangefinder(engine, num_rays=72, max_range=12.0, **kwargs)
    mapper = Mapper(
        MapConfig(
            resolution=RESOLUTION,
            origin_x=-3.0,
            origin_y=-3.0,
            grid_width=60,
            grid_height=60,
        )
    )
    mapper.integrate_scan(sensor.scan(0))
    return mapper


def cells_near(
    mapper: Mapper, x: float, y: float, radius: float
) -> list[tuple[int, int]]:
    """Grid cells whose centres lie within `radius` of a world point."""
    grid = mapper.grid
    span = int(np.ceil(radius / RESOLUTION))
    centre_col, centre_row = grid.world_to_grid(x, y)
    found = []
    for col in range(centre_col - span, centre_col + span + 1):
        for row in range(centre_row - span, centre_row + span + 1):
            if not grid.in_bounds(col, row):
                continue
            wx, wy = grid.grid_to_world(col, row)
            if np.hypot(wx - x, wy - y) <= radius:
                found.append((col, row))
    return found


class TestTeammateIsNotMapped:
    """The teammate leaves no trace in either layer."""

    @pytest.mark.sanity
    def test_no_occupied_cell_at_the_teammate(self, scene: Path) -> None:
        """Case 9: nothing around the teammate is classified occupied."""
        mapper = map_one_scan(scene)
        prob = mapper.grid.probability()

        occupied = [
            cell
            for cell in cells_near(mapper, *TEAMMATE, NEIGHBOURHOOD)
            if prob[cell[1], cell[0]] > 0.6
        ]

        assert occupied == []

    def test_height_layer_is_untouched(self, scene: Path) -> None:
        """Case 10: the consequence with no recovery path.

        `update_occupied` does `height = max(height, hit_z)` — monotonic, no
        decay. A teammate written in at 1.0 m would sit in a floor cell for the
        rest of the mission even after the occupancy washed out.
        """
        mapper = map_one_scan(scene)
        height = mapper.grid.height

        for col, row in cells_near(mapper, *TEAMMATE, NEIGHBOURHOOD):
            assert height[row, col] == -np.inf


class TestFreeSpaceIsPreserved:
    """Filtering keeps the evidence the ray legitimately gathered."""

    def test_space_between_the_drones_is_free(self, scene: Path) -> None:
        """Case 11: the ray did travel that far unobstructed.

        Asserted against unknown (0.5) rather than the 0.4 `free_threshold`:
        `log_odds_free = -0.405` puts a single free observation at p = 0.4001,
        so one scan never crosses the classification bar by design. This cell
        lies on the 0-degree ray alone, so it gets exactly one.
        """
        mapper = map_one_scan(scene)
        prob = mapper.grid.probability()
        col, row = mapper.grid.world_to_grid(0.0, 0.0)

        assert prob[row, col] < 0.5

    def test_space_beyond_the_teammate_stays_unknown(self, scene: Path) -> None:
        """Case 12: the occlusion shadow survives.

        A real LiDAR cannot see through a teammate. Dropping the ray entirely
        or emitting a full-range MISS would both mark this cell free; only the
        shortened MISS leaves it unknown.
        """
        mapper = map_one_scan(scene)
        prob = mapper.grid.probability()
        col, row = mapper.grid.world_to_grid(1.5, 0.0)

        assert prob[row, col] == pytest.approx(0.5)


class TestTheBugIsReal:
    """Without the filter, the same scene misbehaves."""

    def test_unfiltered_scan_maps_the_teammate_as_an_obstacle(
        self, scene: Path
    ) -> None:
        """Case 13: proves the assertions above are not vacuous.

        `exclusion_radius=0.0` disables the filter, and the teammate comes back
        as a solid obstacle with its altitude written into the height layer.
        """
        mapper = map_one_scan(scene, exclusion_radius=0.0)
        prob = mapper.grid.probability()
        height = mapper.grid.height

        cells = cells_near(mapper, *TEAMMATE, NEIGHBOURHOOD)
        occupied = [c for c in cells if prob[c[1], c[0]] > 0.6]

        assert occupied != [], "the teammate should have been mapped as an obstacle"
        assert any(height[row, col] == pytest.approx(ALTITUDE) for col, row in occupied)
