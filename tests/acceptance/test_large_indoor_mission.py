"""Acceptance: the Sprint-2 KPIs, measured on a real scenario.

Minutes, not seconds — these drive full missions over a 250x250 grid. They
carry both `acceptance` (cost) and `sprint(2)` (era): CI runs them on the
progression gate for PRs to main, never on push. Deselect locally with
`-m "not acceptance"`.

Each test asserts a committed KPI from docs/sprint-2-plan.md and reports the
measured value, so a near-miss reads as a number rather than just red.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from swarm_mapping.cli import (
    Mission,
    build_mission,
    coverage_fraction,
    resolve_scene_path,
)
from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.types import MapConfig
from tests.scene_truth import truth_grid

pytestmark = [pytest.mark.sprint(2), pytest.mark.acceptance]

ROOT = Path(__file__).resolve().parents[2]
LARGE = ROOT / "scenarios" / "large_indoor" / "config.yaml"

# One probe per room, matching the Feature 5 geometry tests.
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


def fly(mission: Mission) -> int:
    """Run to completion or the config's tick cap; return the tick count."""
    cap = mission.config.coordination.max_ticks
    while not mission.master.is_complete and mission.master.tick_count < cap:
        mission.master.tick()
    return mission.master.tick_count


@pytest.fixture(scope="module")
def flown() -> tuple[Mission, int, set[tuple[int, int]]]:
    """One three-drone mission over large_indoor, shared by every test here.

    Module-scoped deliberately: this run is the most expensive thing in the
    suite, and every assertion below reads a different facet of the same
    mission. Visited cells are collected during the flight because they cannot
    be recovered afterwards.
    """
    config = load_config(LARGE)
    mission = build_mission(config)
    visited: set[tuple[int, int]] = set()
    cap = config.coordination.max_ticks
    while not mission.master.is_complete and mission.master.tick_count < cap:
        mission.master.tick()
        for state in mission.master.drone_states.values():
            visited.add(state.cell)
    return mission, mission.master.tick_count, visited


class TestCoverage:
    """Tier-1: coverage >= 95% indoor."""

    def test_coverage_meets_the_kpi(
        self, flown: tuple[Mission, int, set[tuple[int, int]]]
    ) -> None:
        """The headline exploration KPI."""
        mission, ticks, _ = flown
        coverage = coverage_fraction(mission.mapper.grid)

        assert coverage >= 0.95, f"coverage {coverage:.2%} after {ticks} ticks"

    def test_the_mission_terminates_rather_than_hitting_the_cap(
        self, flown: tuple[Mission, int, set[tuple[int, int]]]
    ) -> None:
        """Reaching `max_ticks` means the swarm never agreed it was finished.

        A capped run can still hit the coverage KPI while burning thousands of
        ticks on frontier churn, which is what `min_frontier_size` exists to
        prevent — so coverage alone would not catch a regression there.
        """
        mission, ticks, _ = flown

        assert mission.master.is_complete, f"hit the {ticks}-tick cap"


class TestRoomsAreExplored:
    """The scene exists to test doorway traversal, so corridors are not enough."""

    def test_every_room_is_fully_mapped(
        self, flown: tuple[Mission, int, set[tuple[int, int]]]
    ) -> None:
        """Each of the eight rooms is resolved, not just the corridor cross.

        Measured per room rather than globally: a 97% global figure could hide
        one room left entirely dark, which is exactly the failure a doorway one
        cell too narrow would produce.
        """
        mission, _, _ = flown
        grid = mission.mapper.grid
        prob = grid.probability()
        span = int(4.0 / grid.config.resolution)

        unmapped = {}
        for name, (x, y) in ROOMS.items():
            centre = grid.world_to_grid(x, y)
            cells = [
                (col, row)
                for col in range(centre[0] - span, centre[0] + span)
                for row in range(centre[1] - span, centre[1] + span)
                if grid.in_bounds(col, row)
            ]
            known = sum(
                1 for col, row in cells if prob[row, col] < 0.4 or prob[row, col] > 0.6
            )
            fraction = known / len(cells)
            if fraction < 0.95:
                unmapped[name] = fraction

        assert unmapped == {}

    def test_drones_pass_through_doorways(
        self, flown: tuple[Mission, int, set[tuple[int, int]]]
    ) -> None:
        """At least some rooms are physically entered, not all mapped from afar.

        Deliberately not "every room entered". The rangefinder reaches 12 m and
        a room is about 11 m across, so a drone parked in a doorway resolves the
        whole room without going in — asserting entry everywhere would measure
        sensor range rather than exploration. What this pins is that traversing
        a 1.2 m doorway works at all, which is the constriction the scene was
        built for and what clearance inflation could silently forbid.
        """
        mission, _, visited = flown
        grid = mission.mapper.grid
        span = int(3.0 / grid.config.resolution)

        entered = {
            name
            for name, (x, y) in ROOMS.items()
            if any(
                abs(col - grid.world_to_grid(x, y)[0]) <= span
                and abs(row - grid.world_to_grid(x, y)[1]) <= span
                for col, row in visited
            )
        }

        assert len(entered) >= 4, f"only entered {sorted(entered)}"


class TestMapAccuracy:
    """Tier-1: map accuracy >= 98% per-cell classification."""

    def test_classified_cells_match_ground_truth(
        self, flown: tuple[Mission, int, set[tuple[int, int]]]
    ) -> None:
        """Accuracy is measured over cells the map actually classified.

        Unknown cells are excluded rather than counted as errors: most of what
        stays unknown is wall *interior*, which no scan can resolve because
        rays stop at the surface. Scoring those as mistakes would measure the
        sensor model, not the map.
        """
        mission, _, _ = flown
        grid = mission.mapper.grid
        model = mujoco.MjModel.from_xml_path(
            str(resolve_scene_path(mission.config.scene_path))
        )
        truth = truth_grid(
            model,
            MapConfig(
                resolution=grid.config.resolution,
                origin_x=grid.config.origin_x,
                origin_y=grid.config.origin_y,
                grid_width=grid.config.grid_width,
                grid_height=grid.config.grid_height,
            ),
        )

        prob, truth_prob = grid.probability(), truth.probability()
        classified = (prob < 0.4) | (prob > 0.6)
        mapped_occupied = prob > 0.6
        truth_occupied = truth_prob > 0.6

        agree = int(np.sum(classified & (mapped_occupied == truth_occupied)))
        total = int(np.sum(classified))
        accuracy = agree / total

        assert accuracy >= 0.98, f"accuracy {accuracy:.2%} over {total} cells"


class TestZeroCollisions:
    """Tier-1: zero collisions during nominal operations."""

    def test_no_two_drones_ever_breach_separation(self) -> None:
        """Checked every tick, not just at the end.

        Runs its own mission rather than sharing the module fixture: the
        invariant is about every intermediate state, so it has to observe the
        loop rather than its result.
        """
        config = load_config(LARGE)
        mission = build_mission(config)
        grid = mission.mapper.grid
        minimum = config.coordination.min_separation
        cap = config.coordination.max_ticks

        worst = float("inf")
        while not mission.master.is_complete and mission.master.tick_count < cap:
            mission.master.tick()
            positions = [
                grid.grid_to_world(*s.cell)
                for s in mission.master.drone_states.values()
            ]
            for i, (ax, ay) in enumerate(positions):
                for bx, by in positions[i + 1 :]:
                    worst = min(worst, float(np.hypot(ax - bx, ay - by)))

        assert worst >= minimum, (
            f"closest approach {worst:.3f} m, below min_separation {minimum} m"
        )
