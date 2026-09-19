"""CLI entry point for swarm-mapping.

Runs the end-to-end exploration pipeline: load and validate the scenario,
build the swarm, hand it to `CentralizedMaster`, tick until the mission
terminates, then export the 2.5D map as `.npz` + `.png`.

This module owns the mission loop rather than the coordinator, deliberately:
`Coordinator.tick()` advances exactly one step so the live viewer can render
between ticks and tests can inspect intermediate state. The `while` around it,
the tick cap, and the reporting are CLI concerns.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarm_mapping.config.loader import load_config
from swarm_mapping.config.schema import ScenarioConfig
from swarm_mapping.coordination.master import CentralizedMaster
from swarm_mapping.mapping.export import save_npz, save_png
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.planning.frontier_strategy import NearestFrontier
from swarm_mapping.planning.path_planner import AStarPlanner
from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.visualization.renderer import LiveViewer

logger = logging.getLogger(__name__)

# Path to the simulation assets directory.
_ASSETS_DIR = Path(__file__).parent / "simulation" / "assets"

# Live-viewer pacing (used only with --view; has no effect on map output).
# Drones move one cell per tick — 0.1-0.2 m — so no interpolation is needed to
# animate smoothly; one sync per tick is the animation.
_FRAME_DELAY_S = 0.02  # wall-clock pause per frame (~50 fps)

# Cells outside this probability band are classified; cells inside are unknown.
# Matches the free/occupied thresholds the exporter and KPIs use.
_UNKNOWN_LOW = 0.4
_UNKNOWN_HIGH = 0.6

# Camera framing for --view: pull back far enough to see the whole grid.
_VIEW_MARGIN = 1.2


@dataclass(frozen=True)
class MissionResult:
    """What a finished mission is worth reporting.

    Returned by `run_pipeline` rather than printed, because three tests assert
    on mission *outcome* and parsing stdout for it would be the wrong seam.

    Attributes:
        ticks: Ticks executed.
        coverage: Fraction of grid cells classified free or occupied.
        blocked: True if the mission terminated with frontiers it could not
            reach — walled out rather than finished.
        unreachable_frontiers: Frontier regions still detected at termination.
        tick_capped: True if the mission stopped because it hit `max_ticks`
            rather than because the coordinator was done.
        npz_path: Where the map data was written.
        png_path: Where the map image was written.
    """

    ticks: int
    coverage: float
    blocked: bool
    unreachable_frontiers: int
    tick_capped: bool
    npz_path: Path
    png_path: Path

    @property
    def succeeded(self) -> bool:
        """True if the mission reached its own terminal state.

        Deliberately **not** `not blocked and not tick_capped`. Body-clearance
        inflation always leaves a few wall-adjacent frontiers that can be seen
        but not occupied, so `blocked` is True even on a complete run — on
        `small_indoor` a single drone finishes at 98.1% coverage with 3 such
        regions left. Failing on `blocked` would exit non-zero on the happy
        path. Only the tick cap means the mission genuinely did not converge;
        `blocked` is reported for the operator to read against coverage.
        """
        return not self.tick_capped


@dataclass(frozen=True)
class Mission:
    """A wired-up, ready-to-tick mission.

    Bundled so the acceptance suite can drive the tick loop itself — measuring
    coverage per tick for the scaling KPI, or checking separation on every tick
    — without restating the wiring that `run_pipeline` performs. Two copies of
    that wiring would be two things to keep in step.

    Attributes:
        engine: The simulation engine.
        mapper: The shared map every drone contributes to.
        master: The coordinator driving the mission.
        config: The validated scenario this was built from.
    """

    engine: SimulationEngine
    mapper: Mapper
    master: CentralizedMaster
    config: ScenarioConfig


def select_start_positions(
    positions: tuple[tuple[float, float, float], ...],
    count: int | None,
) -> tuple[tuple[float, float, float], ...]:
    """Apply the `--drones` override to the configured start positions.

    Takes the *first* `count` positions, so a smaller swarm is a prefix of a
    larger one and the two runs differ in nothing but drone count. That is what
    makes the scaling KPI a fair comparison.

    Args:
        positions: Start positions from the config, in order.
        count: Requested drone count, or None to use the config unchanged.

    Returns:
        The positions to launch with.

    Raises:
        ValueError: If `count` is below 1, or exceeds the configured positions.
            The CLI never invents a position: a fabricated one has to be
            collision-free, inside the grid, and `min_separation` clear of its
            neighbours, and guessing at that would set a safety parameter by
            accident.
    """
    if count is None:
        return positions
    if count < 1:
        msg = f"--drones must be at least 1, got {count}"
        raise ValueError(msg)
    if count > len(positions):
        msg = (
            f"--drones {count} exceeds the {len(positions)} start position(s) "
            f"in the config. Add positions to the scenario rather than having "
            f"the CLI invent them — a start position must be collision-free, "
            f"inside the grid, and clear of its neighbours."
        )
        raise ValueError(msg)
    return positions[:count]


def coverage_fraction(grid: OccupancyGrid) -> float:
    """Fraction of grid cells confidently classified as free or occupied.

    Cells inside the unknown band have been observed but not resolved; they do
    not count, because the map cannot classify them.

    Args:
        grid: The occupancy grid to measure.

    Returns:
        Coverage as a fraction in [0, 1].
    """
    prob = grid.probability()
    classified = int(np.sum((prob < _UNKNOWN_LOW) | (prob > _UNKNOWN_HIGH)))
    return classified / prob.size


def resolve_scene_path(scene_path: str) -> Path:
    """Resolve a config's scene reference to a file on disk.

    An absolute path is used as-is, so tests (and one-off scenes) need not be
    copied into the package's asset directory. A relative path resolves under
    the bundled assets, which is what every shipped scenario uses.

    Args:
        scene_path: The `scene.path` value from the config.

    Returns:
        The resolved path to the MJCF file.
    """
    candidate = Path(scene_path)
    return candidate if candidate.is_absolute() else _ASSETS_DIR / candidate


def build_mission(config: ScenarioConfig, drones: int | None = None) -> Mission:
    """Wire a validated config into a ready-to-tick mission.

    Args:
        config: The validated scenario configuration.
        drones: Optional `--drones` override on the swarm size.

    Returns:
        The assembled mission.

    Raises:
        ValueError: If the drone override is out of range, or if a safety
            parameter fails the physical floors enforced by `Rangefinder`,
            `AStarPlanner` and `CentralizedMaster`.
    """
    start_positions = select_start_positions(config.drones.start_positions, drones)

    engine = SimulationEngine(
        resolve_scene_path(config.scene_path),
        {
            drone_id: np.array(position, dtype=np.float64)
            for drone_id, position in enumerate(start_positions)
        },
    )

    sensor = Rangefinder(
        engine,
        num_rays=config.sensor.num_rays,
        max_range=config.sensor.max_range,
        exclusion_radius=config.sensor.exclusion_radius,
    )

    mapper = Mapper(
        MapConfig(
            resolution=config.map.resolution,
            origin_x=config.map.origin_x,
            origin_y=config.map.origin_y,
            grid_width=config.map.grid_width,
            grid_height=config.map.grid_height,
        )
    )

    # One planner object, shared: the strategy plans with it and the master
    # re-validates committed paths against it. CentralizedMaster rejects a
    # mismatch, because a silent disagreement about which cells the body may
    # occupy is exactly the bug that guard exists to catch.
    planner = AStarPlanner(clearance_radius=config.planning.clearance_radius)
    strategy = NearestFrontier(
        planner,
        spread_radius=config.planning.spread_radius,
        spread_penalty=config.planning.spread_penalty,
    )

    master = CentralizedMaster(
        engine=engine,
        sensor=sensor,
        mapper=mapper,
        strategy=strategy,
        planner=planner,
        altitude=config.drones.altitude,
        min_separation=config.coordination.min_separation,
        max_wait_ticks=config.coordination.max_wait_ticks,
    )

    return Mission(engine=engine, mapper=mapper, master=master, config=config)


def _open_viewer(mission: Mission) -> LiveViewer:
    """Open the live viewer, framed to the whole grid.

    The camera is set on the handle rather than in the MJCF because the passive
    viewer ignores the model's `<global>` defaults and opens at floor level.
    Distance is derived from the grid extent so a 50x50 m scene is framed as
    well as a 20x20 m one.
    """
    extent = mission.config.map
    span = max(extent.max_x - extent.origin_x, extent.max_y - extent.origin_y)
    viewer = LiveViewer(mission.engine.model, mission.engine.data)
    viewer.set_camera(
        azimuth=130.0,
        elevation=-40.0,
        distance=span * _VIEW_MARGIN,
        lookat=(
            (extent.origin_x + extent.max_x) / 2,
            (extent.origin_y + extent.max_y) / 2,
            1.0,
        ),
    )
    return viewer


def run_pipeline(
    config_path: Path,
    output_dir: Path,
    view: bool = False,
    drones: int | None = None,
) -> MissionResult:
    """Run the full exploration pipeline and export the map.

    Args:
        config_path: Path to the scenario YAML config.
        output_dir: Directory for output files (.npz, .png).
        view: If True, open a live MuJoCo 3D viewer. View-only — the exported
            map is identical whether or not this is enabled.
        drones: Optional override on the swarm size; takes the first N
            configured start positions.

    Returns:
        The mission summary.

    Raises:
        ValueError: If the config fails validation or the drone override is
            out of range.
    """
    config = load_config(config_path)
    mission = build_mission(config, drones)
    master = mission.master
    max_ticks = config.coordination.max_ticks

    logger.info(
        "Starting exploration: %d drone(s), max %d ticks",
        len(mission.engine.drone_ids),
        max_ticks,
    )

    viewer = _open_viewer(mission) if view else None

    try:
        while not master.is_complete and master.tick_count < max_ticks:
            master.tick()

            if viewer is not None and viewer.is_running:
                viewer.sync()
                time.sleep(_FRAME_DELAY_S)

            if master.tick_count % 50 == 0:
                logger.info(
                    "Tick %d/%d — coverage %.1f%%",
                    master.tick_count,
                    max_ticks,
                    100.0 * coverage_fraction(mission.mapper.grid),
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "map.npz"
        png_path = output_dir / "map.png"
        save_npz(mission.mapper.grid, npz_path)
        save_png(mission.mapper.grid, png_path, max_height=config.map.max_height)

        result = MissionResult(
            ticks=master.tick_count,
            coverage=coverage_fraction(mission.mapper.grid),
            blocked=master.is_blocked,
            unreachable_frontiers=master.unreachable_frontiers,
            # `is_complete` is the coordinator's own terminal signal. If the
            # loop stopped without it, the cap is what stopped us.
            tick_capped=not master.is_complete,
            npz_path=npz_path,
            png_path=png_path,
        )

        # Keep the window open after the mission so the finished map can be
        # inspected and orbited. Sync continuously (not just on state change)
        # to keep the passive viewer responsive.
        if viewer is not None and viewer.is_running:
            print("Close the viewer window to exit.")
            while viewer.is_running:
                viewer.sync()
                time.sleep(_FRAME_DELAY_S)

        return result
    finally:
        if viewer is not None:
            viewer.close()


def _report(result: MissionResult) -> None:
    """Print a mission summary that distinguishes finishing from giving up."""
    if result.tick_capped:
        outcome = f"STOPPED at the {result.ticks}-tick cap — mission incomplete"
    elif result.blocked:
        outcome = (
            f"BLOCKED after {result.ticks} ticks — "
            f"{result.unreachable_frontiers} frontier region(s) unreachable"
        )
    else:
        outcome = f"COMPLETE after {result.ticks} ticks"

    print(outcome)
    print(f"Coverage: {result.coverage:.1%} of cells classified")
    print(f"Output: {result.npz_path}, {result.png_path}")


def main() -> None:
    """Parse arguments, run the mission, and exit non-zero if it fell short."""
    parser = argparse.ArgumentParser(
        description="Swarm mapping — drone exploration and 2.5D mapping",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to scenario YAML config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for map files (default: output/)",
    )
    parser.add_argument(
        "--drones",
        type=int,
        default=None,
        help="Override the swarm size, using the first N start positions from "
        "the config (default: every position in the config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open a live MuJoCo 3D viewer and watch the swarm explore "
        "(view-only; does not affect the exported map)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = run_pipeline(args.config, args.output, view=args.view, drones=args.drones)
    _report(result)

    # A mission that never converged exits non-zero. `blocked` alone does not
    # fail the run — see MissionResult.succeeded for why that would fail the
    # happy path — but it is printed above, with its frontier count, so a
    # genuinely walled-out run is visible rather than silently exiting 0.
    if not result.succeeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
