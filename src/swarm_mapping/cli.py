"""CLI entry point for swarm-mapping.

Runs the end-to-end mapping pipeline: load scene, patrol with
a single drone, scan at each waypoint, build occupancy map, export.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.export import save_npz, save_png
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.visualization.renderer import LiveViewer

logger = logging.getLogger(__name__)

# Path to the simulation assets directory
_ASSETS_DIR = Path(__file__).parent / "simulation" / "assets"

# Live-viewer pacing (used only with --view; has no effect on map output).
# Waypoints are far apart, so we teleport in small hops and pause briefly
# between them to animate a smooth flight rather than an instant jump.
_VIEW_STEP_M = 0.15  # max distance moved per rendered frame
_FRAME_DELAY_S = 0.02  # wall-clock pause per frame (~50 fps)


def generate_patrol(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    spacing: float,
    altitude: float,
) -> list[tuple[float, float, float]]:
    """Generate a lawnmower patrol pattern covering a rectangular area.

    Creates a zigzag path with the given spacing, alternating
    left-to-right and right-to-left passes.

    Args:
        x_min: Minimum x-coordinate of the patrol area.
        x_max: Maximum x-coordinate of the patrol area.
        y_min: Minimum y-coordinate of the patrol area.
        y_max: Maximum y-coordinate of the patrol area.
        spacing: Distance between waypoints in meters.
        altitude: Flight height (z-coordinate).

    Returns:
        Ordered list of (x, y, z) waypoints.
    """
    waypoints: list[tuple[float, float, float]] = []

    y_values = np.arange(y_min, y_max + spacing / 2, spacing)

    for i, y in enumerate(y_values):
        if i % 2 == 0:
            x_values = np.arange(x_min, x_max + spacing / 2, spacing)
        else:
            x_values = np.arange(x_max, x_min - spacing / 2, -spacing)

        for x in x_values:
            waypoints.append((float(x), float(y), altitude))

    return waypoints


def interpolate_segment(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    max_step: float,
) -> Iterator[NDArray[np.float64]]:
    """Yield intermediate positions from ``start`` toward ``end``.

    Points are evenly spaced so that no hop exceeds ``max_step`` meters. The
    start point is not emitted; the final emitted point equals ``end``. This
    exists purely to animate smooth motion in the live viewer — the mapping
    pipeline teleports directly to each waypoint and never calls this.

    Args:
        start: Current position, shape (3,).
        end: Target position, shape (3,).
        max_step: Maximum distance between consecutive emitted points.

    Yields:
        Intermediate positions along the segment, ending at ``end``.
    """
    delta = end - start
    dist = float(np.linalg.norm(delta))
    steps = max(1, int(np.ceil(dist / max_step)))
    for k in range(1, steps + 1):
        yield start + delta * (k / steps)


def run_pipeline(config_path: Path, output_dir: Path, view: bool = False) -> None:
    """Run the full mapping pipeline.

    Args:
        config_path: Path to the scenario YAML config.
        output_dir: Directory for output files (.npz, .png).
        view: If True, open a live MuJoCo 3D viewer and animate the drone
            flying between waypoints. View-only — the exported map is
            identical whether or not this is enabled.
    """
    config = load_config(config_path)

    # Extract config sections
    scene_cfg = config["scene"]
    drone_cfg = config["drone"]
    sensor_cfg = config["sensor"]
    map_cfg = config["map"]
    patrol_cfg = config["patrol"]

    # Resolve scene path
    scene_path = _ASSETS_DIR / scene_cfg["path"]

    # Setup simulation
    start_pos = np.array(drone_cfg["start_position"], dtype=np.float64)
    engine = SimulationEngine(scene_path, {0: start_pos})

    # Setup sensor
    sensor = Rangefinder(
        engine,
        num_rays=sensor_cfg["num_rays"],
        max_range=sensor_cfg["max_range"],
    )

    # Setup mapper
    map_config = MapConfig(
        resolution=map_cfg["resolution"],
        origin_x=map_cfg["origin_x"],
        origin_y=map_cfg["origin_y"],
        grid_width=map_cfg["grid_width"],
        grid_height=map_cfg["grid_height"],
    )
    mapper = Mapper(map_config)

    # Generate patrol waypoints
    x_range = patrol_cfg["x_range"]
    y_range = patrol_cfg["y_range"]
    waypoints = generate_patrol(
        x_min=x_range[0],
        x_max=x_range[1],
        y_min=y_range[0],
        y_max=y_range[1],
        spacing=patrol_cfg["spacing"],
        altitude=patrol_cfg["altitude"],
    )

    logger.info("Starting patrol with %d waypoints", len(waypoints))

    # Open the live viewer if requested. It reads the same MjData the loop
    # mutates, so a plain sync() reflects the drone's current position.
    viewer = LiveViewer(engine.model, engine.data) if view else None
    if viewer is not None:
        # Frame the room from above. Must be set on the handle: the passive
        # viewer ignores the model's <global> defaults and opens at floor level.
        viewer.set_camera(
            azimuth=130.0, elevation=-40.0, distance=30.0, lookat=(0.0, 0.0, 1.0)
        )
    prev_pos = start_pos.copy()

    try:
        # Tick loop: (optionally animate) → teleport → scan → map
        for i, wp in enumerate(waypoints):
            wp_arr = np.array(wp, dtype=np.float64)

            # Animate the flight from the previous waypoint. This only moves
            # the drone for rendering; no scanning or mapping happens here, so
            # the resulting map is unaffected. Stops early if the user closes
            # the window, letting the mission finish headless.
            if viewer is not None and viewer.is_running:
                for pos in interpolate_segment(prev_pos, wp_arr, _VIEW_STEP_M):
                    engine.set_drone_position(0, pos)
                    viewer.sync()
                    time.sleep(_FRAME_DELAY_S)

            engine.set_drone_position(0, wp_arr)
            scan = sensor.scan(0)
            mapper.integrate_scan(scan)

            if viewer is not None and viewer.is_running:
                viewer.sync()

            prev_pos = wp_arr

            if (i + 1) % 20 == 0:
                logger.info("Waypoint %d/%d", i + 1, len(waypoints))

        # Export results (always — independent of the viewer).
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "map.npz"
        png_path = output_dir / "map.png"

        save_npz(mapper.grid, npz_path)
        save_png(mapper.grid, png_path, max_height=map_cfg["max_height"])

        # Summary
        prob = mapper.grid.probability()
        mapped = int(np.sum((prob < 0.4) | (prob > 0.6)))
        total = prob.size
        coverage = 100.0 * mapped / total

        print(f"Patrol complete: {len(waypoints)} waypoints")
        print(f"Coverage: {mapped}/{total} cells ({coverage:.1f}%)")
        print(f"Output: {npz_path}, {png_path}")

        # Keep the window open after the patrol so the finished flight/map can
        # be inspected and orbited. Sync continuously (not just on state change)
        # to keep the passive viewer responsive. Exits when the user closes it.
        if viewer is not None and viewer.is_running:
            print("Close the viewer window to exit.")
            while viewer.is_running:
                viewer.sync()
                time.sleep(_FRAME_DELAY_S)
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    """Parse arguments and run the mapping pipeline."""
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
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open a live MuJoCo 3D viewer and watch the drone fly "
        "(view-only; does not affect the exported map)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_pipeline(args.config, args.output, view=args.view)


if __name__ == "__main__":
    main()
