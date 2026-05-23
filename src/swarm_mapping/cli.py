"""CLI entry point for swarm-mapping.

Runs the end-to-end mapping pipeline: load scene, patrol with
a single drone, scan at each waypoint, build occupancy map, export.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.export import save_npz, save_png
from swarm_mapping.mapping.mapper import Mapper
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.rangefinder import Rangefinder
from swarm_mapping.simulation.engine import SimulationEngine

logger = logging.getLogger(__name__)

# Path to the simulation assets directory
_ASSETS_DIR = Path(__file__).parent / "simulation" / "assets"


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


def run_pipeline(config_path: Path, output_dir: Path) -> None:
    """Run the full mapping pipeline.

    Args:
        config_path: Path to the scenario YAML config.
        output_dir: Directory for output files (.npz, .png).
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

    # Tick loop: teleport → scan → map
    for i, wp in enumerate(waypoints):
        engine.set_drone_position(0, np.array(wp, dtype=np.float64))
        scan = sensor.scan(0)
        mapper.integrate_scan(scan)

        if (i + 1) % 20 == 0:
            logger.info("Waypoint %d/%d", i + 1, len(waypoints))

    # Export results
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_pipeline(args.config, args.output)


if __name__ == "__main__":
    main()
