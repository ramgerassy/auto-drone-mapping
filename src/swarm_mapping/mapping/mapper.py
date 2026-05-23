"""Mapper: integrates sensor scans into the occupancy grid."""

from __future__ import annotations

from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.raytrace import bresenham_2d
from swarm_mapping.mapping.types import MapConfig
from swarm_mapping.perception.types import RayObservation, ScanResult


class Mapper:
    """Integrates sensor scans into a 2.5D occupancy grid.

    For each ray observation in a scan, traces the ray through the
    grid using Bresenham's algorithm and applies Bayesian log-odds
    updates: free along the ray path, occupied at the hit endpoint.

    Args:
        config: Grid configuration.
    """

    def __init__(self, config: MapConfig) -> None:
        self._grid = OccupancyGrid(config)

    @property
    def grid(self) -> OccupancyGrid:
        """Read access to the current grid state."""
        return self._grid

    def integrate_scan(self, scan: ScanResult) -> None:
        """Integrate all observations from a single scan into the grid.

        Args:
            scan: A ScanResult containing ray observations to integrate.
        """
        for obs in scan.observations:
            self._integrate_observation(obs)

    def _integrate_observation(self, obs: RayObservation) -> None:
        """Integrate a single ray observation into the grid.

        For a HIT: trace from origin to hit_point. Mark cells along
        the ray as free, mark the endpoint as occupied with height.

        For a MISS: trace from origin for max_range. Mark all cells
        as free.

        Args:
            obs: A single ray observation.
        """
        # Project ray to 2D (drop z)
        start_x, start_y = obs.origin[0], obs.origin[1]

        if obs.distance is not None and obs.hit_point is not None:
            # HIT: trace to hit point
            end_x, end_y = obs.hit_point[0], obs.hit_point[1]
            hit_z = float(obs.hit_point[2])
            is_hit = True
        else:
            # MISS: trace for max_range along direction
            end_x = start_x + obs.direction[0] * obs.max_range
            end_y = start_y + obs.direction[1] * obs.max_range
            hit_z = 0.0
            is_hit = False

        # Convert to grid coordinates
        start_col, start_row = self._grid.world_to_grid(start_x, start_y)
        end_col, end_row = self._grid.world_to_grid(end_x, end_y)

        # Trace ray through grid
        cells = bresenham_2d(start_col, start_row, end_col, end_row)

        if not cells:
            return

        if is_hit:
            # All cells except the last are free
            for col, row in cells[:-1]:
                if self._grid.in_bounds(col, row):
                    self._grid.update_free(col, row)
            # Last cell is occupied
            last_col, last_row = cells[-1]
            if self._grid.in_bounds(last_col, last_row):
                self._grid.update_occupied(last_col, last_row, hit_z)
        else:
            # All cells are free
            for col, row in cells:
                if self._grid.in_bounds(col, row):
                    self._grid.update_free(col, row)
