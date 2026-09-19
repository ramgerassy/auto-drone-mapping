"""Ground-truth occupancy from a parsed MuJoCo scene.

Builds the map a perfect sensor would produce, straight from the scene's box
geometry, so tests can measure what the swarm *should* have mapped against what
it did. Used by the large-indoor geometry tests and by the acceptance suite's
map-accuracy check.

Everything here reads the **parsed model**. A helper that re-parsed the MJCF
text would prove nothing about what MuJoCo actually loaded — which is the whole
point of checking a scene this way.
"""

from __future__ import annotations

import mujoco
import numpy as np

from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.types import MapConfig

# Log-odds written into the truth grid. Magnitude only has to clear the
# 0.4/0.6 classification bands; +-2.0 gives p = 0.88 / 0.12.
_FREE, _OCCUPIED = -2.0, 2.0


def box_geoms(
    model: mujoco.MjModel,
) -> list[tuple[str, float, float, float, float]]:
    """Every box geom as ``(name, pos_x, pos_y, half_x, half_y)``.

    Args:
        model: A parsed MuJoCo model.

    Returns:
        One tuple per box geom. Planes (the floor) and any non-box geometry are
        skipped — the map is 2.5D and only vertical obstacles matter.
    """
    out: list[tuple[str, float, float, float, float]] = []
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        pos, size = model.geom_pos[i], model.geom_size[i]
        out.append((name, float(pos[0]), float(pos[1]), float(size[0]), float(size[1])))
    return out


def truth_grid(model: mujoco.MjModel, config: MapConfig) -> OccupancyGrid:
    """The occupancy grid a perfect sensor would produce for this scene.

    A cell is occupied when its extent overlaps a box geom's — the same
    conservative rule the mapper arrives at when a ray endpoint lands inside a
    cell, without needing to run a scan. Every other cell is free.

    Note this deliberately marks wall *interiors* occupied, which a real scan
    cannot do: rays stop at the surface, so cells behind it stay unknown
    forever. Comparing a real map against this one therefore has to account for
    unobservable interior cells rather than counting them as errors.

    Args:
        model: A parsed MuJoCo model.
        config: The grid geometry to build against.

    Returns:
        A grid with every cell classified free or occupied — never unknown.
    """
    grid = OccupancyGrid(config)
    grid.log_odds[:] = _FREE
    half = config.resolution / 2

    for _, px, py, sx, sy in box_geoms(model):
        col_lo = max(0, int(np.floor((px - sx - config.origin_x) / config.resolution)))
        col_hi = min(
            config.grid_width - 1,
            int(np.ceil((px + sx - config.origin_x) / config.resolution)),
        )
        row_lo = max(0, int(np.floor((py - sy - config.origin_y) / config.resolution)))
        row_hi = min(
            config.grid_height - 1,
            int(np.ceil((py + sy - config.origin_y) / config.resolution)),
        )
        for col in range(col_lo, col_hi + 1):
            for row in range(row_lo, row_hi + 1):
                cell_x, cell_y = grid.grid_to_world(col, row)
                if abs(cell_x - px) < sx + half and abs(cell_y - py) < sy + half:
                    grid.log_odds[row, col] = _OCCUPIED
    return grid
