"""Ray-caster implementation using mujoco.mj_ray."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from swarm_mapping.simulation.types import RayHit


class MjRayCaster:
    """Casts rays into the MuJoCo scene and returns hit results.

    Wraps ``mujoco.mj_ray`` to cast rays from a given origin along
    specified direction vectors. Supports excluding a body (the
    drone itself) so rays don't self-intersect.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data (must stay in sync with model).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self._model = model
        self._data = data

    def cast_rays(
        self,
        origin: NDArray[np.float64],
        directions: NDArray[np.float64],
        body_exclude: int,
    ) -> list[RayHit | None]:
        """Cast rays from a point and return hit results.

        Args:
            origin: The 3D point to cast from, shape (3,).
            directions: Array of unit direction vectors, shape (N, 3).
            body_exclude: MuJoCo body ID to exclude from hits.

        Returns:
            List of length N. Each element is a RayHit if the ray hit
            a geometry, or None if it missed everything.
        """
        results: list[RayHit | None] = []
        geomid_out = np.array([-1], dtype=np.int32)

        for i in range(len(directions)):
            direction = directions[i]
            geomid_out[0] = -1

            distance = mujoco.mj_ray(
                self._model,
                self._data,
                origin,
                direction,
                None,  # geomgroup: include all groups
                1,  # flg_static: include static geoms
                body_exclude,
                geomid_out,
            )

            if distance < 0:
                results.append(None)
            else:
                hit_point = origin + direction * distance
                results.append(
                    RayHit(
                        distance=float(distance),
                        hit_point=hit_point.copy(),
                        geom_id=int(geomid_out[0]),
                    )
                )

        return results
