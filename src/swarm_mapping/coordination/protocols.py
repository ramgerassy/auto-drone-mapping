"""Protocol definitions for coordination module interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from swarm_mapping.coordination.types import DroneState


class Coordinator(Protocol):
    """Interface for orchestrating a mapping mission.

    One of the four named SOLID seams. The only implementation is
    `CentralizedMaster`; a `DistributedAuction` would be a second (explicitly
    not built — see CLAUDE.md).

    The caller owns the loop: `tick()` advances the mission by exactly one step
    so the live viewer can render between ticks and tests can inspect
    intermediate state. A `run()` convenience belongs in the CLI, not here.
    """

    @property
    def is_complete(self) -> bool:
        """True when no drone can make further progress.

        A *termination* signal, not a success one — it covers both "everything
        is mapped" and "nothing left is reachable". Pair it with `is_blocked`
        to tell the two apart.
        """
        ...

    @property
    def is_blocked(self) -> bool:
        """True if the mission terminated with frontiers still on the map.

        The question "did you finish or did you give up" is not specific to a
        centralized master — a `DistributedAuction` owes the same answer — so
        it belongs on the seam. A coordinator that can report only that it
        stopped, and not what it stopped with, is under-specified: a run that
        mapped 58% of a room and one that mapped all of it look identical from
        outside.

        **Not a pass/fail flag on its own.** Body-clearance inflation leaves
        wall-adjacent frontiers that are visible but impossible to occupy, so a
        fully successful mission normally terminates blocked. Callers must weigh
        it against `unreachable_frontiers` and coverage — see
        `CentralizedMaster.is_blocked` for the measured example.
        """
        ...

    @property
    def unreachable_frontiers(self) -> int:
        """How many frontier regions were still detected at termination.

        The magnitude behind `is_blocked`, and the number that makes it usable:
        "stopped with 3 regions left" and "stopped with 200" are different
        outcomes and want different responses.
        """
        ...

    @property
    def drone_states(self) -> Mapping[int, DroneState]:
        """Read-only view of drone state, including each drone's health.

        For visualization.
        """
        ...

    def tick(self) -> None:
        """Advance the mission by one tick."""
        ...
