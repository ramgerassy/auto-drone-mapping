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
        """True when no drone can make further progress."""
        ...

    @property
    def drone_states(self) -> Mapping[int, DroneState]:
        """Read-only view of drone state, for visualization."""
        ...

    def tick(self) -> None:
        """Advance the mission by one tick."""
        ...
