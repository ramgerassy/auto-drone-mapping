"""Scripted drone failures — the simulator's half of Sprint 3.

Injection lives here and detection lives in `coordination`. The coordinator
never sees this schedule: a master that is told which drone failed has detected
nothing. It has to infer failure from missed heartbeats and unrealized moves,
the same evidence a real ground station has.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from swarm_mapping.simulation.engine import SimulationEngine
from swarm_mapping.simulation.types import FailureMode

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledFailure:
    """One failure to inject.

    Attributes:
        drone_id: The drone that fails.
        tick: The tick it fails at, applied before that tick runs.
        mode: How it fails.
    """

    drone_id: int
    tick: int
    mode: FailureMode


class FailureInjector:
    """Applies a failure schedule to the engine, tick by tick.

    Args:
        engine: The simulation to fail drones in.
        schedule: Failures to apply, in any order.

    Raises:
        ValueError: If the schedule names a drone that is not in this run.
            Fail fast: under a `--drones` override that drops the drone, the
            alternative is a failure scenario that quietly runs with no
            failure — and a recovery check that passes for the wrong reason.
    """

    def __init__(
        self, engine: SimulationEngine, schedule: Sequence[ScheduledFailure] = ()
    ) -> None:
        present = set(engine.drone_ids)  # membership only
        for failure in schedule:
            if failure.drone_id not in present:
                msg = (
                    f"failure schedule names drone {failure.drone_id}, which is "
                    f"not in this run (drones {sorted(present)})"
                )
                raise ValueError(msg)
        self._engine = engine
        self._pending = sorted(schedule, key=lambda f: (f.tick, f.drone_id))

    def apply(self, tick: int) -> list[ScheduledFailure]:
        """Fail every drone due at or before `tick` that has not failed yet.

        `<=` rather than `==` so a caller that skips a tick cannot skip a
        failure; each one still fires exactly once.

        Args:
            tick: The tick about to run.

        Returns:
            The failures applied now, in (tick, drone_id) order.
        """
        due = [f for f in self._pending if f.tick <= tick]
        self._pending = [f for f in self._pending if f.tick > tick]
        for failure in due:
            self._engine.fail_drone(failure.drone_id, failure.mode)
            _LOGGER.info(
                "failure_injected",
                extra={
                    "drone_id": failure.drone_id,
                    "tick": tick,
                    "mode": failure.mode.value,
                },
            )
        return due
