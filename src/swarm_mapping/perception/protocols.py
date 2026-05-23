"""Protocol definitions for perception module interfaces."""

from __future__ import annotations

from typing import Protocol

from swarm_mapping.perception.types import ScanResult


class Sensor(Protocol):
    """Interface for producing world-coordinate observations.

    The only implementation is Rangefinder (horizontal ray sweep).
    A depth camera or stereo pair would be a second implementation.
    """

    def scan(self, drone_id: int) -> ScanResult:
        """Perform a sensor scan for the given drone.

        Args:
            drone_id: Integer identifier for the drone.

        Returns:
            A ScanResult containing all ray observations from this scan.
        """
        ...
