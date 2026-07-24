"""Live MuJoCo viewer for human monitoring.

The ``Renderer`` protocol is the visualization seam named in the project
architecture: a read-only consumer of simulation state. ``LiveViewer`` is
its first concrete implementation, wrapping MuJoCo's passive viewer so a
human can watch the drone fly through the scene while a mission runs.

The viewer is strictly a monitor. It never advances the simulation or
mutates state itself — the caller owns the tick loop and pushes frames by
calling :meth:`Renderer.sync`. This keeps the deterministic core untouched:
a run with ``--view`` produces the same map as one without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mujoco
import mujoco.viewer

if TYPE_CHECKING:
    from types import TracebackType


class Renderer(Protocol):
    """Read-only viewer of live simulation state.

    Implementations display the current MuJoCo state to a human. They are
    passive: the caller drives the simulation and calls :meth:`sync` to
    refresh the display. A headless PNG snapshotter or a null test-double
    would be additional implementations.
    """

    @property
    def is_running(self) -> bool:
        """Whether the viewer is still open and accepting frames."""
        ...

    def sync(self) -> None:
        """Push the current simulation state to the display."""
        ...

    def close(self) -> None:
        """Close the viewer and release its resources."""
        ...


class LiveViewer:
    """Interactive 3D viewer backed by ``mujoco.viewer.launch_passive``.

    Opens a window rendering the shared ``MjModel``/``MjData`` handed in by
    the simulation engine. Because the viewer reads the *same* ``MjData`` the
    tick loop mutates, refreshing the display is just a :meth:`sync` call —
    no state is copied and none is written back.

    Usable as a context manager so the window is always closed, even if the
    mission loop raises::

        with LiveViewer(engine.model, engine.data) as viewer:
            while viewer.is_running:
                ...
                viewer.sync()

    Args:
        model: The MuJoCo model to render.
        data: The MuJoCo data whose state is displayed on each sync.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # launch_passive returns a handle that renders on the calling thread;
        # we sync it explicitly rather than letting it spin its own loop, which
        # keeps rendering in lock-step with the deterministic tick loop.
        self._handle = mujoco.viewer.launch_passive(model, data)

    @property
    def is_running(self) -> bool:
        """Whether the viewer window is still open.

        Returns ``False`` once the user closes the window, letting the tick
        loop stop paying the rendering cost while still finishing the mission.
        """
        return bool(self._handle.is_running())

    def set_camera(
        self,
        *,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: tuple[float, float, float],
    ) -> None:
        """Position the free camera.

        Set explicitly rather than via the model's ``<visual><global>`` defaults
        because the passive viewer ignores those on launch and opens at floor
        level, pointing at the floor.

        Args:
            azimuth: Horizontal camera angle in degrees.
            elevation: Vertical camera angle in degrees (negative looks down).
            distance: Camera distance from the look-at point, in meters.
            lookat: World-coordinate point the camera orbits and faces.
        """
        cam = self._handle.cam
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        cam.lookat[:] = lookat

    def sync(self) -> None:
        """Redraw the window from the current ``MjData`` state."""
        self._handle.sync()

    def close(self) -> None:
        """Close the viewer window. Idempotent."""
        self._handle.close()

    def __enter__(self) -> LiveViewer:
        """Enter the context manager, returning self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the viewer on context exit."""
        self.close()
