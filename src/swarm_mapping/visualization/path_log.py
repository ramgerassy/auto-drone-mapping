"""Per-drone path recording, for diagnosing how a swarm divides its work.

The visit heatmap answers *how often* a cell was entered. It cannot answer
*when*, and "when" is what separates a drone sweeping a region once from one
crossing it four times on its way elsewhere — which is exactly the difference
between a good and a bad allocation.

So a track records the drone's cell at every tick, in order. From that single
list both views follow: the **route** (the ordered cells, as flown) and the
**visit index** (per cell, the ticks it was entered). Storing the sequence and
deriving the index, rather than the other way round, keeps the recording a
single append per drone per tick and loses nothing — the index is a regrouping
of the same data.

The index is a sparse dict rather than a grid of lists: a 250x250 scene is
62,500 cells of which a drone touches ~1,200, so an array would be 98% empty
lists.

Lives in `visualization` because it is read-only consumption of simulation
state for a human to inspect, which is that module's stated job. It imports no
other domain module, so recording cannot perturb what it measures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

Cell = tuple[int, int]


@dataclass
class DroneTrack:
    """Every cell one drone occupied, in tick order.

    Attributes:
        drone_id: Which drone this track belongs to.
        cells: The drone's cell at each recorded tick. Index is the tick
            offset from the first recording, so `cells[i]` is where it was on
            tick `i`.
    """

    drone_id: int
    cells: list[Cell] = field(default_factory=list)

    def route(self) -> list[Cell]:
        """The path as flown, with consecutive repeats collapsed.

        A drone that holds station for twenty ticks contributes one entry, not
        twenty: the route is about *where it went*, and a stationary run is one
        place. Use `cells` directly when the timing matters.

        Returns:
            Ordered cells, each differing from the one before it.
        """
        out: list[Cell] = []
        for cell in self.cells:
            if not out or out[-1] != cell:
                out.append(cell)
        return out

    def visits(self) -> dict[Cell, list[int]]:
        """Ticks at which each cell was entered.

        A cell entered on ticks 20 and 90 maps to `[20, 90]`. Only *entries*
        count — consecutive ticks in one cell record the first — so the list
        length is the number of separate visits, not the dwell time.

        Returns:
            Cell to ascending tick list. Sparse: untouched cells are absent.
        """
        index: dict[Cell, list[int]] = defaultdict(list)
        previous: Cell | None = None
        for tick, cell in enumerate(self.cells):
            if cell != previous:
                index[cell].append(tick)
            previous = cell
        return dict(index)

    def revisited(self) -> dict[Cell, list[int]]:
        """Only the cells entered more than once.

        Returns:
            The subset of `visits` with two or more entries.
        """
        return {cell: ticks for cell, ticks in self.visits().items() if len(ticks) > 1}

    def gap_summary(self) -> list[tuple[Cell, int]]:
        """Revisited cells with the longest gap between first and last entry.

        A short gap is a drone turning around in place; a long one means it
        left, went elsewhere, and came back — which is the wasteful kind and
        the kind an allocation change should remove. Sorted longest first, so
        the worst offenders read off the top.

        Returns:
            `(cell, gap_in_ticks)` pairs, descending by gap.
        """
        gaps = [
            (cell, ticks[-1] - ticks[0]) for cell, ticks in self.revisited().items()
        ]
        return sorted(gaps, key=lambda item: (-item[1], item[0]))


@dataclass
class PathLog:
    """Tracks for every drone in one mission.

    Attributes:
        tracks: Track per drone id.
    """

    tracks: dict[int, DroneTrack] = field(default_factory=dict)

    def record(self, states: Mapping[int, object]) -> None:
        """Append the current cell of every drone.

        Args:
            states: The coordinator's `drone_states`. Only `.cell` is read, so
                this does not couple the log to the rest of `DroneState`.
        """
        for drone_id, state in states.items():
            cell = state.cell  # type: ignore[attr-defined]
            track = self.tracks.get(drone_id)
            if track is None:
                track = DroneTrack(drone_id=drone_id)
                self.tracks[drone_id] = track
            track.cells.append((int(cell[0]), int(cell[1])))

    def exclusive_fraction(self) -> float:
        """Share of visited cells that exactly one drone ever entered.

        The division-of-labour measure: 1.0 means no drone ever set foot where
        another had been, 0.5 means half the explored area was covered twice
        over by different drones.

        Returns:
            The fraction, or 0.0 if nothing was visited.
        """
        owners: dict[Cell, set[int]] = defaultdict(set)
        for drone_id, track in self.tracks.items():
            for cell in track.cells:
                owners[cell].add(drone_id)
        if not owners:
            return 0.0
        alone = sum(1 for holders in owners.values() if len(holders) == 1)
        return alone / len(owners)

    def workload_balance(self) -> float:
        """How evenly the drones split the ground, as smallest / largest.

        1.0 is a perfectly even split; 0.5 means one drone covered twice what
        another did.

        Prefer this to `exclusive_fraction` when comparing allocations.
        Exclusivity is **confounded by coverage**: a run that explores less has
        fewer chances to overlap, so a swarm that quits early scores high.
        Measured on large_indoor, global allocation scored 98.2% exclusive —
        the best of four variants — while visiting 2165 cells against the
        baseline's 2919 and splitting them 414 / 656 / 1133, which is the worst
        balance of the four. Exclusivity said it divided the work best; it had
        in fact divided it worst and stopped early.

        Returns:
            The ratio, or 0.0 if any drone visited nothing.
        """
        counts = [len(track.visits()) for track in self.tracks.values()]
        if not counts or min(counts) == 0:
            return 0.0
        return min(counts) / max(counts)

    def shared_cells(self) -> dict[Cell, list[int]]:
        """Cells more than one drone entered, and which drones those were.

        Returns:
            Cell to sorted drone ids, for cells with two or more visitors.
        """
        owners: dict[Cell, set[int]] = defaultdict(set)
        for drone_id, track in self.tracks.items():
            for cell in track.cells:
                owners[cell].add(drone_id)
        return {
            cell: sorted(holders)
            for cell, holders in owners.items()
            if len(holders) > 1
        }

    def summary(self) -> dict[int, dict[str, int | float]]:
        """Per-drone headline figures.

        Returns:
            Drone id to a dict of route length, distinct cells, revisited
            cells, and the longest revisit gap.
        """
        out: dict[int, dict[str, int | float]] = {}
        for drone_id, track in sorted(self.tracks.items()):
            gaps = track.gap_summary()
            out[drone_id] = {
                "ticks": len(track.cells),
                "route_length": len(track.route()),
                "distinct_cells": len(track.visits()),
                "revisited_cells": len(track.revisited()),
                "longest_gap": gaps[0][1] if gaps else 0,
            }
        return out


def save_path_log(log: PathLog, path: str | Path) -> None:
    """Write a log as JSON, keyed by drone.

    Cells become `"col,row"` strings because JSON object keys must be strings;
    the route is kept as `[col, row]` pairs so it can be plotted directly.

    Args:
        log: The recorded log.
        path: Destination `.json` path.
    """
    payload = {
        "exclusive_fraction": round(log.exclusive_fraction(), 4),
        "drones": {
            str(drone_id): {
                "route": [list(cell) for cell in track.route()],
                "visits": {
                    f"{cell[0]},{cell[1]}": ticks
                    for cell, ticks in sorted(track.visits().items())
                },
            }
            for drone_id, track in sorted(log.tracks.items())
        },
        "summary": {str(k): v for k, v in log.summary().items()},
    }
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def iter_segments(route: Iterable[Cell]) -> Iterable[tuple[Cell, Cell]]:
    """Consecutive cell pairs along a route, for drawing it.

    Args:
        route: Ordered cells from `DroneTrack.route`.

    Yields:
        `(from_cell, to_cell)` pairs.
    """
    previous: Cell | None = None
    for cell in route:
        if previous is not None:
            yield previous, cell
        previous = cell
