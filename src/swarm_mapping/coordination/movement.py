"""Collision-free movement resolution.

Pure logic: given where every drone is and where each wants to go, decide who
actually moves. Kept free of the simulator and the map so the trickiest rule in
`coordination` is testable without MuJoCo.

Right of way goes to the **higher drone id** — see `docs/progress.md`.
"""

from __future__ import annotations

from collections.abc import Mapping

from swarm_mapping.coordination.types import Cell


def resolve_moves(
    current: Mapping[int, Cell],
    desired: Mapping[int, Cell | None],
    min_separation_cells: float,
) -> dict[int, Cell]:
    """Decide each drone's end-of-tick cell, refusing moves that would collide.

    Drones are considered in **descending id order**, so the higher id reserves
    its cell first and therefore has right of way. A drone may advance only if
    its target is at least `min_separation_cells` from every other drone's
    reserved position; otherwise it holds station.

    The reservation table starts at every drone's *current* cell, which is what
    makes this one rule sufficient. An unprocessed drone still occupies its own
    cell, so a pair cannot swap through each other — the classic edge conflict
    is blocked without a separate check for it.

    The rule is deliberately conservative: a drone may wait on a neighbour that
    was about to move away anyway. At 1-5 drones that costs an occasional tick
    and buys a much simpler invariant.

    Args:
        current: Each drone's current (col, row) cell.
        desired: Each drone's intended next cell, or None if it has nowhere to
            go (idle, or already arrived).
        min_separation_cells: Required centre-to-centre spacing, in cells.

    Returns:
        Each drone's cell at the end of the tick — the target for drones that
        advanced, the current cell for drones that held.
    """
    reserved = dict(current)
    threshold_sq = min_separation_cells * min_separation_cells

    for drone_id in sorted(current, reverse=True):
        target = desired.get(drone_id)
        if target is None or target == current[drone_id]:
            continue  # nowhere to go, or already there

        # sorted() only for legibility — this is a boolean test, so iteration
        # order cannot affect the outcome.
        blocked = False
        for other_id in sorted(reserved):
            if other_id == drone_id:
                continue
            other_col, other_row = reserved[other_id]
            d_col = target[0] - other_col
            d_row = target[1] - other_row
            if d_col * d_col + d_row * d_row < threshold_sq:
                blocked = True
                break

        if not blocked:
            reserved[drone_id] = target

    return reserved
