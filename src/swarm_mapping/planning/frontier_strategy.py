"""Frontier selection — deciding which frontier a drone explores next.

`mapping` detects and clusters frontier regions; this module *chooses* among
them. `NearestFrontier` scores each candidate by the true A* travel cost to it,
so walls — not straight-line distance — decide the winner, and an unreachable
frontier is skipped rather than assigned. A spatial spreading penalty keeps
multiple drones from converging on the same corner of the map.

Stateless: `select` is a pure function of its inputs. The coordinator owns the
claimed-frontier bookkeeping and passes it in.

Reads `mapping` read-only; `planning` never imports `coordination`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.planning.path_planner import Cell, PathPlanner, path_cost


@dataclass(frozen=True)
class FrontierAssignment:
    """A drone's next exploration target and the route to it.

    Attributes:
        region: The chosen frontier region.
        path: Cells from the drone's start cell to `region.cell` inclusive.
        cost: The route's true cost in the planner's 10/14 integer units. This
            is the *unpenalized* path cost — the spreading penalty affects
            ranking only, so this stays a truthful path length for logging and
            the path-length KPI.
    """

    region: FrontierRegion
    path: list[Cell]
    cost: int


class FrontierStrategy(Protocol):
    """Interface for choosing which frontier a drone should explore next.

    One of the four named SOLID seams. `NearestFrontier` is the primary
    implementation; `InformationGainFrontier` (weighting by region size, i.e.
    expected new information) is the stretch goal and would be the second.
    Consumers depend on this Protocol, never on the concrete class.
    """

    def select(
        self,
        grid: OccupancyGrid,
        frontiers: Sequence[FrontierRegion],
        start: Cell,
        claimed: Sequence[FrontierRegion] = (),
    ) -> FrontierAssignment | None:
        """Choose a frontier for the drone at `start`, or None if none suits."""
        ...


class NearestFrontier:
    """Selects the cheapest reachable unclaimed frontier by true path cost.

    Scoring runs the path planner against every candidate. That is more work
    than a straight-line estimate, but it is wall-aware (a frontier just behind
    a wall is genuinely far) and yields reachability for free — an unreachable
    frontier returns no path and is skipped, instead of being assigned and
    failing a tick later. With 1-5 drones and a handful of frontier regions the
    extra planning is not a bottleneck.

    Deliberately ignores `FrontierRegion.size`: weighting by expected
    information gain is what `InformationGainFrontier` is for.

    Args:
        planner: The path planner used to score and route to candidates.
            Injected so the strategy is testable against a stub and so
            swapping the planner never edits this class.
        spread_radius: World-frame radius, in metres, around a claimed
            frontier's centroid inside which candidates are penalized. 0
            disables the proximity penalty (hard exclusion still applies).
        spread_penalty: Cost units added to a candidate inside that radius.
            Additive and integer, so ranking stays in exact integer
            arithmetic — no float comparison in a decision path.
    """

    def __init__(
        self,
        planner: PathPlanner,
        spread_radius: float = 0.0,
        spread_penalty: int = 0,
    ) -> None:
        self._planner = planner
        self._spread_radius = spread_radius
        self._spread_penalty = spread_penalty

    def select(
        self,
        grid: OccupancyGrid,
        frontiers: Sequence[FrontierRegion],
        start: Cell,
        claimed: Sequence[FrontierRegion] = (),
    ) -> FrontierAssignment | None:
        """Choose the cheapest reachable unclaimed frontier.

        Args:
            grid: The occupancy grid to plan over (read-only).
            frontiers: Candidate regions, as returned by `Mapper.get_frontiers`.
            start: The drone's current (col, row) cell.
            claimed: Regions already assigned to other drones. These are never
                re-selected, and candidates near them are penalized.

        Returns:
            The assignment, or None when no candidate is both unclaimed and
            reachable — the coordinator reads that as "this drone idles".
        """
        # Membership test only; set iteration order never affects the result.
        claimed_cells = {region.cell for region in claimed}
        # Compare squared distances to avoid a sqrt per candidate pair.
        radius_sq = self._spread_radius * self._spread_radius
        spreading = self._spread_radius > 0.0 and self._spread_penalty > 0

        best: FrontierAssignment | None = None
        best_key: tuple[int, int, int] | None = None

        for region in frontiers:
            if region.cell in claimed_cells:
                continue  # hard exclusion: one drone per frontier
            path = self._planner.plan(grid, start, region.cell)
            if path is None:
                continue  # unreachable through known-free space

            cost = path_cost(path)
            score = cost
            if spreading and self._crowds_claimed(region, claimed, radius_sq):
                score += self._spread_penalty

            # Ties break on the region's (row, col) — never on list order — so
            # the assignment is identical across runs.
            col, row = region.cell
            key = (score, row, col)
            if best_key is None or key < best_key:
                best_key = key
                best = FrontierAssignment(region=region, path=path, cost=cost)

        return best

    @staticmethod
    def _crowds_claimed(
        region: FrontierRegion,
        claimed: Sequence[FrontierRegion],
        radius_sq: float,
    ) -> bool:
        """True if `region`'s centroid lies within the radius of a claimed one."""
        col_x, col_y = region.centroid
        for other in claimed:
            other_x, other_y = other.centroid
            d_x, d_y = col_x - other_x, col_y - other_y
            if d_x * d_x + d_y * d_y <= radius_sq:
                return True
        return False
