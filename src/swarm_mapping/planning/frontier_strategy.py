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
from fractions import Fraction
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.mapping.raytrace import bresenham_2d
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

    @property
    def planner(self) -> PathPlanner:
        """The planner this strategy routes with.

        Read by `CentralizedMaster` to check that the planner it re-validates
        committed paths against is the one that planned them. Deliberately NOT
        on the `FrontierStrategy` Protocol: a strategy scoring by expected
        information gain against straight-line distance would have no planner
        to expose, and the seam should not demand one.
        """
        return self._planner

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

        # Plan cheapest-lower-bound first so the search can stop early. The
        # bound comes from the planner (see `PathPlanner.cost_lower_bound`), so
        # this makes no assumption about its cost model; the spreading penalty
        # only ever adds, so once the bound exceeds the best score found no
        # remaining candidate can win. The ordering key
        # carries (row, col) so the traversal is deterministic, and the break
        # uses a strict `>` so equal-bound candidates are still compared on the
        # existing tie-break rather than dropped by arrival order.
        #
        # This is pure cost, not behaviour: the selected assignment is
        # identical, but a re-selecting drone stops running A* against every
        # frontier on the map. That matters because dropping stale assignments
        # made re-selection common.
        ordered = sorted(
            (
                self._planner.cost_lower_bound(start, region.cell),
                region.cell[1],
                region.cell[0],
                region,
            )
            for region in frontiers
            if region.cell not in claimed_cells
        )

        for bound, _, _, region in ordered:
            if best_key is not None and bound > best_key[0]:
                break  # every remaining candidate is at least this expensive
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


def _has_line_of_sight(origin: Cell, target: Cell, occupied: NDArray[np.bool_]) -> bool:
    """Whether a straight ray from `origin` reaches `target` unobstructed.

    Traced with the same Bresenham the mapper uses, so "visible here" means the
    same thing to the planner as it does to the map that produced the frontier.
    Unknown cells do not block: a scan's rays pass through unmapped space, and
    unmapped space is precisely what the drone is going to look at.

    Args:
        origin: The candidate viewpoint.
        target: The cell to be observed.
        occupied: Mask of known-occupied cells, indexed [row, col].

    Returns:
        True if no known-occupied cell lies strictly between the two.
    """
    cells = bresenham_2d(origin[0], origin[1], target[0], target[1])
    return not any(bool(occupied[row, col]) for col, row in cells[1:-1])


def _unknown_window_counts(
    grid: OccupancyGrid,
    free_threshold: float,
    occ_threshold: float,
) -> NDArray[np.int64]:
    """Summed-area table of unknown cells, for O(1) window queries.

    Built once per `select` call so every candidate's information gain is a
    constant-time lookup instead of a scan. `select` is the hottest loop in the
    system — Feature 6 needed admissible pruning to make it affordable at all —
    so a per-candidate scan would undo that work.

    Args:
        grid: The occupancy grid to read.
        free_threshold: Probability below which a cell counts as free.
        occ_threshold: Probability above which a cell counts as occupied.

    Returns:
        A table of shape (height + 1, width + 1), zero-padded on the top and
        left so a window sum never needs a bounds branch.
    """
    prob = grid.probability()
    unknown = (prob >= free_threshold) & (prob <= occ_threshold)
    table = np.zeros((unknown.shape[0] + 1, unknown.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = unknown.cumsum(axis=0).cumsum(axis=1)
    return table


class InformationGainFrontier:
    """Selects the frontier offering the most unknown space per unit of travel.

    `NearestFrontier` plans a path to the frontier *cell*. But a frontier is a
    free cell adjacent to unknown space, which is almost always against a wall,
    so "go to the cheapest frontier" means "cross the map to a wall" — and with
    a 12 m rangefinder the drone only ever needed a vantage point from which the
    target becomes *observable*. Measured on large_indoor before this existed,
    one drone retrod 34% of the cells it visited and another covered half the
    map while a third never left the corridor junction.

    So this scores **viewpoints**, not frontier cells:

    1. Plan to the region as usual.
    2. Truncate the path at the first cell within `sensor_range` of it. Every
       step past that is travel the sensor has made unnecessary.
    3. Score that viewpoint by how many unknown cells sit within sensor range.
    4. Rank by gain per unit travel cost.

    Step 3 is what stops the swarm chasing phantoms: wall-surface cells drift
    across the classification bands and emit frontier regions over space no scan
    can resolve, and such a region has almost no unknown space around it. A
    zero-gain candidate is skipped outright — there is nothing to learn there.

    Args:
        planner: The path planner used to route to and score candidates.
        sensor_range: How far the drone can see, in metres. Sourced from
            `sensor.max_range` rather than given its own config key, so the
            truncation cannot disagree with the sensor it models.
        spread_radius: World-frame radius around a claimed frontier's centroid
            inside which a candidate is penalized. 0 disables the penalty.
        spread_penalty: Travel cost added to a candidate inside that radius.
            Charged against cost rather than subtracted from gain: crowding a
            teammate does not make a region less informative, it makes going
            there less worthwhile.
    """

    def __init__(
        self,
        planner: PathPlanner,
        sensor_range: float,
        spread_radius: float = 0.0,
        spread_penalty: int = 0,
        free_threshold: float = 0.4,
        occ_threshold: float = 0.6,
    ) -> None:
        self._planner = planner
        self._sensor_range = sensor_range
        self._spread_radius = spread_radius
        self._spread_penalty = spread_penalty
        self._free_threshold = free_threshold
        self._occ_threshold = occ_threshold

    @property
    def planner(self) -> PathPlanner:
        """The planner this strategy routes with. See `NearestFrontier`."""
        return self._planner

    def select(
        self,
        grid: OccupancyGrid,
        frontiers: Sequence[FrontierRegion],
        start: Cell,
        claimed: Sequence[FrontierRegion] = (),
    ) -> FrontierAssignment | None:
        """Choose the viewpoint with the best information gain per unit cost.

        Args:
            grid: The occupancy grid to plan over (read-only).
            frontiers: Candidate regions from `Mapper.get_frontiers`.
            start: The drone's current cell.
            claimed: Regions already taken by other drones this tick.

        Returns:
            The chosen assignment, whose `path` ends at the **viewpoint** rather
            than at the frontier cell, or None if nothing is worth visiting.
        """
        claimed_cells = {region.cell for region in claimed}
        radius_cells = int(self._sensor_range / grid.config.resolution)
        table = _unknown_window_counts(grid, self._free_threshold, self._occ_threshold)
        # Known-free cells; see `_has_line_of_sight` on why unknown blocks.
        transparent = grid.probability() < self._free_threshold
        radius_sq = self._spread_radius * self._spread_radius
        spreading = self._spread_radius > 0.0 and self._spread_penalty > 0

        best: FrontierAssignment | None = None
        best_key: tuple[Fraction, int, int] | None = None

        for region in frontiers:
            if region.cell in claimed_cells:
                continue
            path = self._planner.plan(grid, start, region.cell)
            if path is None:
                continue

            viewpoint_index = self._viewpoint_index(
                path, region.cell, radius_cells, transparent
            )
            viewpoint_path = path[: viewpoint_index + 1]
            gain = self._gain(table, grid, viewpoint_path[-1], radius_cells)
            if gain == 0:
                continue  # nothing observable there — a phantom frontier

            cost = path_cost(viewpoint_path)
            if spreading and self._crowds_claimed(region, claimed, radius_sq):
                cost += self._spread_penalty

            # An exact rational, not a float: ranking is a decision path, and
            # `cost + 1` is one unit of standing still, keeping the denominator
            # non-zero when the target is already observable from `start`.
            col, row = region.cell
            key = (-Fraction(gain, cost + 1), row, col)
            if best_key is None or key < best_key:
                best_key = key
                best = FrontierAssignment(region=region, path=viewpoint_path, cost=cost)

        return best

    @staticmethod
    def _viewpoint_index(
        path: list[Cell],
        target: Cell,
        radius_cells: int,
        transparent: NDArray[np.bool_],
    ) -> int:
        """Index of the earliest cell on `path` that can actually see `target`.

        "Can see" is range **and** line of sight. Range alone is not enough and
        the difference is not academic: at 12 m and 0.2 m cells the sensor
        reaches 60 cells across a 250-cell map, so nearly every frontier is
        nominally in range of wherever the drone happens to be — including ones
        behind three walls. Truncating on range alone returned the start cell
        itself, so every assignment completed instantly, every target was marked
        exhausted, and the mission ended after 6 ticks having mapped 4%.

        Walks **backward** from the target and stops at the first break, so the
        work is proportional to how far line of sight actually extends rather
        than to the length of the path. In a room-and-corridor map that is a few
        cells — sight breaks at the doorway.

        Args:
            path: The full route, ending on the frontier cell.
            target: The frontier's representative cell.
            radius_cells: Sensor range in cells.
            transparent: Mask of known-free cells, indexed [row, col].

        Returns:
            An index into `path`.
        """
        limit_sq = radius_cells * radius_cells
        viewpoint = len(path) - 1
        for index in range(len(path) - 1, -1, -1):
            col, row = path[index]
            d_col, d_row = col - target[0], row - target[1]
            if d_col * d_col + d_row * d_row > limit_sq:
                break
            if not _has_line_of_sight(path[index], target, transparent):
                break
            viewpoint = index
        return viewpoint

    @staticmethod
    def _gain(
        table: NDArray[np.int64],
        grid: OccupancyGrid,
        viewpoint: Cell,
        radius_cells: int,
    ) -> int:
        """Unknown cells within `radius_cells` of `viewpoint`, as a box count.

        A square window rather than a disc, and without line-of-sight: this is a
        *ranking* heuristic, and the over-count from unknown space hidden behind
        a wall is broadly uniform among candidates in the same neighbourhood.
        True visibility would mean ray-casting per candidate, multiplying the
        cost of the hottest loop in the system.

        Args:
            table: The summed-area table from `_unknown_window_counts`.
            grid: The grid, for its bounds.
            viewpoint: The cell being scored.
            radius_cells: Half the window's side, in cells.

        Returns:
            The number of unknown cells in the window.
        """
        col, row = viewpoint
        col_lo = max(0, col - radius_cells)
        row_lo = max(0, row - radius_cells)
        col_hi = min(grid.config.grid_width, col + radius_cells + 1)
        row_hi = min(grid.config.grid_height, row + radius_cells + 1)
        return int(
            table[row_hi, col_hi]
            - table[row_lo, col_hi]
            - table[row_hi, col_lo]
            + table[row_lo, col_lo]
        )

    @staticmethod
    def _crowds_claimed(
        region: FrontierRegion,
        claimed: Sequence[FrontierRegion],
        radius_sq: float,
    ) -> bool:
        """True if `region`'s centroid lies within the radius of a claimed one."""
        for taken in claimed:
            d_x = region.centroid[0] - taken.centroid[0]
            d_y = region.centroid[1] - taken.centroid[1]
            if d_x * d_x + d_y * d_y <= radius_sq:
                return True
        return False
