"""The instrument for a hypothesis test: is a next-best-view strategy worth it?

**Hypothesis.** `NearestFrontier` sends drones back across ground they have
already covered because it ranks candidates on travel cost alone, ignoring how
much each one would reveal. A strategy that weighed expected information gain
would explore measurably faster.

That hypothesis drove `InformationGainFrontier` through three failed attempts
in Sprint 2, each abandoned on cost: scoring expected gain needs line-of-sight
from every candidate, and tracing it against a *belief* map every tick ran ~17x
slower than `NearestFrontier` for no measured benefit. Sprint 2.5 then showed
the backtracking it targeted is mostly the tree floor plan, not the strategy.
So rather than attempt it a fourth time, test the hypothesis directly.

**The test.** `OracleInfoGainFrontier` scores candidates with the exact number
of unknown cells a visit would reveal, read off the ground-truth scene instead
of estimated from the map so far. No implementable strategy can score better
than exactly right, so this is an upper bound on the entire family: whatever it
fails to win is not available to buy, at any price.

Two things it does NOT cheat at, deliberately:

- **Routing.** Paths are planned on the belief grid, through known-free cells
  only, exactly as the shipped system flies. An oracle that also took shortcuts
  through unmapped space would be measuring a different system.
- **Candidates.** Frontier regions come from the live map. The oracle chooses
  better among the same options, it does not invent options.

So the bound covers *target selection* and nothing else, which is precisely
what a `FrontierStrategy` controls.

**What it found** — see `benchmarks/oracle_ceiling.py` for the numbers. In
short: the hypothesis does not survive. A perfect oracle reaches 95% coverage
~17% sooner and pays for it with a longer mission and 42% more travel, and in
the shipped configuration it could not have acted at all.

Benchmark-only. Nothing here is importable from `swarm_mapping`, reachable from
the CLI, or selectable from a config. It satisfies the `FrontierStrategy`
Protocol so it can be dropped in without the coordinator knowing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from swarm_mapping.mapping.frontier import FrontierRegion
from swarm_mapping.mapping.grid import OccupancyGrid
from swarm_mapping.planning.frontier_strategy import FrontierAssignment
from swarm_mapping.planning.path_planner import Cell, PathPlanner, path_cost

# Classification bands, matching `mapping`: a cell is unknown between them.
_UNKNOWN_LOW, _UNKNOWN_HIGH = 0.4, 0.6

# Rays cast per viewpoint when tracing true visibility, and the step along each
# in cells. 720 rays at a half-cell step leaves no gap a 1-cell wall can hide
# in out to the ranges these scenarios use; the sensor's own ray count is
# deliberately NOT reused, because the oracle is meant to be better-informed
# than any sensor configuration, not tied to one.
_VIS_RAYS = 720
_VIS_STEP = 0.5


class TruthVisibility:
    """Exact visible sets over a static ground-truth grid.

    The ground truth never changes during a mission, so a viewpoint's visible
    set is a constant and is computed once per cell. Only the *unknown* mask
    moves, and intersecting against it is a numpy gather.

    That memo is the one concession to speed here, and it is what makes the
    experiment runnable at all: without it, tracing visibility per candidate
    per tick reproduces the very cost that killed the real implementation, and
    a bound nobody can afford to measure answers nothing. It is sound only
    because the grid is immutable — which is true of ground truth and was not
    true of the belief map the shipped attempt had to trace against.

    Args:
        truth: A fully-classified grid from `tests.scene_truth.truth_grid`.
        max_range: Sensor range in metres. Visibility is clipped to it.
    """

    def __init__(self, truth: OccupancyGrid, max_range: float) -> None:
        self._shape = truth.log_odds.shape
        self._solid = truth.probability() > _UNKNOWN_HIGH
        self._radius = max_range / truth.config.resolution
        angles = np.linspace(0.0, 2.0 * math.pi, _VIS_RAYS, endpoint=False)
        self._cos = np.cos(angles)[:, None]
        self._sin = np.sin(angles)[:, None]
        self._steps = np.arange(_VIS_STEP, self._radius + _VIS_STEP, _VIS_STEP)[None, :]
        self._memo: dict[Cell, NDArray[np.intp]] = {}

    def visible_from(self, cell: Cell) -> NDArray[np.intp]:
        """Flat indices of every cell with true line of sight from `cell`.

        The first solid cell each ray meets is included — a wall face is
        observed, which is how the mapper classifies it — and everything behind
        it is not.

        Args:
            cell: The (col, row) viewpoint.

        Returns:
            Sorted unique flat indices into a grid of the truth grid's shape.
        """
        memoized = self._memo.get(cell)
        if memoized is None:
            memoized = self._trace(cell)
            self._memo[cell] = memoized
        return memoized

    def gain(self, cell: Cell, unknown_flat: NDArray[np.bool_]) -> int:
        """How many currently-unknown cells a visit to `cell` would reveal.

        Args:
            cell: The (col, row) viewpoint.
            unknown_flat: Flattened boolean mask of unknown cells in the
                belief grid.

        Returns:
            Count of visible cells that are still unknown.
        """
        return int(np.count_nonzero(unknown_flat[self.visible_from(cell)]))

    def _trace(self, cell: Cell) -> NDArray[np.intp]:
        """Ray-march true visibility from one viewpoint."""
        height, width = self._shape
        col, row = cell
        cols = np.rint(col + self._cos * self._steps).astype(np.intp)
        rows = np.rint(row + self._sin * self._steps).astype(np.intp)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        # Clamp before indexing so out-of-bounds samples read a real cell; they
        # are discarded by `inside` a moment later and never reach the result.
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)

        solid = self._solid[rows, cols] & inside
        # A ray sees up to and including its first solid sample. `argmax` on a
        # boolean row gives that sample's index, or 0 when the row is all
        # False — hence the explicit `any` test for the no-hit case.
        hit_at = np.where(solid.any(axis=1), solid.argmax(axis=1), solid.shape[1] - 1)
        within = np.arange(solid.shape[1])[None, :] <= hit_at[:, None]

        keep = within & inside
        flat = (rows * width + cols)[keep]
        return np.unique(flat)


class OracleInfoGainFrontier:
    """Next-best-view with zero estimation error.

    Scores each reachable candidate as `gain * exp(-decay * cost_metres)`,
    where `gain` is the exact count of unknown cells the visit would reveal.
    One knob spans the whole family: `decay = 0` is pure information gain
    (fly anywhere, however far, for the biggest reveal) and large `decay`
    collapses onto nearest-frontier. Sweeping it and taking the best run is
    what makes the result a bound over the family rather than a verdict on one
    arbitrary weighting.

    The spreading penalty is applied to cost exactly as `NearestFrontier`
    applies it, so a high-decay run reproduces baseline behaviour and the
    comparison differs in the scoring rule alone. Changing two things at once
    is how the last three benchmarks in this project reached wrong conclusions.

    Args:
        planner: The path planner. Must be the same object the coordinator
            re-validates committed paths against.
        visibility: Ground-truth visibility oracle.
        decay: Cost weight per metre in the exponential. 0 ignores distance.
        spread_radius: As `NearestFrontier`.
        spread_penalty: As `NearestFrontier`.
    """

    def __init__(
        self,
        planner: PathPlanner,
        visibility: TruthVisibility,
        *,
        decay: float,
        spread_radius: float = 0.0,
        spread_penalty: int = 0,
    ) -> None:
        self._planner = planner
        self._visibility = visibility
        self._decay = decay
        self._spread_radius = spread_radius
        self._spread_penalty = spread_penalty

    @property
    def planner(self) -> PathPlanner:
        """The planner this strategy routes with — read by the master's guard."""
        return self._planner

    def select(
        self,
        grid: OccupancyGrid,
        frontiers: Sequence[FrontierRegion],
        start: Cell,
        claimed: Sequence[FrontierRegion] = (),
    ) -> FrontierAssignment | None:
        """Choose the highest-scoring reachable unclaimed frontier."""
        claimed_cells = {region.cell for region in claimed}
        radius_sq = self._spread_radius * self._spread_radius
        spreading = self._spread_radius > 0.0 and self._spread_penalty > 0

        probability = grid.probability()
        unknown_flat = (
            (probability >= _UNKNOWN_LOW) & (probability <= _UNKNOWN_HIGH)
        ).ravel()
        metres_per_unit = grid.config.resolution / 10.0

        best: FrontierAssignment | None = None
        best_key: tuple[float, int, int] | None = None

        # No lower-bound pruning here: unlike a pure-cost score, a distant
        # candidate can still win on gain, so every reachable candidate has to
        # be scored. That is affordable because this is a benchmark.
        for region in sorted(frontiers, key=lambda r: (r.cell[1], r.cell[0])):
            if region.cell in claimed_cells:
                continue
            path = self._planner.plan(grid, start, region.cell)
            if path is None:
                continue  # unreachable through known-free space

            cost = path_cost(path)
            effective = cost
            if spreading and self._crowds_claimed(region, claimed, radius_sq):
                effective += self._spread_penalty
            gain = self._visibility.gain(region.cell, unknown_flat)
            score = gain * math.exp(-self._decay * effective * metres_per_unit)

            # Maximising, so the key negates score; ties break on (row, col)
            # exactly as `NearestFrontier` does, keeping the run reproducible.
            col, row = region.cell
            key = (-score, row, col)
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
