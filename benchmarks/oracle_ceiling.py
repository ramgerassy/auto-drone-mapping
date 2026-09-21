"""Hypothesis test: would a next-best-view strategy explore measurably faster?

The claim under test is that `NearestFrontier` wastes travel by ranking
frontiers on distance alone, and that weighing expected information gain would
fix it. `benchmarks.oracle.OracleInfoGainFrontier` is the instrument: it scores
candidates with the exact unknown-cell count a visit reveals, read from ground
truth, so it cannot be beaten by any implementable strategy.

    uv run python benchmarks/oracle_ceiling.py --mode=greedy large_indoor

RESULT (2026-09-21, `large_indoor`, `assignment: greedy`, 12 runs):

    drones  variant          ticks   t@95%      cov   revis   path
         1  baseline          3414    2371   97.37%     503   3413
         1  oracle d=0        4000       —   91.43%     992   4000   capped
         1  oracle d=0.05     4000    2043   97.34%     591   4000   capped
         1  oracle d=0.1      3862    1934   97.36%     664   3861
         1  oracle d=0.2      3303    1976   97.36%     460   3302
         1  oracle d=0.5      3205       —   88.76%     232   3205   stalled
         3  baseline          1049     743   97.36%     229   2864
         3  oracle d=0        2987    1398   97.35%    1991   8754
         3  oracle d=0.05     1427     686   97.35%     529   4217
         3  oracle d=0.1      2522     619   97.34%     666   4509   stalled
         3  oracle d=0.2      1411     617   97.36%     508   4076
         3  oracle d=0.5      1157     666   97.36%     308   3375

**The hypothesis does not survive.** Against baseline, the best *completed*
oracle run reaches 95% coverage 17% sooner at both swarm sizes — and at three
drones takes 35% longer to finish and travels 42% further doing it. It
front-loads the big reveals, then pays to mop up what it scattered. That is the
ceiling; every real implementation sits below it, and the attempted one ran 17x
slower per tick, which swallows 17% several times over.

`d=0` — pure information gain, distance ignored — is catastrophic at both sizes.
Distance dominates gain on these maps, which is a property of the problem
rather than a tuning artifact.

Three of the twelve runs ended capped or stalled; their tick counts are
censored, not measured, and they are excluded from the comparison above. One
map, so this is a statement about `large_indoor`, not a law.

**Runs under `assignment: greedy`, not the shipped `global`, and that is the
first result rather than a detail.** Under `global`, `_assign_globally` picks
each target with a Dijkstra cost field and a swarm-wide matching, then hands
`FrontierStrategy.select` a *one-element* candidate list. Measured over four
missions on two maps: 718 selection calls, every one with a single candidate.
Under `greedy` the same maps give 99-100% of calls two or more candidates, up
to sixteen. So in every shipped scenario the strategy seam chooses nothing —
it routes — and no next-best-view implementation could have changed a single
decision. The sweep therefore prices the family where it is live at all.

`decay` is swept rather than fixed. It weights distance against gain, and the
two ends of the sweep are the two strategies already in hand: `decay=0` is pure
information gain, and a large `decay` collapses onto nearest-frontier. Taking
the best row is what makes this a bound over the family instead of a verdict on
one arbitrary weighting — the mistake that would turn a badly tuned run into
"NBV does not help".

Metrics, and why each is here:

- **ticks** — mission length; the headline. What a better strategy is for.
- **t@95%** — ticks to the Tier-1 indoor coverage target. Separates "explores
  faster" from "spends longer finishing the last few cells", which is the
  distinction the 3-vs-4 drone scaling row turned on.
- **coverage** — guards the rest. Finishing sooner by exploring less is not an
  improvement.
- **revisited** — cells a drone entered more than once. The direct measure of
  the wasted travel this whole line of work is about.
- **path** — total cells stepped, summed over drones. Ticks conflate travel
  with waiting; this does not.
- **balance** — smallest / largest per-drone cell count, so a run that finishes
  by parking a drone is visible as such.

Imports `tests.scene_truth` for the ground-truth grid. Both are dev-only and
the alternative is a second copy of the scene-parsing rule, which is worse: two
definitions of "what the map should have been" is how a measurement harness
starts disagreeing with the acceptance suite it is supposed to corroborate.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tests.scene_truth import truth_grid

from benchmarks.oracle import OracleInfoGainFrontier, TruthVisibility
from swarm_mapping.cli import build_mission, coverage_fraction
from swarm_mapping.config.loader import load_config
from swarm_mapping.mapping.types import MapConfig

ROOT = Path(__file__).resolve().parent.parent

# Tier-1 indoor coverage target, from CLAUDE.md's KPI table.
_TARGET_COV = 0.95

# Swept distance weights, in score units per metre. 0 ignores distance
# entirely; 1.0 discounts a frontier 10 m away to e^-10 of its gain, which is
# nearest-frontier in all but name. `None` marks the baseline run, which uses
# the shipped `NearestFrontier` rather than the oracle at all.
DECAYS: tuple[float | None, ...] = (None, 0.0, 0.05, 0.1, 0.2, 0.5)

# Where the residue lives (the tree) and where it does not (the loop). Running
# both is the point: a benchmark on one map cannot tell a strategy property
# from a map property, which is the lesson this sprint was built on.
SCENARIOS = ("large_indoor", "loop_indoor")
DRONE_COUNTS = (1, 3)


def run(scenario: str, drones: int, decay: float | None, mode: str) -> dict[str, Any]:
    """Fly one mission and collect its metrics.

    Args:
        scenario: Scenario directory name under `scenarios/`.
        drones: Swarm size override.
        decay: Oracle distance weight, or None for the shipped baseline.
        mode: `coordination.assignment` value. Under `global` the allocator
            picks each target and hands the strategy a one-element candidate
            list, so no strategy can change the outcome — see this module's
            note on why the sweep runs under `greedy`.

    Returns:
        One row of the comparison table.
    """
    base = load_config(ROOT / "scenarios" / scenario / "config.yaml")
    config = replace(base, coordination=replace(base.coordination, assignment=mode))
    mission = build_mission(config, drones)
    master, mapper = mission.master, mission.mapper

    if decay is not None:
        truth = truth_grid(
            mission.engine.model,
            MapConfig(
                resolution=config.map.resolution,
                origin_x=config.map.origin_x,
                origin_y=config.map.origin_y,
                grid_width=config.map.grid_width,
                grid_height=config.map.grid_height,
            ),
        )
        # Reaches into the master to swap the strategy rather than threading a
        # factory through `build_mission`. A benchmark-only extensibility point
        # in shipped code would be exactly the "just in case" seam CLAUDE.md
        # says not to add, and the planner identity the master guards at
        # construction still holds: the oracle exposes the same planner object.
        master._strategy = OracleInfoGainFrontier(  # noqa: SLF001
            master._strategy.planner,  # noqa: SLF001
            TruthVisibility(truth, config.sensor.max_range),
            decay=decay,
            spread_radius=config.planning.spread_radius,
            spread_penalty=config.planning.spread_penalty,
        )

    shape = (config.map.grid_height, config.map.grid_width)
    visits = {d: np.zeros(shape, dtype=np.int64) for d in master.drone_states}
    steps = dict.fromkeys(master.drone_states, 0)
    previous_cell: dict[int, tuple[int, int] | None] = dict.fromkeys(
        master.drone_states
    )
    ticks_to_target: int | None = None

    started = time.time()
    while not master.is_complete and master.tick_count < config.coordination.max_ticks:
        master.tick()
        for drone_id, state in master.drone_states.items():
            col, row = state.cell
            visits[drone_id][row, col] += 1
            if previous_cell[drone_id] != state.cell:
                steps[drone_id] += 1
                previous_cell[drone_id] = state.cell
        if ticks_to_target is None and coverage_fraction(mapper.grid) >= _TARGET_COV:
            ticks_to_target = master.tick_count

    per_drone = [int(np.count_nonzero(v > 0)) for v in visits.values()]
    balance = min(per_drone) / max(per_drone) if min(per_drone) else 0.0

    return {
        "scenario": scenario,
        "drones": drones,
        "decay": decay,
        "mode": mode,
        "ticks": master.tick_count,
        "capped": master.tick_count >= config.coordination.max_ticks,
        "t95": ticks_to_target,
        "coverage": round(coverage_fraction(mapper.grid), 4),
        "revisited": int(sum(int(np.count_nonzero(v > 1)) for v in visits.values())),
        "path": int(sum(steps.values())),
        "balance": round(balance, 3),
        "seconds": round(time.time() - started, 1),
    }


def label(decay: float | None) -> str:
    """Human name for a variant."""
    return "baseline" if decay is None else f"oracle d={decay:g}"


def main() -> None:
    """Run the sweep and write `benchmarks/oracle_ceiling.json`."""
    args = sys.argv[1:]
    mode = "greedy"
    for arg in list(args):
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            args.remove(arg)
    scenarios = args or list(SCENARIOS)

    rows: list[dict[str, Any]] = []
    header = (
        f"{'scenario':<14}{'drones':>7}{'variant':>14}{'ticks':>7}"
        f"{'t@95%':>7}{'cov':>8}{'revis':>8}{'path':>8}{'bal':>7}{'wall':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for scenario in scenarios:
        for drones in DRONE_COUNTS:
            for decay in DECAYS:
                row = run(scenario, drones, decay, mode)
                rows.append(row)
                t95 = "—" if row["t95"] is None else row["t95"]
                print(
                    f"{row['scenario']:<14}{row['drones']:>7}{label(decay):>14}"
                    f"{row['ticks']:>7}{t95:>7}{row['coverage']:>8.2%}"
                    f"{row['revisited']:>8}{row['path']:>8}{row['balance']:>7.2f}"
                    f"{row['seconds']:>6.0f}s",
                    flush=True,
                )

    out = ROOT / "benchmarks" / f"oracle_ceiling_{mode}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
