# Feature 1 — Frontier detection (`mapping.get_frontiers()`)

Branch: `feat/frontier-detection` (off `sprint-2`)
Status: **planning — one decision pending (clustering algorithm)**

## Goal

Add frontier detection to `mapping`: scan the occupancy grid, find **frontier
cells** (free cells bordering unknown space), group them into **regions**, and
return each region's world-coordinate centroid, a representative free cell (an
A\* goal), and its size. This is the raw material the `planning` module selects
from; detection lives here to keep `planning → mapping` one-way.

## Public API

A pure function (grid in → regions out, no MuJoCo — easy to unit test on
hand-built grids), plus a thin `Mapper` convenience method.

```python
# src/swarm_mapping/mapping/frontier.py
@dataclass(frozen=True)
class FrontierRegion:
    centroid: tuple[float, float]   # world (x, y) — for display + spreading penalty
    cell: tuple[int, int]           # representative (col, row): a FREE frontier cell → A* goal
    size: int                       # number of frontier cells in the region

def detect_frontiers(
    grid: OccupancyGrid,
    *,
    free_threshold: float = 0.4,    # p < this  => free
    occ_threshold: float = 0.6,     # p > this  => occupied; between => unknown
    min_region_size: int = 2,       # drop noise regions smaller than this
) -> list[FrontierRegion]: ...
```

```python
# Mapper convenience
def get_frontiers(self) -> list[FrontierRegion]:
    return detect_frontiers(self._grid)
```

## Definitions

- **Cell class** from `grid.probability()`: `free` if `p < free_threshold`,
  `occupied` if `p > occ_threshold`, else `unknown` (includes never-touched
  cells, `log_odds == 0 → p = 0.5`). Thresholds default to the existing
  Sprint-1 convention (0.4 / 0.6); can migrate into `MapConfig` later.
- **Frontier cell**: a `free` cell with **≥1 4-connected neighbor that is
  `unknown`**. Occupied neighbors (walls) do not make a frontier.
- **Region**: a connected cluster of frontier cells (see clustering decision).
- **Representative cell**: the region's frontier cell nearest its centroid
  (tie-break `(row, col)`). Guaranteed free → a valid A\* goal (the centroid
  itself may land on an unknown/occupied cell).

## Edge cases (baked into tests)

- All-unknown grid → no free cells → **no frontiers**.
- All-free grid (nothing unknown) → **no frontiers**.
- Free cell bordered only by occupied (wall) → **not** a frontier.
- **Grid border**: out-of-bounds neighbors are *not* treated as unknown, so
  free cells at the map edge are not frontiers just for touching the edge (we
  don't send drones to chase the finite grid boundary).
- Regions smaller than `min_region_size` are dropped.

## Determinism

- Iterate cells row-major; BFS with a fixed neighbor order.
- Output list sorted by representative cell `(row, col)` so it is stable.
- No dict/set iteration in the ordered output path.

## ⏳ DECISION PENDING — clustering algorithm (your call, per Decision 4)

How to group frontier cells into regions:

| Option | How | Deps | Notes |
| --- | --- | --- | --- |
| **A. Hand-written BFS connected-components** (recommended) | flood-fill over frontier cells, 8-connectivity | none | ~20 lines, fully deterministic, no new dep, matches "prefer the simple thing" |
| B. `scipy.ndimage.label` | label connected components in one call | **+scipy** | ~3 lines, battle-tested, but adds a heavy dependency (needs flagging) |
| C. `sklearn` DBSCAN | density clustering on cell coords | **+scikit-learn** | overkill for grid adjacency, heavier, determinism needs care |

**Recommendation: A** — connected-components via BFS, 8-connectivity for
clustering (merges diagonally-touching frontier cells into one region), while
the frontier *neighbor test* uses 4-connectivity (standard). No new dependency,
deterministic by construction.

Sub-choices to confirm with A: **8-connectivity for clustering** (vs 4), and
`min_region_size = 2` (vs 1).

## Test plan (write signatures first, review, then implement)

`tests/unit/test_mapping/test_frontier.py`, tagged `pytest.mark.sprint(2)`;
pure-logic on hand-built `OccupancyGrid`s (no MuJoCo). `*` = candidate `sanity`.

1. all-unknown grid → `[]`
2. all-free grid → `[]`
3. \* single free cell next to unknown → one region, correct centroid/cell/size
4. free cell walled by occupied (no unknown neighbor) → `[]`
5. boundary free cells (free region flush against unknown) → frontier along the seam
6. two separated frontier regions → two regions, deterministically ordered
7. grid-edge free cells (out-of-bounds neighbor) → not a frontier
8. `min_region_size` drops a 1-cell region
9. representative `cell` is free and near the centroid
10. determinism: identical output across repeated calls

## Files

- `src/swarm_mapping/mapping/frontier.py` — `FrontierRegion`, `detect_frontiers`
- `src/swarm_mapping/mapping/mapper.py` — add `get_frontiers()`
- `src/swarm_mapping/mapping/__init__.py` — export
- `tests/unit/test_mapping/test_frontier.py` — `sprint(2)` tests

## Done when

- `get_frontiers()` returns correct, deterministic clustered regions on the test
  grids; edge cases covered; coverage counts toward mapping's ≥70%; ruff + mypy
  clean; merged into `sprint-2`.
