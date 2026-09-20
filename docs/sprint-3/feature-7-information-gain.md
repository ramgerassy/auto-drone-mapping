# Feature 7 — Information-gain frontier selection

Branch: `feat/information-gain-frontier`
Depends on: Feature 6 (the `--visit-heatmaps` diagnostic is the measurement)
Seam: `FrontierStrategy` — this is the second implementation CLAUDE.md names

---

## The problem, measured

`--visit-heatmaps` on `large_indoor`, 3 drones, 1498 ticks, 97.6% coverage:

```
Drone 0: 1103 cells visited, 221 revisited, worst cell 61 times
Drone 1: 1204 cells visited, 198 revisited, worst cell  6 times
Drone 2:  770 cells visited, 264 revisited, worst cell 16 times
```

Reading the images: drone 0 covers the **western half and nothing east**; drone
2 never leaves the corridor junction and **retreads 34%** of the cells it
visits; drone 0's trace is a zigzag of long diagonal traverses across rooms it
already cleared.

All three symptoms share one cause. **`NearestFrontier` plans a path to the
frontier *cell*, when the rangefinder reaches 12 m and only needs a vantage
point from which the frontier becomes *observable*.** A frontier is a free cell
adjacent to unknown space, which is almost always against a wall — so "go to
the cheapest frontier" means "cross the map to a wall" by construction.

It also has no notion of *how much* a target is worth. A frontier facing an
unexplored room and one facing a two-cell crevice of classification noise score
identically, so the swarm spends ticks on both.

## Approach

`InformationGainFrontier` scores **viewpoints**, not frontier cells, and trades
expected gain against travel cost.

For each unclaimed candidate region:

1. Plan a path as today (with Feature 6's lower-bound pruning).
2. **Truncate it** at the first cell within `observation_radius` of the
   region — the earliest point from which the target is in sensor range. That
   is the viewpoint; everything past it is travel the sensor has made
   unnecessary.
3. Score the viewpoint by **expected information gain**: how many unknown cells
   lie within sensor range of it. **A candidate with zero gain is skipped
   entirely** — there is nothing to learn there, which is exactly what a
   phantom frontier over classification noise looks like.
4. Rank by gain per unit travel cost.

Step 3 is what deprioritizes phantom frontiers: wall-surface classification
noise has almost no unknown space around it, so it scores near zero and the
swarm stops chasing it. That should also reduce the `no_progress_ticks` stalls,
which exist because the swarm cannot currently tell junk targets from real ones.

## ✅ DECISION D1 (resolved: windowed unknown count) — how gain is computed

- **(a) Windowed unknown count.** Count unknown cells in the square window of
  side `2R+1` around the viewpoint, via a summed-area table computed once per
  `select` call — O(grid) once, then O(1) per candidate. Ignores line of sight,
  so it over-counts unknown space hidden behind a wall.
- **(b) True visibility.** Ray-cast on the grid from each viewpoint and count
  unknown cells actually reachable by a ray. Exact, and far more expensive:
  ~36 rays x ~60 cells per candidate, per candidate, per re-selection.

**Resolved: (a).** The number is a *ranking* heuristic, not a quantity anyone
reports, and the over-count is largely uniform across candidates in the same
neighbourhood. (b) multiplies the cost of the hottest loop in the system, which
Feature 6 had to work hard to make affordable. If ranking quality proves
insufficient, (b) is a drop-in replacement for one method.

## ✅ DECISION D2 (resolved: ratio, gain / cost) — how gain and cost combine

- **(a) Ratio, `gain / cost`.** Scale-free: no constant to tune per scenario.
  Compared by integer cross-multiplication (`a.gain * b.cost` vs
  `b.gain * a.cost`) so ranking stays in exact integer arithmetic, as the
  determinism rule requires.
- **(b) Linear, `gain - lambda * cost`.** One tuning constant, whose right
  value depends on map size and resolution — a per-scenario knob, and this
  project already has a config key for every one of those.

**Resolved: (a).**

## ✅ DECISION D3 (resolved: add beside, config-selected) — replace `NearestFrontier` or add beside it

- **(a) Add beside**, selected by a new `planning.strategy` config key.
  `NearestFrontier` stays tested and available. Open-closed, which is what the
  seam exists for — and it gives a clean A/B: the same scenario under both
  strategies, compared on the heatmap numbers above.
- **(b) Replace it.** Less code, no config key, no dead implementation.

**Resolved: (a)**, specifically because of the A/B. The claim "this is better"
is worth nothing without the comparison, and the comparison needs both
strategies runnable.

## Determinism

Gain is an integer count and cost an integer in the planner's 10/14 units, so
the score is an exact rational: ranking uses `fractions.Fraction(gain, cost + 1)`
rather than a float division. The `+ 1` is one unit of standing still — it keeps
the denominator non-zero when a frontier is already observable from where the
drone is, and makes that case rank on pure gain.
Ties break on `(row, col)` exactly as `NearestFrontier` does. The summed-area
table is a deterministic function of the grid.

## Test plan (review these before I write them)

**Gain measurement** — pure, no simulator

1. A viewpoint surrounded by unknown cells outscores one surrounded by free
   cells.
2. Gain counts only cells inside the window, not the whole map.
3. Gain is 0 on a fully explored map, for every candidate.

**Viewpoint truncation**

4. A path to a distant frontier is truncated: the chosen viewpoint is within
   `observation_radius` of the region, and strictly closer to the drone than
   the frontier cell itself.
5. A frontier already within observation range yields the drone's own cell as
   the viewpoint, with zero travel cost.
6. The truncated path is a prefix of the full path — the drone never detours.

**Ranking**

7. Between two equal-cost candidates, the higher-gain one wins.
8. Between two equal-gain candidates, the cheaper one wins.
9. A high-gain distant target beats a near-zero-gain adjacent one — the
   phantom-frontier case, and the reason this feature exists.
10. Unreachable candidates are skipped, as in `NearestFrontier`.
11. Claimed regions are hard-excluded, as in `NearestFrontier`.
12. Determinism: identical inputs, identical assignment.

**Seam conformance**

13. `InformationGainFrontier` satisfies `FrontierStrategy` and is accepted by
    `CentralizedMaster` wherever `NearestFrontier` is.

**The A/B — acceptance, marked `acceptance`**

14. `large_indoor`, 3 drones, both strategies, comparing: ticks to completion,
    coverage, total cells revisited, and worst-cell revisit count. Asserts the
    new strategy is **not worse** on coverage and **better** on revisits, and
    prints the full table so the improvement is legible rather than just green.

## Risks

- **It may not actually be better.** The honest outcome is the A/B table; if
  information gain loses, that is a finding to record, not a number to bend.
  This is why D3 recommends keeping both strategies.
- **Cost.** `select` is the hottest loop in the system. The summed-area table
  is O(grid) once per call; if that proves too slow it can be hoisted to once
  per tick in the master, at the price of a parameter.

## Done when

- All 14 tests green, full suite green, ruff + mypy clean.
- The A/B table recorded in `docs/progress.md` with real numbers.
- `docs/design.md` §3 updated: the `FrontierStrategy` seam now has its second
  implementation, which is the claim the architecture has been making since
  Sprint 2.


## Sensor range is not a new config key

`observation_radius` and the gain window are both "how far this drone can see",
which `sensor.max_range` already states. The strategy takes it as a constructor
parameter sourced from there, so the number keeps one definition. A second key
could disagree with the sensor and would silently make the truncation wrong.

---

## Status: INCOMPLETE — does not work, do not ship

Implemented, measured, and **not** merged into the scenarios: both configs
ship `strategy: nearest`. Three flaws were found by measurement, two fixed and
the third unresolved. Recorded in order, because each was a wrong assumption
worth not repeating.

### Flaw 1 — range without line of sight (fixed)

Truncating at "the first path cell within `sensor_range` of the target"
returned the drone's **own cell** for almost every candidate: 12 m at 0.2 m
cells is 60 cells on a 250-cell map, so nearly every frontier is nominally in
range of wherever the drone stands, including ones behind three walls. Every
assignment therefore completed instantly, Feature 6's exhausted-frontier logic
marked each target dead, and the mission ended at **tick 6 with 4.25%
coverage**.

D1 accepted ignoring line of sight because the gain figure is only a *ranking*
heuristic. That reasoning was sound for gain and wrong for truncation, which
is a *geometric* question. Fixed by requiring line of sight.

### Flaw 2 — unknown cells treated as transparent (fixed)

The first line-of-sight implementation let rays pass through unknown cells, on
the reasoning that a scan's rays do. But then a frontier deep in unexplored
territory is "visible" from the drone's current cell, so every candidate is
already observable and none is worth travelling to — still **tick 6, 4.25%**.

The correct reading is the contrapositive of what unknown means: if the space
between really were observable from here, it would not still be unknown. Only
known-free cells are transparent.

### Flaw 3 — cost (UNRESOLVED)

With both fixed, the strategy runs but is far too slow and no better:

```
                 tick 400 coverage    wall
nearest              ~36.9%            ~11s
information_gain      36.78%           191s
```

~17x slower per tick for no coverage gain at tick 400. Removing the truncation
to isolate the gain *ranking* alone did not finish 25 ticks in 90 s, which
suggests the cost is not only the line-of-sight walk.

**Not diagnosed.** Candidates, in the order worth trying:

1. The summed-area table is rebuilt per `select` call, and each build calls
   `grid.probability()` — an `exp` over every cell. Hoisting it to once per
   tick in the master was flagged as a risk in this document before any code
   was written.
2. The backward line-of-sight walk is O(k) Bresenhams of O(k) cells each, so
   O(k²) per candidate where k is how far sight extends — worst in exactly the
   open spaces where it is least useful.
3. Gain ranking may pick systematically more distant targets than nearest-cost
   does, lengthening every path. That would be a *behavioural* cost, not an
   implementation one, and would argue against the whole approach.

Distinguishing 3 from 1 and 2 is the next step, and it needs a profile rather
than another guess — the three flaws above were each found by measurement after
a plausible-sounding assumption, and the pattern is not accidental.

### What is worth keeping

`NearestFrontier` is untouched and remains the default. The `planning.strategy`
config key, the `FrontierStrategy` seam finally having a second implementation,
and the A/B harness are all reusable. The problem this feature targets is real
and measured — the heatmaps at the top of this document are unchanged.
