# Feature 6 — Sprint 2 integration

Branch: `feat/sprint2-integration`
Depends on: Features 4, 4b, 4c, 5 — everything the sprint built
Closes: the sprint (`sprint-2 → main` after this merges)

---

## Goal

Make the sprint's parts run as a system from the command line. Everything
built since Feature 1 is reachable only from tests: `cli.py` still runs the
Sprint-1 single-drone patrol, and three safety parameters
(`clearance_radius`, `exclusion_radius`, `min_separation`) exist only as
Python defaults with no way to set them.

## What is unwired today

| Parameter | Introduced | Reachable from config? |
| --- | --- | --- |
| `exclusion_radius` | 4b | ✗ — Python default only |
| `clearance_radius` | 4c | ✗ — required arg, only tests pass it |
| `min_separation` | 4 | ✗ — constructor arg only |
| `CentralizedMaster` | 4 | ✗ — `cli.py` never constructs one |

`scenarios/large_indoor/config.yaml` already carries
`planning.clearance_radius`, written in Feature 5 for this feature to pick up.

## Scope

1. Multi-drone config schema + validation
2. `cli.py` driving `CentralizedMaster` instead of the patrol
3. Multi-drone `--view`
4. End-to-end + acceptance tests
5. The scaling KPI check
6. `docs/design.md` brought up to date

## Proposed schema

```yaml
scene:
  path: large_indoor.xml

drones:                          # replaces the single `drone:` block
  start_positions:
    - [0.0, 0.0, 1.0]
    - [0.0, -1.2, 1.0]
    - [0.0, 1.2, 1.0]
  altitude: 1.0

sensor:
  num_rays: 72
  max_range: 12.0
  exclusion_radius: 0.30         # Feature 4b — teammate filter

map:
  resolution: 0.2
  origin_x: -25.0
  origin_y: -25.0
  grid_width: 250
  grid_height: 250
  max_height: 3.0

planning:
  clearance_radius: 0.20         # Feature 4c — body inflation
  spread_radius: 0.0             # Feature 3 — spatial spreading
  spread_penalty: 0

coordination:
  min_separation: 0.5            # Feature 4 — centre-to-centre, metres
  max_wait_ticks: 5
  max_ticks: 2000                # safety cap; a blocked mission must end
```

Drone count is `len(start_positions)` rather than a separate `count` key, so
the two can never disagree.

## Validation — and where it may not live

CLAUDE.md commits to "validated at startup, fail-fast on errors", and `config`
**depends on nothing**. So `config` checks only what it can see without
importing another module:

- required keys present, correct types
- `1 <= len(start_positions) <= 5` (CLAUDE.md's stated range)
- positive `resolution`, `num_rays`, `max_range`, `max_ticks`
- non-negative radii
- ~~`grid_width * resolution` and the origin covering the scene~~ — **dropped.**
  `config` cannot know the scene's extent without parsing the MJCF, and the
  scene's geometry is `simulation`'s knowledge; reaching for it would mean
  `config → simulation`, the same violation this section exists to avoid. The
  checkable neighbour is *internal* consistency: every `start_position` must
  fall inside the grid extent implied by `origin_x/y` and
  `grid_width/height * resolution`. Same class of bug — a grid that does not
  fit the mission — caught without the dependency. Test 5 moves with it.

`load_config` returns a frozen `ScenarioConfig` (nested `DronesConfig`,
`SensorConfig`, `MapSectionConfig`, `PlanningConfig`, `CoordinationConfig`)
rather than `dict[str, Any]`. Validation needs somewhere to live, and the
constructor is where this project already puts invariants; the side effect is
that `cli.py` stops indexing an untyped dict and mypy starts checking the wiring
that this whole feature is about.

**Physical floors stay where the geometry is visible** — the body-diagonal floor
on `min_separation` and the corner-radius floor on `exclusion_radius` are
already enforced in `CentralizedMaster.__init__` and `Rangefinder.__init__`,
which may legally see `DRONE_HALF_EXTENT`. Duplicating them into `config` would
mean `config → simulation`, which the dependency direction forbids. Settled in
the 2026-08-19 progress entry; restated here because it is the obvious thing to
get wrong.

## ✅ DECISION D1 — what happens to patrol mode → **(c), extended**

**Settled 2026-09-20: exploration only, e2e tests re-tagged, and the dead CLI
helpers deleted with the mode they served.**

`generate_patrol` and `interpolate_segment` go too, along with their unit tests.
The doc below argued for keeping `generate_patrol` on the grounds that it is
still correct and still tested — but "correct and tested" is not a reason to
keep a function nothing calls. `interpolate_segment` exists only to animate 2 m
patrol hops; under exploration a drone moves one cell (0.1–0.2 m) per tick and
`sync()` per tick animates smoothly on its own. Both would be dead code in
`cli.py` carrying live tests, which is the shape that makes a module look busier
than it is. Git remembers them if a lawnmower baseline is ever wanted back.

### Original options (retained for the record)

## ~~❓ DECISION D1~~ — what happens to patrol mode

`run_pipeline` currently generates a lawnmower patrol. Three options:

- **(a) Exploration only.** Delete the patrol path; both scenarios move to the
  new schema. Simplest code, one mode. But `tests/integration/test_e2e.py` is
  the **Sprint-1 regression gate** — it would keep passing (its assertions are
  about output files, npz shape and coverage > 80%, all of which exploration
  satisfies) while silently testing different behaviour.
- **(b) Dual mode**, chosen by which config block is present. Keeps the Sprint-1
  path genuinely under test; costs a branch in `run_pipeline` and two code
  paths to maintain for a mode the project has outgrown.
- **(c) Exploration only, and re-tag the Sprint-1 e2e tests** as Sprint-2
  progression, replacing them with explicit exploration assertions.

**Recommend (c).** The regression gate should keep guarding *Sprint-1
behaviour*, and after this feature that behaviour no longer exists — pretending
otherwise by leaving the tests pointed at a different pipeline is worse than
moving them deliberately. `generate_patrol` and its unit tests stay (the
function is still correct and still tested); it simply stops being what the CLI
runs.

## ✅ DECISION D2 — does `is_blocked` join the `Coordinator` seam → **(a), extended**

**Settled 2026-09-20: widen the seam, with `unreachable_frontiers` alongside
`is_blocked`.**

⚠️ This is a change to one of the four named stability points, so it is called
out here deliberately. `CentralizedMaster` already implements both properties,
so the diff is the Protocol and its docstrings — no implementation moves.

`is_blocked` alone would let a coordinator report *that* it gave up without
reporting *how much* it gave up on, and the CLI's honest summary needs the
second number as much as the first. A `DistributedAuction` owes both answers for
the same reason a centralized master does: the question is about the mission,
not about who orchestrated it.

### Original options (retained for the record)

## ~~❓ DECISION D2~~ — does `is_blocked` join the `Coordinator` seam

Feature 4c added `is_blocked` and `unreachable_frontiers` to
`CentralizedMaster` but deliberately **not** to the `Coordinator` Protocol,
leaving the choice here. The CLI must distinguish "explored" from "walled out"
to report honestly, so it needs one of:

- **(a) Widen the seam.** Any `Coordinator` must answer "did you finish or were
  you stuck". Arguably that *is* part of what a coordinator is for.
- **(b) Leave the seam alone**; the CLI depends on `CentralizedMaster`
  concretely for reporting.

**Recommend (a).** `Coordinator` is a named stability point, so this needs
flagging — but the question "did the mission finish or give up" is not specific
to a centralized master, and a `DistributedAuction` would owe the same answer.
A seam that can only report *termination* and not *outcome* is under-specified.

## ✅ DECISION D3 — how the scaling KPI is measured → **(a)**

**Settled 2026-09-20: ticks to reach 95% coverage.** Reported with the measured
ratio so a near-miss is legible rather than just red.

### Original options (retained for the record)

## ~~❓ DECISION D3~~ — how the scaling KPI is measured

The sprint commits to "speedup >= 1.5x from 1 → 3 drones (small indoor)".
Undefined so far: speedup in *what*. Candidates:

- **(a) Ticks to reach 95% coverage.** Deterministic, no wall-clock noise,
  directly comparable across runs. Measures the thing the swarm is for.
- **(b) Wall-clock time to completion.** What a user feels, but noisy on CI and
  it would mostly measure per-tick compute, which *rises* with drone count.
- **(c) Total path length per drone.** A Tier-3 metric, not this KPI.

**Recommend (a).** With (b), adding drones can make the number worse while the
swarm genuinely explores faster, which would make the KPI misleading.

## ✅ DECISION D4 — how the scaling KPI gets two drone counts

Raised during review, not in the original draft. Drone count is
`len(start_positions)`, deliberately, so one config file cannot express both
arms of a 1-vs-3 comparison. Three ways out: a second committed config file, an
in-memory override inside the test, or a CLI flag.

**Settled 2026-09-20: a `--drones N` CLI override.** A second config file can
drift from the first in a way that silently invalidates the comparison — the one
failure mode that would make the KPI lie rather than fail. A test-only override
keeps the two arms honest but leaves the KPI unreproducible by hand, which is
poor for a number that goes in the report.

⚠️ This adds public CLI surface, so: `--drones N` takes the **first N**
`start_positions`. It never invents one — a fabricated position has to be
collision-free, inside the grid, and `min_separation` clear of its neighbours,
and a CLI flag guessing at that is exactly the silent-safety-failure this
codebase keeps refusing. `N > len(start_positions)` and `N < 1` are both errors
naming both counts. Config stays the source of truth for *where*; the flag
controls only *how many*.

## Reporting — `run_pipeline` returns a `MissionResult`

Tests 8, 9 and 14 all assert on mission *outcome*, and stdout is the wrong seam
to read it from. `run_pipeline` returns a frozen `MissionResult` (`ticks`,
`coverage`, `blocked`, `unreachable_frontiers`, `tick_capped`, `npz_path`,
`png_path`); `main()` prints it and exits non-zero when the mission was blocked
or hit `max_ticks`.

⚠️ A public CLI change: a walled-out mission now fails loudly instead of exiting
0 with a cheerful coverage figure. That is the point of D2 — an exit code is
where "did you finish or were you stuck" actually reaches a caller.

## Test gating

There is no `acceptance` pytest marker today; `pyproject.toml` has `slow`,
`sprint(n)`, `regression`, `progression`, and `sanity`, and `tests/acceptance/`
is scaffolded but empty. This feature adds the marker. Acceptance tests still
carry `sprint(2)` so they stay inside the regression/progression machinery that
`tests/conftest.py` builds — the marker selects *cost*, the sprint tag selects
*era*, and the two are orthogonal.

## Test plan (review these before I write them)

**Config validation** — `tests/unit/test_config/`

1. A valid multi-drone config loads and produces the expected values.
2. Missing required key → `ValueError` naming the key.
3. Zero drones and six drones both rejected (CLAUDE.md's 1–5 range).
4. Negative `clearance_radius` / `exclusion_radius` / `resolution` rejected.
5. A `start_position` outside the grid extent is rejected (see "Validation" —
   this replaces the original "grid does not cover the scene", which `config`
   cannot check without depending on `simulation`).

**CLI wiring** — `tests/integration/`

6. `run_pipeline` on `small_indoor` with 1 drone produces `.npz` + `.png`, and
   the npz keys and shapes match Sprint 1's format exactly (the output-format
   compatibility CLAUDE.md calls out).
7. `run_pipeline` on `small_indoor` with 3 drones reaches >= 95% coverage —
   the Tier-1 KPI, asserted rather than assumed.
8. A blocked mission terminates and is *reported* as blocked, not as success.
9. `max_ticks` is honoured, so a pathological config cannot hang CI.
10. Determinism: two runs of the same config produce byte-identical `.npz`.

**Acceptance** — `tests/acceptance/`

11. `large_indoor`, 3 drones: coverage >= 95%, and every one of the eight rooms
    entered — the scenario exists to test doorway traversal, so "explored" must
    mean more than "mapped the corridors".
12. Map accuracy >= 98% per-cell against a ground-truth grid built from the
    parsed geometry (the Feature 5 helper generalised).
13. Zero collisions: no two drones within `min_separation` on any tick, and no
    drone's planned cell ever clearance-blocked at the time it moves there.

**Scaling KPI** — `tests/acceptance/`

14. Ticks-to-95%-coverage on `small_indoor` with 1 drone vs 3 drones, asserting
    >= 1.5x. Reported with the measured number so a near-miss is visible rather
    than just red.

## Risks

- **Runtime.** Acceptance runs on a 250x250 grid with 3 drones are the most
  expensive tests in the suite. They are marked `acceptance` and excluded from
  the per-commit gate, per CLAUDE.md ("PRs and nightly, not every commit").
- **Tier-1 KPIs may not pass first time.** Coverage >= 95% with clearance
  inflation refusing wall-adjacent frontiers is the one I would bet against. If
  it misses, the honest options are a finer resolution (Feature 5's geometry
  already supports 0.1 unchanged) or an explicit, recorded tolerance — not a
  quietly weakened assertion.

## `docs/design.md` — written, not updated

The repo has `progress.md` and the sprint plans; there is no `design.md`. "Brought
up to date" is really "written from scratch" — the full seven-module document the
target layout calls for. It lands as its own commit at the end of the feature so
the code and tests are not held behind it.

## Done when

- All 14 tests green; full suite green; ruff + mypy clean.
- `swarm-mapping --config scenarios/large_indoor/config.yaml --view` shows
  three drones exploring the floor plan.
- `docs/design.md` matches what the code does.
- `feat/sprint2-integration` merged into `sprint-2`.
