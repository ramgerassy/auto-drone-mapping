# Feature 6 — handoff

Branch: `feat/sprint2-integration`
Spec: `docs/sprint-2/feature-6-integration.md` (updated with D1–D4 settled)

> **SUPERSEDED — kept as a record, not as current state.** Written mid-feature
> when the `large_indoor` convergence bug (Finding 2 below) stopped the work.
> All six scope items are now complete and Finding 2 is resolved — though *not*
> by the hypothesis this document proposes, which measurement refuted. See the
> 2026-09-20 entries in `docs/progress.md` for what actually happened.

---

## Where things stand

| Scope item | State |
| --- | --- |
| 1. Multi-drone config schema + validation | ✅ done, 35 tests |
| 2. `cli.py` driving `CentralizedMaster` | ✅ done |
| 3. Multi-drone `--view` | ✅ done, **not manually verified** (needs a display) |
| 4. End-to-end + acceptance tests | ⚠️ e2e done (11 tests); acceptance suite **not started** |
| 5. The scaling KPI check | ❌ not started |
| 6. `docs/design.md` | ❌ not started (file does not exist yet) |

Commits:

- `a8a2b8b` — spec update recording D1–D4
- `aa89ba3` — the wiring: config schema, Coordinator seam, CLI rewrite

---

## Decisions settled (all four confirmed by the user)

- **D1 — patrol mode:** exploration only. `tests/integration/test_e2e.py`
  re-tagged `sprint(1)` → `sprint(2)` with exploration assertions.
  `generate_patrol` and `interpolate_segment` deleted along with their tests —
  nothing calls them once the CLI explores, and one cell per tick needs no
  interpolation to animate.
- **D2 — `Coordinator` seam:** widened with **both** `is_blocked` and
  `unreachable_frontiers`. `CentralizedMaster` already implemented both, so the
  diff was the Protocol and its docstrings.
- **D3 — scaling KPI:** ticks to reach 95% coverage.
- **D4 (raised in review, not in the original draft) — KPI drone counts:** a
  `--drones N` CLI override taking the *first* N start positions, so the two
  arms of the comparison differ in nothing else. A second config file could
  drift and make the KPI lie rather than fail.

Two corrections to the original spec, both recorded in it:

- `config` **cannot** check "the grid covers the scene" — that needs the MJCF's
  extent, and reading it would mean `config → simulation`. It checks that every
  start position falls inside the grid instead. Test 5 moved with it.
- `docs/design.md` does not exist; it is written from scratch, not updated.

---

## What was built

**`src/swarm_mapping/config/schema.py`** (new). Frozen dataclasses —
`ScenarioConfig` holding `DroneSettings`, `SensorSettings`, `MapSettings`,
`PlanningSettings`, `CoordinationSettings`. `load_config` returns one instead of
`dict[str, Any]`. Note `_number`/`_integer` reject `bool` explicitly: it
subclasses `int`, so `num_rays: true` would otherwise run the mission with a
single ray.

**`src/swarm_mapping/cli.py`** (rewritten). `run_pipeline` returns a
`MissionResult` (`ticks`, `coverage`, `blocked`, `unreachable_frontiers`,
`tick_capped`, `npz_path`, `png_path`). `build_mission(config, drones)` returns a
`Mission` bundle (engine, mapper, master, config) **specifically so the
acceptance suite can drive its own tick loop** — measuring coverage per tick for
the KPI, or checking separation every tick — without duplicating the wiring.
`resolve_scene_path` accepts an absolute `scene.path` so tests can point at a
scene outside the package assets.

**Both scenario configs migrated** to the new schema. `large_indoor` gained
`spread_radius: 6.0` / `spread_penalty: 40` (eight rooms off a corridor cross is
exactly the layout where two drones chase the same doorway) and `max_ticks: 4000`.

**Deleted `main.py`** — unused `uv init` scaffold, the repo's only ruff failure
(pre-existing, unrelated to this feature), duplicating the real
`swarm-mapping = "swarm_mapping.cli:main"` entry point.

---

## Finding 1 (resolved): `is_blocked` fires on *successful* runs

`small_indoor`, 1 drone: terminates at **371 ticks, 98.13% coverage,
`blocked=True`, 3 unreachable frontier regions.**

Investigated rather than assumed. Of 749 unknown cells out of 40 000:

```
unknown inside obstacle footprints: 425
unknown within 0.5 m of a wall:     324
unknown elsewhere:                    0
```

Zero elsewhere. The map is as complete as clearance physically permits — body
inflation refuses the wall margin, so wall-adjacent frontiers stay visible but
unoccupiable forever.

**Consequence:** `is_blocked` is not a pass/fail signal. The first cut of the
CLI exited non-zero on it, which would have failed `small_indoor` at 98.1%
coverage. Fixed: `MissionResult.succeeded` is `not tick_capped` only. Docstrings
on `CentralizedMaster.is_blocked` and the `Coordinator` Protocol previously
claimed it separates "finished" from "walled out"; they now say what it actually
reports, with the measured example. What separates residue from a real wall is
*magnitude*: 3 regions at 98% vs 200 regions at 58%.

---

## Finding 2 (OPEN — this is where to pick up)

**`large_indoor` with 3 drones never terminates.**

```
tick  200  cov 16.202%   15s
tick  400  cov 55.109%   90s
tick  600  cov 75.758%  177s
tick  800  cov 83.826%  198s
tick 1000  cov 97.618%  205s
tick 1200  cov 97.598%  208s
...        (flat, oscillating 97.587%–97.619%)
tick 4000  cov 97.619%  279s
DONE ticks 4000 coverage 97.619% blocked False unreachable 0
```

Coverage plateaus at ~97.6% by tick 1000 and the mission then burns 3000 more
ticks achieving nothing, stopping only at the `max_ticks` cap. `blocked=False`
and `unreachable=0` because `_complete` was never True, so those fields were
never populated. `small_indoor` terminates fine (371 ticks), so this is specific
to the large scene.

### Evidence gathered

Frontier count at the plateau oscillates between 7 and 9 and never reaches 0.
Several frontier cells persist *unchanged* across every sample —
`(213,36)`, `(90,135)`, `(137,135)`, `(35,223)` — while drones travel 90–100
cells toward others. Drone 0 meanwhile chases transient frontiers along row ~187
that appear and vanish (`(215,188)`, `(222,187)`, `(233,187)`, `(244,188)`).

The decisive measurement — remaining unknown cells at tick 1100, against ground
truth built from the parsed MJCF box geoms:

```
unknown total 1498, of which inside a solid geom: 1484
unknown NOT inside a solid geom:                    14
```

And the 8 live frontier regions, with how many of their unknown neighbours are
inside solid geometry:

```
frontier (213, 36)  world=(17.6,-17.8)  unknown_nbrs=1  inside solid=0
frontier (90, 135)  world=(-7.0,  2.0)  unknown_nbrs=2  inside solid=1
frontier (137,135)  world=( 2.4,  2.0)  unknown_nbrs=4  inside solid=3
frontier (147,187)  world=( 4.4, 12.4)  unknown_nbrs=1  inside solid=0
frontier (173,187)  world=( 9.6, 12.4)  unknown_nbrs=2  inside solid=0
frontier (158,188)  world=( 6.6, 12.6)  unknown_nbrs=2  inside solid=2
frontier (217,188)  world=(18.4, 12.6)  unknown_nbrs=3  inside solid=3
frontier (35, 223)  world=(-18.0,19.6)  unknown_nbrs=1  inside solid=0
```

### Leading hypothesis (NOT yet confirmed)

**99% of the remaining unknown space is wall interior, which no scan can ever
resolve** — rays stop at the wall surface, so the cells behind that surface stay
at p=0.5 forever. A frontier is "known-free cell adjacent to unknown", so every
free cell facing a wall interior is a *permanent* frontier. At `resolution 0.2`
with `clearance_radius 0.20` the inflation radius is only 1 cell, so many of
these permanent frontiers remain *reachable*, get assigned, and the swarm chases
them forever. `small_indoor` escapes this because at `resolution 0.1` the
inflation radius is 2 cells, which makes the equivalent frontiers unreachable —
so every drone ends up unassigned and `is_complete` goes True.

That would make the difference between the two scenarios a direct consequence of
the resolution/clearance ratio, not of scene size.

### What is NOT yet established

- Whether a drone standing *on* one of these frontiers fails to clear it (the
  hypothesis predicts yes; not directly observed).
- Why the row-187 frontiers are transient rather than permanent — some cells
  flip between classified and unknown, which the coverage oscillation
  (97.587%–97.619%) corroborates. Likely log-odds drifting across the 0.4/0.6
  band at grazing incidence, but unverified.
- Whether the 14 non-solid unknown cells matter at all.

### Candidate directions (deliberately not acted on)

Do not pick one without finishing Phase 1 — each of these is a different root
cause and only one can be right.

1. **Frontier detection should not treat permanently-unobservable space as a
   frontier.** The most likely correct fix, and it belongs in `mapping`, not in
   the CLI. Needs a definition of "unobservable" that does not require
   ground-truth geometry.
2. **The coordinator should detect no-progress and stop.** Cheap, but it is a
   symptom fix — it would make the tick cap tidier without addressing why the
   swarm has nothing to do yet will not admit it.
3. **The resolution/clearance ratio.** `feature-5` notes the geometry supports
   `resolution: 0.1` unchanged. That would raise inflation to 2 cells and may
   make the scenario terminate the way `small_indoor` does — but it is a
   20× cost increase on the clearance mask and does not fix the underlying
   modelling question.

**Consequence for the feature either way:** acceptance test 11 (`large_indoor`,
3 drones) cannot assert clean termination until this is settled. Coverage ≥95%
already passes (97.6%).

---

## Remaining work

1. **Settle Finding 2.** Blocks acceptance test 11.
2. **Add the `acceptance` pytest marker** to `pyproject.toml`. It does not exist
   — the repo has `slow`, `sprint(n)`, `regression`, `progression`, `sanity`.
   Acceptance tests should still carry `sprint(2)`: the marker selects *cost*,
   the sprint tag selects *era*, and `tests/conftest.py` warns about any test
   carrying neither.
3. **Generalise the Feature 5 truth-grid helper.** `box_geoms` and `truth_grid`
   in `tests/unit/test_simulation/test_large_indoor.py` are hardcoded to
   large_indoor's resolution/origin/cell count. Acceptance test 12 (map accuracy
   ≥98%) needs them parameterised by `MapConfig` — suggest extracting to a
   shared `tests/scene_truth.py` and updating the Feature 5 module to import it.
4. **Acceptance tests 11–13** — `tests/acceptance/` is scaffolded but empty.
   Drive the tick loop via `build_mission`, which exists for this.
5. **Scaling KPI test 14** — ticks-to-95% on `small_indoor`, `--drones 1` vs 3,
   assert ≥1.5×, report the measured ratio so a near-miss is legible.
6. **`docs/design.md`** — written from scratch, own commit, last.
7. **Manually verify `--view`** with 3 drones on `large_indoor`. Never run; this
   session has no display. It is in the spec's "Done when".
8. **Merge `feat/sprint2-integration` → `sprint-2`** when the feature is done
   (the user's instruction; not done yet).

### Runtime budget warning

`tests/integration/test_e2e.py` alone is **108 s**; the full suite is 113 s.
CLAUDE.md budgets <5 min for the per-commit gate. A `large_indoor` 3-drone
acceptance run is **279 s on its own** — it must not land in the per-commit
gate. This is exactly what the `acceptance` marker is for.

---

## Useful commands

```bash
uv run pytest -q                      # full suite (113 s)
uv run pytest tests/unit -q           # fast, no MuJoCo scenes
uv run ruff check . && uv run ruff format --check . && uv run mypy

# Reproduce Finding 2
uv run python -c "
from pathlib import Path
from swarm_mapping.config.loader import load_config
from swarm_mapping.cli import build_mission, coverage_fraction
cfg = load_config(Path('scenarios/large_indoor/config.yaml'))
m = build_mission(cfg)
while not m.master.is_complete and m.master.tick_count < 1200:
    m.master.tick()
print(m.master.tick_count, coverage_fraction(m.mapper.grid), len(m.mapper.get_frontiers()))
"

swarm-mapping --config scenarios/small_indoor/config.yaml --drones 1 --verbose
```

Note the CLI now exits **1** when a mission hits its tick cap — so
`large_indoor` currently exits non-zero until Finding 2 is resolved.
