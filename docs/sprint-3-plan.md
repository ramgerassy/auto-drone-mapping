# Sprint 3 Plan — Robustness: the swarm survives losing a drone

Master plan for Sprint 3. Per-feature detailed plans live in
[`docs/sprint-3/`](sprint-3/) and are written (and reviewed) as each feature
starts. This file is the index and the source of truth for scope, decisions,
and progress.

## Goal

A drone can fail mid-mission — go silent, or stop responding to motion
commands — and the swarm notices from symptoms alone, releases the failed
drone's frontier within the latency KPI, routes around the wreck, and still
finishes the map. An operator console lets an end user run any scenario —
including new rooms they upload — and browse every run's map, drone paths, log
and metrics without touching the command line.

## Scope

**In** (from CLAUDE.md's Sprint 3 definition):
- `coordination`: heartbeats, stuck-drone detection, frontier reclaim.
- The dashboard, **as specified in D2**: an operator console (run, upload and
  validate scenarios, choose allocation variant, headless or MuJoCo view, run
  history with maps, paths, logs and metrics).
- Failure-injection scenarios.

**Also in, by request (2026-09-21):** promoting finished-sprint tests to
regression and fixing the CI gates so the promotion does what it claims — Task 0
below. CLAUDE.md lists CI as out of scope for this sprint; this is the one
exception, and it goes first because every later feature's pushes depend on it.

**Out (explicit):** acceptance tests, other CI changes, a second `FrontierStrategy`
(closed in Sprint 2.5 — see `docs/progress.md`, 2026-09-21), drone
recovery/rejoin, wind, outdoor scenario, physics-based flight.

**Carried, not in scope:** README still describes the Sprint-1 patrol;
`greedy` vs `global` allocation question from Sprint 2.5.

## Task 0 — Promote finished tests to regression, and make the CI gates honest (do first)

### What's wrong

`tests/conftest.py` derives `regression` / `progression` from each test's
`sprint(N)` marker against `CURRENT_SPRINT` — and **`CURRENT_SPRINT` is still
`2`**. It was never bumped when Sprint 2 closed, nor after Sprint 2.5. So all
283 Sprint-2 and 2.5 tests are still "progression" and **every push this sprint
has run 103 of the 365 tests** (Sprint 1's 82 plus sanity).

A bare bump would open a different hole, because the two gates split the suite
rather than cover it:

| Gate | Selection | Today (`CURRENT_SPRINT=2`) | After a bare bump to 3 |
| --- | --- | --- | --- |
| push | `regression or sanity` | 103 / 365 | 365 — **including the 7 acceptance tests**, which `pyproject.toml` says never run on push |
| PR → main | `progression or sanity` | 288 / 365 | **Sprint-3 tests + sanity only** — the sprint PR to `main` skips all earlier work |

**No CI gate has ever run the whole suite.** It went unnoticed because Sprint 1
is small. Separately, the comments contradict each other: `conftest.py` and the
`pyproject.toml` marker help call regression the *PR* gate and progression the
*per-commit* gate; `ci.yml` does the reverse.

### The fix

The split is worth keeping — a red CI step should say whether shipped behaviour
broke or in-flight work isn't done. But it should *label* results, not *drop*
tests. So both gates run everything they can afford, as separate steps:

| Gate | Steps |
| --- | --- |
| push (any branch) | **Regression** `-m "regression and not acceptance"` → **Progression** `-m "progression and not acceptance"` |
| PR → main | the same two, then **Acceptance** `-m acceptance` |

This matches CLAUDE.md's own CI line — unit + integration on every commit,
acceptance on PRs to main — which the current gates never did. Non-acceptance is
~90 s locally, inside the 5-minute budget.

### Steps

- [ ] **`tests/conftest.py`** — `CURRENT_SPRINT = 3`. Rewrite the header comment
  to match the gates above. Add a guard in `pytest_collection_modifyitems`: a
  test marked with a sprint **ahead of** `CURRENT_SPRINT` raises a collection
  error ("bump CURRENT_SPRINT at sprint kickoff"). That is the moment the bump is
  forgotten, so that is where it gets caught — it would have caught this the
  first time a `sprint(3)` test was written.
- [ ] **`pyproject.toml`** — correct the `regression` / `progression` marker help
  text to "prior sprints, must not break" / "current sprint, work in flight",
  without the swapped gate names.
- [ ] **`.github/workflows/ci.yml`** — replace the two conditional pytest steps
  with the three above. Coverage (`--cov`) stays on the Regression and
  Progression steps; `--cov-append` on the second so the report covers both.
  Also lint `benchmarks/` (`ruff check src/ tests/ benchmarks/`), since it now
  holds code with tests of its own.
- [ ] **Verify locally**, and record the counts in the commit message:
  - `pytest --collect-only -q -m "regression and not acceptance"` → every
    Sprint 1, 2 and 2.5 non-acceptance test (358 at time of writing).
  - `pytest --collect-only -q -m "progression"` → 0 until Feature 9 adds
    `sprint(3)` tests. An empty step must pass, not error: pytest exits 5 on "no
    tests collected", so the progression step accepts exit code 5.
  - A throwaway `sprint(4)` test triggers the new guard.
- [ ] **Verify on CI**: push the branch; the Regression step must run the full
  pre-Sprint-3 suite and pass.
- [ ] **Sprint close, going forward**: the kickoff of every sprint bumps
  `CURRENT_SPRINT`. Added to the Workflow section below so it isn't forgotten
  again.

## How failure works — the one idea the sprint hangs on

**Injection lives in `simulation`; detection lives in `coordination`; the
master never reads the failure schedule.** A coordinator that is told which
drone failed has detected nothing. It has to infer failure from what a real
ground station would see:

| Failure mode | What the simulator does | What the master observes | Diagnosis |
| --- | --- | --- | --- |
| `silent` — crash, comms loss | stops the drone's heartbeat; ignores motion commands | no heartbeat → no scan that tick | `LOST` after `heartbeat_timeout_ticks` missed heartbeats |
| `stuck` — motor fault | heartbeat and sensor continue; ignores motion commands | commanded a step, localizer says it didn't happen | `STUCK` after `stuck_timeout_ticks` unrealized moves |

That second row needs the master to **read the drone's pose back** from the
localizer after moving it. Today it never does: `_move` teleports and writes
the commanded cell straight into `DroneState.cell`, trusting the command. That
is a latent bug independent of this sprint — the master's belief about where a
drone is should come from the `Localizer`, not from its own command log. For a
healthy drone the read-back is exact (cell centre → `world_to_grid` round-trips),
so every no-failure run must reproduce its Sprint 2.5 tick count **exactly**.
That is the sprint's zero-regression gate.

## Decisions

Resolved 2026-09-21.

| # | Question | Decision |
| --- | --- | --- |
| D1 | What is a second? No tick duration exists — `tick()` never advances MuJoCo time and a step is a one-cell teleport. | **`drones.cruise_speed`** (m/s) in config; `tick_seconds = map.resolution / cruise_speed`. Timeouts stay in ticks, matching the existing `*_ticks` convention. At 1.0 m/s `large_indoor` is 0.2 s/tick, so the KPI is **< 10 ticks**. *Config schema change.* |
| D2 | What is the dashboard? | **An operator console**, not an overlay — see below. |
| D3 | How does health reach the UI? | **`health: DroneHealth = ACTIVE` on the frozen `DroneState`.** `Coordinator.drone_states` already exposes it, so the seam's signature is unchanged. |
| D4 | Wreck in a doorway? | **In scope.** Plan against a planning-only overlay with wreck footprints marked occupied; never written to the map. |
| D5 | `.gitattributes` | **At kickoff** (`* text=auto eol=lf`). CRLF churn has now hit three times, most recently this morning's pull. |
| D6 | A failure schedule names a drone not in this run | **Fail fast** with `ValueError`. |

### D2 — what the console must do (your spec)

1. A menu to pick which scenario to run.
2. Add new rooms by uploading config files (and the scene they use); validate
   each, including that it can spawn **up to 5 drones**.
3. Choose the allocation variant: **baseline, A, B, or A+B**.
4. Run **headless** or **with the MuJoCo view**.
5. Log every execution.
6. Browse results comfortably: output map, each drone's path, the log, and the
   run's values, with runs comparable side by side.

**Technology: Streamlit, as an optional `ui` extra.** You left the tool open.
Streamlit gives file upload, selectors, tables, images and log panes in a few
lines each, runs in the browser (so no WSL display issues beyond the MuJoCo
window itself), and ships `AppTest` for headless UI tests. It is a **heavy new
dependency**, so it goes in `[project.optional-dependencies] ui`, never in the
core install — the simulator, CI and Docker image stay exactly as lean as today.
The zero-dependency alternative is Tkinter (stdlib, with Pillow for images); it
works, but costs roughly twice the code for a dated result. Veto here if you'd
rather have that.

Three interpretations worth checking:

- **"Frontier strategy A / B / A+B / baseline" are the allocation variants**
  from `benchmarks/strategy_matrix.py`, and the console will label them that
  way. Sprint 2.5 showed `FrontierStrategy` is not what they change — calling
  them strategies would bring that confusion back.

  | Label | `coordination.assignment` | `target_tolerance_cells` |
  | --- | --- | --- |
  | baseline | `greedy` | 0 |
  | A | `global` | 0 |
  | B | `greedy` | 3 |
  | A+B | `global` | 3 |

- **"Can spawn up to 5 drones"** means the config declares five start positions
  and **every one is valid** — in bounds, clear of geometry, and far enough from
  the others — so any run from 1 to 5 drones works. A config with fewer is
  rejected with a message saying how many it has.
- **A new room is a config plus its MJCF scene.** A config alone can only point
  at a scene that already exists, which isn't a new room.

**Architecture.** The console is a new top layer, `swarm_mapping.app`, sitting
beside `cli.py` and consuming it; nothing in the core imports it. That is a new
module (CLAUDE.md: flag) — no existing one fits, because `visualization` is
read-only by definition and a console *starts* runs. It launches each run as a
**subprocess of the existing CLI**, so a run from the console is byte-identical
to the same run from a terminal, the MuJoCo viewer gets its own process, and the
UI stays responsive.

## Feature breakdown

Each feature is a branch off `sprint-3`, merged back when green. The whole
sprint opens **one** PR `sprint-3 → main` at the end.

| # | Branch | Deliverable | Depends on | Status |
| --- | --- | --- | --- | --- |
| 0 | *(on `sprint-3`)* | **Task 0 first:** `CURRENT_SPRINT = 3`, forgotten-bump guard, CI gates run the whole suite (see Task 0). Then kickoff: commit CLAUDE.md sprint definition and this plan, `.gitattributes` (D5) | — | ☐ |
| 9 | `feat/failure-injection` | `simulation`: `FailureMode`, `SimulationEngine.fail_drone` / `heartbeat`, `FailureInjector`. `config`: optional `failures:` section. `cli`: `Mission.tick()` applies the schedule | 0 | ☐ — [plan](sprint-3/feature-9-failure-injection.md) |
| 10 | `feat/failure-handling` | `coordination`: `DroneHealth` on `DroneState` (D3), heartbeat tracking, pose read-back + stuck detection, frontier reclaim, wreck-aware planning overlay (D4), all-drones-lost termination | 9 | ☐ — plan written at start |
| 11 | `feat/failure-scenarios` | `scenarios/failure_injection/` (silent) and `scenarios/failure_stuck/` (stuck), both on `large_indoor.xml`; `drones.cruise_speed` (D1); fast integration test on `small_indoor`; `benchmarks/failure_recovery.py` measuring the latency KPI | 10 | ☐ — plan written at start |
| 12 | `feat/run-records` | Everything the console needs that is *not* UI, usable from the CLI on its own: per-run **JSON Lines log file** (finally meeting CLAUDE.md's logging convention); `run.json` summary (scenario, variant, drones, seed-equivalent inputs, ticks, coverage, blocked, failure events and latency, per-drone path stats, wall time); paths and route PNGs always written; `validate_scenario()` (D2 rules); `list_runs()` / `load_run()` | 9 | ☐ — plan written at start |
| 13 | `feat/operator-console` | `swarm_mapping.app` Streamlit console (optional `ui` extra): **Run** (scenario, 1–5 drones, variant, headless/view, live log tail), **Scenarios** (list, upload config + MJCF, validate, save), **History** (runs table, map, per-drone routes, log, metrics, compare two runs). `AppTest` smoke tests | 12 | ☐ — plan written at start |
| 14 | `feat/sprint3-docs` | `design.md` failure-handling and console sections, KPI table; `progress.md` entry with measured latency; README "how to run" rewritten around the console; fix CLAUDE.md's dependency graph (`perception → mapping` is drawn backwards) | 11, 13 | ☐ |

Feature 12 depends only on 9, so it can run **in parallel with 10 and 11**.
13 follows 12. If the day runs short, the console is the feature to cut down,
not the failure handling: 12 alone already gives the CLI per-run logs and
records, and a console with **Run + History** but no upload is still usable.

### Feature 10 — test cases to review now

Listed here so they can be discussed before the detailed plan is written
(CLAUDE.md: discuss test cases before coding). All use a real
`SimulationEngine` on inline or small MJCF — never a MuJoCo mock.

**Heartbeat / `LOST`**
1. A silent drone contributes **no** new cells to the map from the tick it goes silent — before it is declared lost, not only after.
2. It is declared `LOST` on exactly the `heartbeat_timeout_ticks`-th missed heartbeat — not one tick early, not one late.
3. A healthy drone is never declared lost over a full mission.

**Stuck / `STUCK`**
4. A stuck drone is declared `STUCK` after `stuck_timeout_ticks` commanded-but-unrealized moves.
5. **A drone yielding to a teammate is not stuck.** Waiting in `resolve_moves` is not a commanded move; it must never count. This is the test most likely to catch a real bug.
6. A stuck drone's scans still reach the map (its sensor works) until it is declared, and after.

**Reclaim**
7. The failed drone's frontier is back in the pool the tick it is declared, and the same tick's assignment pass can hand it out.
8. A failed drone is never assigned again, never sent home, never counted as "idle".
9. A failed drone's cell stays in `resolve_moves` as a static body: no teammate ever comes within `min_separation` of the wreck.
10. Every drone failed → the mission terminates, reported `blocked` if frontiers remain. No infinite loop, no crash on an empty swarm.

**Wreck-aware planning (D4)**
11. A wreck in a corridor: teammates route around it rather than thrashing — bounded number of target changes.
12. The wreck is **never** written to the map — the exported grid is unaffected by the overlay.

**Zero regression**
13. `small_indoor` with no failures reproduces its Sprint 2.5 tick count exactly, with pose read-back on.
14. Diagnosis is only from symptoms: `CentralizedMaster` takes no failure schedule, and its `DroneHealth` values match the injected `FailureMode`s in a scripted run.

### Features 12–13 — test cases to review now

**Run records (12)** — pure Python, no UI.
1. Every run writes `run.json` and `log.jsonl` into its own directory; two runs never share one.
2. `run.json` reproduces the run: it holds every input that changes the outcome (scenario, config snapshot, drones, variant) — enough to re-run it and get the identical map.
3. Each line of `log.jsonl` is valid JSON with `event` and `tick` keys; `failure_injected` and `drone_failed` events appear in a failure run.
4. `list_runs()` returns runs newest-first and skips a directory with a corrupt or missing `run.json` rather than crashing the history page — and says which it skipped.
5. A headless run and a `--view` run of the same inputs produce identical `run.json` metrics (the viewer is view-only).

**Scenario validation (12)**
6. Each shipped scenario validates clean.
7. Fewer than five start positions → rejected, message states the count.
8. A spawn inside a wall → rejected, naming the spawn index. Checked by MuJoCo contacts on the real scene, not by re-parsing the XML.
9. Two spawns closer than `min_separation` → rejected.
10. A config whose `scene.path` does not exist → rejected before MuJoCo is touched.
11. An uploaded name that would overwrite a shipped scenario, or escape the scenarios directory (`../`), → rejected.

**Console (13)** — Streamlit `AppTest`, headless; skipped when the `ui` extra is not installed.
12. Each page renders without exception on an empty run history.
13. The variant selector maps baseline / A / B / A+B to exactly the four `(assignment, tolerance)` pairs above.
14. The Run page builds the same CLI arguments a user would type — asserted on the argument list, without launching MuJoCo.
15. An invalid upload shows the validator's messages and saves nothing.

## Locked decisions

1. **`CURRENT_SPRINT = 3`** in Task 0: Sprint-2 and 2.5 tests become *regression*; new tests are tagged `pytest.mark.sprint(3)` → *progression*. Both run on every push.
2. **Failure is permanent.** No recovery, no rejoin. YAGNI — nothing in the KPIs needs it.
3. **Two modes, `silent` and `stuck`**, and both ignore motion commands. They differ in what they still *report*, which is exactly what makes them need different detectors.
4. **`config` stays dependency-free.** It validates `mode` as a string in `{"silent", "stuck"}`; `cli` maps it to `simulation.FailureMode`. `simulation` never imports `config`.
5. **The master trusts the localizer.** `DroneState.cell` comes from the read-back pose every tick.
6. **Deterministic by construction.** Failures fire at scripted ticks; no RNG anywhere in the failure path.
7. **The console never runs the mission in-process.** It shells out to the CLI, so there is exactly one way a mission executes, and the UI cannot perturb it.

## KPIs this sprint closes

- **Tier 2 — frontier reassignment latency < 2 s after drone failure.** Measured as ticks from injection to the failed drone's claim being released (available to that same tick's assignment pass), × `tick_seconds` (D1). Reported for both failure modes on `large_indoor`. Detection latency *is* the reassignment latency here, because release and reassignment happen in the tick that declares the failure.
- **Tier 1 — coverage ≥ 95% indoor still holds with a drone lost** mid-mission (the swarm finishes the map; it may take longer).
- **Tier 1 — zero collisions**, now including with the wreck.

Acceptance tests are out of scope this sprint, so the KPIs are measured by
`benchmarks/failure_recovery.py` and recorded in `docs/progress.md`, with a
fast integration test on `small_indoor` guarding the behaviour on every commit.

## Cross-cutting constraints

- **Determinism (hard requirement).** Sorted drone-id order in every new loop; no set iteration in decision paths.
- **Dependency direction.** `coordination → simulation` (heartbeat, pose) is already an allowed edge. `app` is a new top layer beside `cli` and depends on it; **nothing imports `app`**. Nothing new crosses upward.
- **Core stays UI-free.** Streamlit lives only in the optional `ui` extra; importing `swarm_mapping` without it must still work.
- **Tests first.** Test cases reviewed before implementation (above, for Feature 10).
- **Coverage ≥ 70%** on `coordination`.

## Workflow

```
main
 └── sprint-3                      (integration branch)
      ├── (kickoff commit)
      ├── feat/failure-injection   → merge into sprint-3
      ├── feat/failure-handling    → merge   ┐
      ├── feat/run-records         → merge   ┘ parallel
      ├── feat/failure-scenarios   → merge   ┐
      ├── feat/operator-console    → merge   ┘ parallel
      └── feat/sprint3-docs        → merge into sprint-3
 sprint-3 → ONE PR → main
```

**Every sprint kickoff bumps `CURRENT_SPRINT`.** It was missed at the close of
Sprint 2 and again after 2.5, which silently dropped 283 tests from every push.
Task 0's guard now fails collection if it is missed again.
