# Feature 13 — Operator console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser console where an operator runs any scenario (drone count, allocation variant, headless or with the MuJoCo view), uploads and validates new rooms, and browses every past run — map, per-drone routes, log, metrics — comparing two runs side by side.

**Architecture:** Three new files in the `swarm_mapping.app` top layer, plus one small pure helper module. `runner.py` (pure, no streamlit) turns a `RunRequest` into the exact CLI argument list, names the run directory, and launches the CLI as a subprocess. `history.py` (pure) shapes `RunRecord`s into table rows and a two-run comparison. `console.py` is the Streamlit UI — thin: every decision it shows is made in `runner`, `history`, `records` or `validation`. `launch.py` is the `swarm-console` entry point. Nothing in the core imports `app`; the console never imports or calls `run_pipeline` — a console run is the CLI run, byte for byte.

**Tech Stack:** Python 3.13, Streamlit ≥ 1.40 (optional `ui` extra; pandas comes with it), pytest + Streamlit `AppTest`, ruff, mypy (strict).

**Spec:** [`docs/sprint-3-plan.md`](../sprint-3-plan.md) — "D2 — what the console must do", the Feature 13 row, "Features 12–13 — test cases" 12–15, Cross-cutting constraint 7; controller rulings C1–C10 (Feature 13 brief), restated below. Builds on [Feature 12](feature-12-run-records.md): `records.list_runs` / `load_run` / `VARIANTS`, `app.validation.validate_scenario` / `install_scenario`.

## Rulings (binding)

| # | Ruling |
| --- | --- |
| C1 | `app/runner.py` pure: `RunRequest` (frozen: config_path, scenario, drones, variant, view), `build_argv(request, output_dir)`, `new_run_dir(root, request, now)`, `launch(request, root) -> (Popen, Path)`. `app/console.py` thin Streamlit UI. `app/launch.py` `main()` runs `streamlit run <console.py>`; `swarm-console` script. |
| C2 | `[project.optional-dependencies] ui = ["streamlit>=1.40"]`, `uv.lock` updated. Only `console.py` and `launch.py` import streamlit (`launch.py` lazily, printing `install the console with: uv sync --extra ui` if missing). A test proves `swarm_mapping`, `app.runner`, `app.validation` import with streamlit blocked. |
| C3 | Runs root `runs/` under the cwd, git-ignored. Directory `YYYYmmdd-HHMMSS_<scenario>_<variant>_<N>d`, `A+B` → `AB`; collisions get `-2`, `-3`, … Wall-clock only in names, never a mission input. |
| C4 | Subprocess `[sys.executable, -m, swarm_mapping.cli, --config, P, --output, D, --drones, N, --assignment, A, --target-tolerance, T]` + `--view` iff requested; (A, T) from `VARIANTS`. stdout+stderr → `<run dir>/console.txt`. |
| C5 | Sidebar pages **Run**, **Scenarios**, **History** (content below). |
| C6 | Roots from `SWARM_RUNS_ROOT`, `SWARM_SCENARIOS_ROOT`, `SWARM_ASSETS_DIR`; defaults `runs/`, `scenarios/`, bundled assets. Tests always use tmp dirs. |
| C7 | `tests/unit/test_app/test_runner.py` always runs; `tests/unit/test_app/test_console.py` `importorskip("streamlit")`, `AppTest`, never launches MuJoCo (monkeypatch **our** `runner.launch`). |
| C8 | CI unchanged. |
| C9 | No failure-injection controls; failure scenarios run like any other. |
| C10 | Stock widgets, no custom CSS/JS, no dependency beyond streamlit (pandas allowed for tables). |

## Global Constraints

- `from __future__ import annotations`; Google docstrings on every module and public function/class; type hints; mypy strict on `src/`. Streamlit gets a targeted `[[tool.mypy.overrides]] ignore_missing_imports` so CI (no extra) type-checks `console.py` too.
- New test modules carry `pytestmark = pytest.mark.sprint(3)`.
- Determinism: the console adds nothing to a mission. The subprocess argv is a pure function of the request and the directory; the timestamp is only in the directory name.
- Never open a viewer or start a real mission in a test.
- Pre-commit, with `uv sync --extra ui` locally so AppTest runs: `uv run ruff check src/ tests/ benchmarks/ && uv run ruff format --check src/ tests/ benchmarks/ && uv run mypy && uv run pytest -q -m "not acceptance"`.

## CI note (C8 — raise with the user)

CI runs `uv sync` **without** the `ui` extra, so `tests/unit/test_app/test_console.py` **skips in CI** (it `importorskip`s streamlit); `test_runner.py` and the no-streamlit import test always run. The console's AppTest coverage (sprint tests 12, 14, 15) is therefore local-only until CI syncs with `--extra ui` — a one-line CI change deliberately left out of this feature.

## File map

| File | Change | Responsibility |
| --- | --- | --- |
| `pyproject.toml`, `uv.lock` | modify | `ui` extra, `swarm-console` script, mypy override |
| `.gitignore` | modify | `runs/` |
| `src/swarm_mapping/app/runner.py` | create | roots, `RunRequest`, `build_argv`, `new_run_dir`, `launch`, `scenario_choices` |
| `src/swarm_mapping/app/history.py` | create | history table rows, two-run comparison, log tail/filter helpers |
| `src/swarm_mapping/app/console.py` | create | the Streamlit UI |
| `src/swarm_mapping/app/launch.py` | create | `swarm-console` entry point |
| `tests/unit/test_app/conftest.py` | create | `make_run`: hand-written run dirs via `write_run_record` |
| `tests/unit/test_app/test_runner.py` | create | argv, variants, naming, roots, scenario menu, no-streamlit import |
| `tests/unit/test_app/test_launch.py` | create | `swarm-console`: install hint, streamlit argv |
| `tests/unit/test_app/test_history.py` | create | rows, comparison, log reading |
| `tests/unit/test_app/test_console.py` | create | `AppTest` smoke and behaviour |

---

### Task 1: `runner` — the request becomes the CLI command

**Files:** create `src/swarm_mapping/app/runner.py`, `tests/unit/test_app/__init__.py`, `tests/unit/test_app/test_runner.py`.

**Interfaces (produces):**
- `runs_root() / scenarios_root() / assets_dir() -> Path` — env var or default (C6).
- `RunRequest(config_path: Path, scenario: str, drones: int, variant: str, view: bool)` — frozen; `__post_init__` rejects an unknown variant or `drones < 1` with `ValueError`.
- `build_argv(request, output_dir) -> list[str]` — C4 exactly, flag spellings from `cli.main`.
- `run_dir_name(request, now) -> str`; `new_run_dir(root, request, now) -> Path` — creates the directory (`mkdir(exist_ok=False)`, so the collision check and the claim are one step).
- `launch(request, root, now=None) -> tuple[Popen[bytes], Path]` — `console.txt` gets stdout+stderr.
- `ScenarioChoice(name, config_path, spawns: int | None, problems: list[str])`, `scenario_choices(root) -> list[ScenarioChoice]` — every `<root>/*/config.yaml`, sorted by name, validated with `validate_scenario` (default rules).

- [x] **Step 1: tests** — sprint test 13: one parametrised case per label asserting `--assignment`/`--target-tolerance` equal `VARIANTS[label]`, and that `VARIANTS` has exactly the four pairs `{baseline: (greedy, 0), A: (global, 0), B: (greedy, 3), A+B: (global, 3)}`; the full argv for one request; `--view` present iff `view`; argv parses with the CLI's own flag names (each flag appears in `cli.py`'s `add_argument` set); unknown variant / zero drones rejected; name format `20260921-143005_small_indoor_AB_3d`; collision → `-2`, then `-3`; `new_run_dir` creates the root; env-var roots and defaults; `scenario_choices` lists valid and invalid scenarios with problems and spawn counts; `launch` on a trivial request is **not** tested with the real CLI (no mission in tests) — instead `launch` is tested with `build_argv` monkeypatched to `[sys.executable, "-c", "print('hi')"]` so the `console.txt` capture is real. C2: a fresh interpreter with a `sys.meta_path` finder that raises `ImportError` for `streamlit` imports `swarm_mapping`, `swarm_mapping.app.runner`, `swarm_mapping.app.history`, `swarm_mapping.app.validation` and asserts `"streamlit" not in sys.modules`.
- [x] **Step 2: run, see them fail.**
- [x] **Step 3: implement.**
- [x] **Step 4: green; commit** `feat(app): runner — a console run is the CLI command`.

### Task 2: `history` — what the History page shows, as data

**Files:** create `src/swarm_mapping/app/history.py`, `tests/unit/test_app/test_history.py`.

**Interfaces (produces):**
- `HISTORY_COLUMNS`; `history_row(record) -> dict[str, object]` — started, run (dir name), scenario (config's parent dir name), variant, drones, ticks, coverage, blocked, succeeded, wall seconds, failures detected = `event_counts.get("drone_failed", 0)`.
- `compare(a, b) -> list[MetricDelta]` — `MetricDelta(metric, a, b, difference)`; numeric metrics get `b − a`, others `None`.
- `read_log(path, last=None) -> list[dict[str, Any]]` — parsed `log.jsonl` lines; a partial final line (a run still writing) is skipped, never raised.
- `route_images(record) -> list[tuple[str, Path]]` — `(drone id, route_drone_<id>.png)` for each drone in `outputs.paths` whose file exists, in drone-id numeric order.

- [x] **Step 1: tests** — row fields from a hand-built `RunRecord`; failures detected read from `event_counts`, 0 when absent; comparison differences and non-numeric rows; `read_log` tail, partial line, missing file → `[]`; `route_images` order `2 < 10` and missing files skipped.
- [x] **Step 2–4:** fail, implement, green; commit `feat(app): history rows, run comparison and log reading`.

### Task 3: `console` and `launch` — the Streamlit UI

**Files:** create `src/swarm_mapping/app/console.py`, `src/swarm_mapping/app/launch.py`, `tests/unit/test_app/test_console.py`; modify `pyproject.toml`, `uv.lock`, `.gitignore`.

Pages (sidebar radio "Page"):
- **Run** — scenario select over valid `scenario_choices`; invalid ones listed with their problems in an expander and not selectable. Drones slider `1..spawns` (default all). Radio "Allocation variant" (default: the scenario's configured variant when it is one of the four, else baseline), with a caption table of the (assignment, tolerance) pairs. Radio "Mode": Headless / With MuJoCo view. **Run** button disabled while the stored process is running. While running, a fragment refreshed every second shows the tick (last log line's `tick`) and the last 20 log lines. On exit: exit code, `run.json` summary (or the `console.txt` tail if there is none), `map.png`, "see History".
- **Scenarios** — table (name, valid, spawn positions, problems); upload form (name, config `.yaml`, scene `.xml`) → `install_scenario(name, config, scene, scenarios_root(), assets_dir())` → problems as errors, or success.
- **History** — `list_runs` table; skipped runs as warnings. Select a run: metrics, `map.png`, a tab per drone with its route PNG, log viewer with an event multiselect filter, `console.txt` in an expander. Compare: two selects → metric table with the difference, both maps side by side.

`launch.main()`: if `importlib.util.find_spec("streamlit")` is None, print the install hint to stderr and exit 1; else run `[sys.executable, -m, streamlit, run, <console.py>, *extra args]` and exit with its code.

- [x] **Step 1: tests (skip without streamlit)** — sprint test 12: each page renders without exception on an empty runs root. Sprint test 14: select a scenario, 2 drones, variant B, "With MuJoCo view", click Run with `runner.launch` monkeypatched to capture the request → `build_argv(captured, dir)` equals the argv typed by hand. Sprint test 15: an invalid upload (three spawns) shows the validator's message as an error and both tmp roots stay empty — the file uploader is replaced by a stand-in returning the upload's bytes, because `AppTest` cannot drive `st.file_uploader`. History renders a hand-written run directory (written with `write_run_record`), including its per-drone tab and compare section. Launcher: missing streamlit prints the hint.
- [x] **Step 2–4:** fail, implement, green (with `--extra ui`); commit `feat(app): operator console — run, scenarios, history`.

### Task 4: smoke check, report

- [x] Start `uv run --extra ui streamlit run src/swarm_mapping/app/console.py --server.headless true --server.port 8599` in the background, `curl -s localhost:8599/_stcore/health` → `ok`, stop it.
- [x] Full pre-commit gate; push the branch.

## Sprint test mapping

| Sprint test | Test |
| --- | --- |
| 12 | `test_console.py::TestPagesRender` |
| 13 | `test_runner.py::TestVariants` |
| 14 | `test_console.py::TestRunPage::test_run_builds_the_argv_a_user_would_type` |
| 15 | `test_console.py::TestUpload::test_invalid_upload_shows_problems_and_writes_nothing` |

## As built

- Tasks 1 and 2 landed as one commit (the no-streamlit import test covers both modules). The launcher tests live in `test_launch.py` so they arrive with `launch.py`.
- The live panel reads progress by event name (`mission_progress`'s `coverage`) and the log's `tick` field, never message text — matching Feature 12's review-round-1 event names.
- `st.fragment` is applied at the call site, not as a decorator: in CI (no extra) streamlit is untyped, and mypy strict rejects an untyped decorator. pandas joins streamlit in the mypy override (no stubs).
