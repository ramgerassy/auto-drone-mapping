# CLAUDE.md

This file provides guidance to Claude Code when working on this project.

---

## Project overview

A cooperative swarm of 1–5 drones that explore and map unknown environments in MuJoCo. The system uses ground-truth localization from the simulator and focuses on the autonomy problem: where drones go, how they coordinate, and how they handle failures. SLAM is explicitly out of scope.

**Course context:** final-semester workshop project (Open University, course 20973). Evaluated on system design, SOLID, tests, documentation, KPIs, and extensibility — not just on whether it works.

**Output:** 2.5D map (2D occupancy grid + per-cell height), exported as `.npz` (data) + `.png` (human inspection).

---

## Non-negotiable constraints

These shape every decision. Don't propose changes to them without flagging it explicitly.

- **Python 3.11+.** No other languages.
- **MuJoCo only.** Don't suggest Gazebo, PyBullet, Isaac, etc.
- **Ground-truth localization.** Drone pose comes from MuJoCo state. No SLAM. The `Localizer` interface exists so SLAM could be added later, but the implementation is `GroundTruthLocalizer`.
- **2.5D mapping only.** 2D occupancy grid + per-cell height. No 3D voxels. No point clouds.
- **Master-slave coordination.** One `CentralizedMaster` orchestrates all drones. No peer-to-peer, no distributed consensus, no auction protocols. The `Coordinator` interface exists for future P2P; don't implement it.
- **Synchronous tick loop.** Coordination steps the simulator, reads state, makes decisions, sends commands — sequentially. No threading, no async, no queues. This is required for determinism and reproducibility.
- **Determinism is a hard requirement.** Same config + same seed = identical run. Don't introduce nondeterministic operations (unordered dict iteration, set-based ordering, threading) without a strong reason.
- **1–5 drones.** Don't optimize for 50 or 500. The simple thing is correct.

---

## Architecture

Seven modules with one-way dependencies. Each module has a defined responsibility and a named interface (the SOLID seam).

| Module          | Responsibility                                                                  | Key interface(s)                  |
| --------------- | ------------------------------------------------------------------------------- | --------------------------------- |
| `config`        | Load + validate scenario config from YAML at startup                            | `ConfigLoader`                    |
| `simulation`    | Wrap MuJoCo; provide poses, ray-casts, motor commands                           | `Localizer`, `RayCaster`          |
| `perception`    | Convert raw ray data into world-coord observations                              | `Sensor`                          |
| `mapping`       | Maintain 2.5D global map via Bayesian log-odds update                           | `Mapper`                          |
| `planning`      | Stateless. Given map + drone state + claimed frontiers → return frontier + path | `FrontierStrategy`, `PathPlanner` |
| `coordination`  | Orchestrate mission — tick loop, assignments, heartbeats, failures              | `Coordinator`                     |
| `visualization` | Read-only renderer of state for human monitoring                                | `Renderer`                        |

### Dependency direction

Always one-way. Anything below depends only on things further down, never up:

```
coordination
    ├── planning
    └── simulation (for poses and commands)
planning
    └── mapping (read-only)
perception
    ├── simulation (for ray-casts)
    └── mapping (writes observations)
mapping
    └── (no dependencies on other domain modules)
visualization
    ├── mapping (read-only)
    └── coordination (read-only)
config
    └── (no dependencies; everyone reads it at startup)
```

If you find yourself adding an import that goes against this direction, stop and reconsider — it's usually a sign that responsibilities are mis-assigned.

### The four SOLID seams

These are extensibility points. Each has at least one concrete implementation, but the interface is what consumers depend on:

- **`Localizer`** — `GroundTruthLocalizer` is the only implementation. SLAM would be a second.
- **`Sensor`** — `Rangefinder` is the only implementation. A depth camera or stereo pair would be a second.
- **`FrontierStrategy`** — `NearestFrontier` is the primary. `InformationGainFrontier` is the stretch goal.
- **`Coordinator`** — `CentralizedMaster` is the only implementation. `DistributedAuction` would be a second (don't build it).

**When adding new behavior:** check if it fits behind one of these interfaces. If yes, prefer adding a new implementation over modifying the existing one (open-closed).

---

## Tooling and project conventions

- **Dependency management:** `uv`. All deps in `pyproject.toml` with locked versions.
- **Lint + format:** `ruff` (both linting and formatting — don't add black).
- **Types:** `mypy`. Run in CI. Add type hints to all module-level functions and class methods.
- **Tests:** `pytest`. Coverage target ≥70% on `mapping`, `planning`, `coordination`.
- **Pre-commit:** `ruff`, `mypy`, basic checks. Run on staged files.
- **CI:** GitHub Actions. Unit + integration on every commit (<5 min). Acceptance suite on PRs to main and nightly.
- **Container:** `Dockerfile` for headless CI/grading. Interactive dev runs locally.
- **Logging:** JSON Lines, per-run output directory. Don't use `print()` outside CLI entry points.
- **Config:** Single YAML per scenario, validated at startup, fail-fast on errors.

### Code style preferences

- Prefer composition over inheritance. Use Protocols (`typing.Protocol`) for interfaces unless ABCs add genuine value.
- Use dataclasses (or `attrs`) for state-bearing classes. Mark them `frozen=True` where mutation isn't needed.
- Type hints everywhere. `from __future__ import annotations` at the top of every file.
- Docstrings on every module and every public function/class. Google-style.
- Keep functions short and pure where possible. `planning` in particular should be stateless and a pure function of its inputs.
- No premature abstraction. Don't add an interface until there's a real second implementation or a clear test-double need.

---

## Testing philosophy

- **Unit tests come first.** When implementing a new module, write the test signatures from the design before the implementation. AI-generated code is fine, but I read and understand every line — tests prove I understand the contract.
- **Tests on core modules (`mapping`, `planning`, `coordination`) are required.** Other modules are tested mostly via integration.
- **Integration tests** validate cross-module behavior (e.g., perception → mapping pipeline). Run every commit.
- **Acceptance tests** are the four scenarios with expected outputs and tolerance bands. Compare coverage and timing against reference within ±X%, not exact match (floating-point order makes exact comparison flaky). Run on PRs and nightly, not every commit.
- **No mocking of MuJoCo for unit tests.** Use a tiny inline MJCF environment for tests that need the simulator. For pure-logic tests (planning, mapping updates), don't touch MuJoCo at all.

---

## Scenarios

Four reference scenarios live in `scenarios/`. Each is an MJCF file + a YAML config:

1. **Small indoor** — 20m × 20m, single room, few obstacles. Baseline.
2. **Large indoor** — 50m × 50m, corridors and doorways. Tests coordination.
3. **Outdoor** — 100m × 100m, scattered cylinders as obstacles. Tests open-terrain behavior.
4. **Failure injection** — same as large indoor, with one drone scripted to fail mid-mission. Tests recovery.

Drone count, wind profile, sensor config, seed, and algorithm parameters are all in the YAML — not hardcoded.

---

## KPIs

Don't pad the system with metrics. The committed ones are:

**Tier 1 (hard targets, pass/fail):**

- Coverage ≥95% indoor, ≥85% outdoor.
- Map accuracy ≥98% per-cell classification.
- Zero collisions during nominal ops.

**Tier 2 (relative metrics):**

- Scaling speedup ≥1.5× from 1→3 drones (small indoor), ≥2× from 1→5 drones (large outdoor).
- Frontier reassignment latency <2s after drone failure.

**Tier 3 (measured but not committed):**

- Wall-clock time per scenario, total path length per drone, map merge conflict rate, communication volume.

---

## Working with Claude Code

### How to be helpful

- **Read this file every session start.** The constraints above are load-bearing.
- **When in doubt, ask before adding scope.** Don't introduce a new dependency, a new module, a new interface, or a non-trivial new feature without checking. The MVP shape is intentional.
- **Prefer the simple thing.** If you're about to add threading, async, a queue, a cache, or a clever optimization — stop. The system is small. Sequential and obvious wins.
- **Tests first when implementing.** Write the test scaffolding (function signatures, expected behaviors) from the design before writing the implementation. Discuss the test cases with me before coding them.
- **Respect the dependency direction.** If you need to import something that violates it, that's a design signal, not a "just import it" situation.
- **Explain non-obvious decisions in code comments.** Future-me reads the code in 3 weeks; the "why" matters more than the "what."
- **Match existing patterns.** Once a pattern is established (e.g., how a module exposes its interface, how config is consumed, how logging is structured), follow it.

### Things to push back on

If I suggest any of these, push back:

- Adding a new module without first checking if existing modules could cover the responsibility.
- Implementing SLAM, real localization, 3D voxel mapping, peer-to-peer coordination, or RL-based anything (those are explicit future work).
- Introducing threading, async, multiprocessing, or queues into the tick loop.
- Mocking MuJoCo in unit tests instead of using inline MJCF or skipping the simulator.
- Bumping coverage by adding trivial tests that don't actually exercise behavior.
- Adding "just in case" extensibility points that don't have a concrete second use case in mind.

### Things to flag immediately

- Any change that introduces nondeterminism (unordered iteration, threading, time-based behavior, non-seeded RNG).
- Any change that crosses the dependency direction.
- Any change to one of the four named interfaces (`Localizer`, `Sensor`, `FrontierStrategy`, `Coordinator`) — these are stability points.
- Anything that meaningfully changes the public CLI, config schema, or output file format. Compatibility matters because acceptance tests rely on it.

---

## Repo layout (target)

```
.
├── pyproject.toml
├── uv.lock
├── README.md
├── CLAUDE.md                    # this file
├── docs/
│   ├── design.md                # full design document (updated each sprint)
│   └── diagrams/                # architecture + sequence (mermaid sources)
├── src/swarm_mapping/
│   ├── __init__.py
│   ├── config/                  # YAML loading + validation
│   ├── simulation/              # MuJoCo wrapper, Localizer, RayCaster
│   ├── perception/              # Sensor implementations
│   ├── mapping/                 # Mapper, grid, frontier detection
│   ├── planning/                # FrontierStrategy, PathPlanner
│   ├── coordination/            # Coordinator, tick loop, failure handling
│   ├── visualization/           # Renderer (matplotlib live + PNG export)
│   └── cli.py                   # entry point
├── scenarios/
│   ├── small_indoor/{scene.xml, config.yaml}
│   ├── large_indoor/{scene.xml, config.yaml}
│   ├── outdoor/{scene.xml, config.yaml}
│   └── failure_injection/{scene.xml, config.yaml}
├── tests/
│   ├── unit/                    # per-module
│   ├── integration/             # cross-module
│   └── acceptance/              # full scenario runs with tolerances
└── .github/workflows/           # CI
```

---

## Current sprint

Update this section each sprint so Claude Code knows what's in-flight.

**Sprint 1 — Skeleton (in progress).** Goal: 1 drone follows a hardcoded patrol path in the small indoor scenario, producing a `.npz` + `.png` map. Pipeline works end-to-end. Out of scope: `planning` (use hardcoded path), `visualization` (PNG export only), wind, failures, multi-drone.

Current focus: `simulation` module — MJCF drone model, `Localizer` and `RayCaster` interfaces, basic step loop.
