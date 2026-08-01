# Sprint 2 Plan — Multi-drone frontier exploration

Master plan for Sprint 2. Per-feature detailed plans live in
[`docs/sprint-2/`](sprint-2/) and are written (and reviewed) as each feature
starts. This file is the index and the source of truth for scope, decisions,
and progress.

## Goal

Multiple drones (1–5) with a real planner explore an unknown environment using
frontier-based exploration, producing the same `.npz` + `.png` 2.5D map as
Sprint 1 — but discovered autonomously rather than via a hardcoded patrol.

## Scope

**In:** `planning` module (frontier selection + A\* path planning),
`mapping.get_frontiers()`, multi-drone `coordination`, the large-indoor
scenario.

**Out (explicit):** wind, failure handling, outdoor scenario, live dashboard,
physics-based flight (movement stays teleport — see Decision 2).

## New interfaces (SOLID seams — stability points)

Two of the four named seams are created this sprint. Once defined, consumers
depend on the Protocol, not the implementation.

| Interface | Kind | First implementation |
| --- | --- | --- |
| `FrontierStrategy` | Protocol (seam) | `NearestFrontier` |
| `Coordinator` | Protocol (seam) | `CentralizedMaster` |
| `PathPlanner` | Protocol (non-seam) | `AStarPlanner` |

Dependency direction stays one-way: `coordination → planning → mapping (read-only)`.

## Feature breakdown

Each feature is a branch off `sprint-2`, merged back when green. The whole
sprint opens **one** PR `sprint-2 → main` at the end.

| # | Branch | Deliverable | Depends on | Status |
| --- | --- | --- | --- | --- |
| 1 | `feat/frontier-detection` | `mapping.get_frontiers()` — detect free-adjacent-to-unknown cells, cluster into regions, return world-coord centroids + size | mapping | ✅ merged (PR #7) |
| 2 | `feat/path-planner` | `planning`: `PathPlanner` protocol + `AStarPlanner` on the occupancy grid | mapping | ✅ implemented — PR pending — [plan](sprint-2/feature-2-path-planner.md) |
| 3 | `feat/frontier-strategy` | `planning`: `FrontierStrategy` protocol + `NearestFrontier` (+ spatial spreading penalty) | mapping (1) | ☐ |
| 4 | `feat/coordination-master` | `coordination`: `Coordinator` protocol + `CentralizedMaster` — multi-drone tick loop, frontier assignment, claimed list, step-along-path; refactor tick loop out of `cli.py` | 2, 3, simulation | ☐ |
| 5 | `feat/large-indoor` | `scenarios/large_indoor/` MJCF (50×50, corridors, doorways) + config, with the open-top/lighting/handle-camera treatment | — | ☐ |
| 6 | `feat/sprint2-integration` | Multi-drone config schema, CLI wiring, multi-drone `--view`, end-to-end + acceptance test + scaling-KPI check, update `docs/design.md` | 4, 5 | ☐ |

## Locked decisions

1. **`CURRENT_SPRINT = 2`** — bumped at sprint start, so Sprint-1 tests become
   the *regression* gate and new Sprint-2 tests are *progression*.
2. **Movement stays teleport.** Each tick a drone teleports one cell-step along
   its A\* path (no physics), preserving determinism. Physics-based flight is a
   candidate for a *later* sprint after the project is otherwise complete.
3. **Frontier detection lives in `mapping`** (`get_frontiers()` does detection +
   clustering); `planning` only *selects* among the returned regions. Keeps the
   dependency direction clean.
4. **`get_frontiers()` returns clustered region centroids** (usable targets +
   enables spreading), not raw cells. *The clustering algorithm itself is chosen
   in Feature 1's plan — see [`docs/sprint-2/feature-1-frontier-detection.md`](sprint-2/feature-1-frontier-detection.md).*
5. **Config schema changes** to support multiple drones (count + start
   positions) and planning params. New scenario configs follow the new schema.

## Cross-cutting constraints

- **Determinism (hard requirement).** Same config + seed = identical run.
  Process drones in sorted id order; sort frontier/candidate lists; break ties
  deterministically. No unordered dict/set iteration in decision paths.
- **Tests first.** For each feature, write the test signatures from the design
  and review the cases *before* implementing (CLAUDE.md rule). New tests are
  tagged `pytest.mark.sprint(2)` → they run as *progression*.
- **Required coverage** on `mapping`, `planning`, `coordination` (≥70%).

## KPIs / acceptance (verified in Feature 6)

- Coverage ≥95% indoor; map accuracy ≥98% per-cell; zero collisions.
- Scaling speedup ≥1.5× from 1→3 drones (small indoor).
- Frontier reassignment latency <2s after a drone finishes/invalidates a target.
- Large-indoor acceptance run compared to reference within tolerance.

## Workflow

```
main
 └── sprint-2                     (integration branch)
      ├── feat/frontier-detection → merge into sprint-2
      ├── feat/path-planner       → merge into sprint-2
      ├── feat/frontier-strategy  → merge into sprint-2
      ├── feat/coordination-master→ merge into sprint-2
      ├── feat/large-indoor       → merge into sprint-2
      └── feat/sprint2-integration→ merge into sprint-2
 sprint-2 → open ONE PR → main    (when the whole sprint is green)
```
