# auto-drone-mapping

Cooperative drone swarm for autonomous exploration and 2.5D mapping in MuJoCo.

A swarm of 1–5 drones explores an unknown indoor environment and builds a shared
2.5D map: a 2D occupancy grid plus a per-cell height, saved as `.npz` (data) and
`.png` (to look at). No route is scripted. One central master assigns each drone
a frontier, the edge of what is known, and plans a collision-free path to it
with A\*. Drones yield to each other and the swarm stops when nothing reachable
is left. A drone can also be scripted to fail mid-mission, either by going
silent or by jamming its motors. The master detects the failure from symptoms
alone, hands the failed drone's frontier to a teammate, routes around the wreck
and still finishes the map. Poses come from the simulator's ground truth, so
there is no SLAM.

The design and its reasoning are in [`docs/design.md`](docs/design.md). The
decision log is in [`docs/progress.md`](docs/progress.md).

## Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # the simulator, CLI and dev tools
uv sync --extra ui      # also the operator console (Streamlit), optional
```

## Run a scenario from the CLI

```bash
uv run swarm-mapping --config scenarios/small_indoor/config.yaml --output output/
```

| Scenario | What it is | Start positions |
| --- | --- | --- |
| `small_indoor` | 20 m × 20 m single room. The baseline. | 3 |
| `large_indoor` | 50 m × 50 m corridor cross with 8 rooms. The coordination scene. | 5 |
| `loop_indoor` | 50 m × 50 m ring corridor with 12 rooms. The control for `large_indoor`: the same extent, but there are always two routes between any two places. | 5 |
| `comb_indoor` | 40 m × 40 m spine with nine dead-end ribs. An adversarial allocation benchmark. | 3 |
| `failure_injection` | `large_indoor` where drone 1 goes **silent** at tick 300. | 5 |
| `failure_stuck` | `large_indoor` where drone 1's motors **jam** at tick 300. It keeps reporting but stops moving. | 5 |

Each scenario is `scenarios/<name>/config.yaml`. The config names its MJCF scene
and sets every parameter. Nothing is hardcoded.

Key flags (`uv run swarm-mapping --help` lists all of them):

| Flag | Effect |
| --- | --- |
| `--config PATH` | Scenario YAML. Required. |
| `--output DIR` | Where the run's files go. Default `output/`. |
| `--drones N` | Fly the first N start positions. It never invents new ones. |
| `--assignment {greedy,global}` | Override how frontiers are handed out. |
| `--target-tolerance CELLS` | Override how far a frontier may drift before a drone's target counts as gone. |
| `--view` | Open a live MuJoCo 3D viewer. View-only: the map is identical without it. Needs a display. |
| `--visit-heatmaps` | Also write one visit-count PNG per drone. Diagnostic. |
| `--verbose` | Log INFO to stderr. |

`--assignment` and `--target-tolerance` together select the allocation
variants that the console and benchmarks compare: **baseline** (`greedy`, 0),
**A** (`global`, 0), **B** (`greedy`, 3), **A+B** (`global`, 3).

The process exits non-zero if the mission hit its tick cap without finishing.
A run that ends "blocked" (a few frontiers it could see but not reach) still
exits 0. See [design.md §8.1](docs/design.md) for why.

### What a run writes

Everything goes into `--output DIR`:

| File | Contents |
| --- | --- |
| `map.npz` | The map data: log-odds, probability, height, resolution, origin. |
| `map.png` | The map image, with obstacles shaded by height. |
| `log.jsonl` | Every log event, one JSON object per line, each with `event` and `tick`. Runs with identical inputs produce identical logs. |
| `run.json` | The inputs (a config snapshot, drones, variant) and the outcome (ticks, coverage, blocked, succeeded, per-drone path stats, event counts). |
| `paths.json`, `route_drone_<id>.png` | Each drone's path. |
| `visits_drone_<id>.png` | Only with `--visit-heatmaps`. |

## The operator console

```bash
uv sync --extra ui
uv run swarm-console        # opens in the browser
```

Streamlit is an optional extra, so the core install, CI and Docker image never
include it. Every console run launches the CLI above as a separate process, so
it is identical to typing the same command. Runs land in
`runs/<time>_<scenario>_<variant>_<N>d/`, which git ignores.

- **Run**: pick a scenario, a drone count (up to the number of start
  positions it declares), an allocation variant, and headless or MuJoCo view. A live tick and log panel shows while the run is in flight.
- **Scenarios**: list scenarios with their validity, and upload a new room (a
  config plus its MJCF scene). It is validated, including all five spawns, and
  installed without overwriting anything.
- **History**: every past run with its map, each drone's route, the log with
  an event filter, its metrics, and a side-by-side comparison of two runs.

## Tests

```bash
uv run pytest -m "not acceptance"                    # everything except full-scenario runs
uv run pytest -m "regression and not acceptance"     # prior sprints: must not break
uv run pytest -m "progression and not acceptance"    # current sprint: work in flight
uv run pytest -m acceptance                          # full-scenario runs; minutes, not seconds
```

Every test module carries a `sprint(N)` marker. `tests/conftest.py` labels it
regression or progression against `CURRENT_SPRINT`. CI runs both labels on
every push. It adds coverage and the acceptance suite on pull requests to
`main`. The console's Streamlit UI tests run only when the `ui` extra is
installed, so they run locally and skip in CI.

## Benchmarks

These are not tests. They run full missions and print a table. Run them by hand.

```bash
uv run python benchmarks/strategy_matrix.py [scenario]   # compare the allocation variants (baseline, A, B, A+B) across scenarios
uv run python benchmarks/oracle_ceiling.py --mode=greedy large_indoor   # upper bound for a perfect information-gain frontier strategy
uv run python benchmarks/failure_recovery.py   # failure-detection latency KPI and recovery cost, silent and stuck
```


## Design Decisions

### Pose Representation: Quaternions vs Rotation Matrices

MuJoCo stores body orientations as **unit quaternions** `(w, x, y, z)` rather than 3x3 rotation matrices or Euler angles. Our `Pose` dataclass follows this convention, storing position and quaternion separately — equivalent to a 4x4 homogeneous transform `T = [R | p; 0 0 0 1]` but more compact.

| Representation | Storage | Compose cost | Re-normalization | Interpolation | Gimbal lock |
|---------------|---------|-------------|-----------------|--------------|-------------|
| Rotation matrix (SO(3)) | 9 floats (3 DOF) | 27 multiplies | Gram-Schmidt (expensive) | Not natural | No |
| Euler angles (roll, pitch, yaw) | 3 floats (3 DOF) | Via matrix conversion | Trivial | Not natural | Yes |
| **Quaternion** | **4 floats (3 DOF)** | **16 multiplies** | **Divide by norm (trivial)** | **SLERP (natural)** | **No** |

**Why quaternions are preferred for this project:**

1. **Compact state vector** — MuJoCo's `qpos` uses 7 values per freejoint body (3 position + 4 quaternion) instead of 12 (3 position + 9 rotation matrix). With 1-5 drones, this keeps the state vector small.

2. **Numerical stability** — after thousands of simulation steps, floating-point drift can cause a rotation matrix to lose orthogonality (`R^T R != I`). Re-orthogonalizing requires Gram-Schmidt, which is expensive. Quaternions just need `q /= ||q||` — a single division.

3. **No gimbal lock** — Euler angles suffer from gimbal lock when pitch approaches +/-90 degrees, losing a degree of freedom. Quaternions represent all orientations uniformly.

4. **Efficient composition** — combining two rotations is 16 multiplications (quaternion multiply) vs 27 (matrix multiply).

5. **Smooth interpolation** — SLERP (Spherical Linear Interpolation) between two quaternions produces a constant-speed rotation along the shortest arc. Interpolating rotation matrices or Euler angles does not have this property.

**Conversion:** our `_rotate_vectors_by_quaternion` helper in the perception module converts the quaternion to a rotation matrix for the actual vector rotation, since rotating N vectors via a matrix (`R @ v`) is more efficient than N individual quaternion multiplies (`q * v * q_conj`). The quaternion-to-matrix formula is:

```
R = [ 1-2(y^2+z^2)    2(xy-wz)      2(xz+wy)   ]
    [ 2(xy+wz)         1-2(x^2+z^2)  2(yz-wx)   ]
    [ 2(xz-wy)         2(yz+wx)      1-2(x^2+y^2)]
```

Where `(w, x, y, z)` is the unit quaternion following MuJoCo's w-first convention.

### Ray Tracing Algorithm

The mapping module traces sensor rays through the occupancy grid to determine which cells are free (ray passed through) and which are occupied (ray hit an obstacle). We evaluated four algorithms:

| Algorithm | How it works | Pros | Cons |
|-----------|-------------|------|------|
| **Bresenham** (chosen) | Integer-only stepping along the line | Fast, deterministic, simple to implement and test | "Thin line" — may miss cells at shallow angles |
| DDA (Digital Differential Analyzer) | Float-based stepping with fixed increments | Slightly simpler code | Uses floating point — rounding can vary across platforms, less deterministic |
| Supercover / thick line | Visits ALL cells the mathematical line touches | Most complete coverage | Slower, more cells to process per ray |
| Amanatides & Woo | Steps through grid by tracking next axis crossing | Exact grid traversal — visits precisely the cells the ray intersects | More complex implementation |

**Why Bresenham:** Determinism is a hard requirement for this project (same config + same seed = identical run). Bresenham uses integer arithmetic only, guaranteeing identical results across platforms. At our grid scale (200x200 cells, 36 rays/scan), performance differences are negligible. The "thin line" approximation is acceptable at 10cm resolution — a ray that barely clips a cell corner does not meaningfully affect the occupancy map. If we find coverage artifacts later, we can upgrade to Amanatides & Woo without changing the Mapper's public API.
