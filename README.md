# auto-drone-mapping

Cooperative drone swarm for autonomous exploration and 2.5D mapping in MuJoCo.

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
uv sync
```

### Run the Demo

```bash
uv run swarm-mapping --config scenarios/small_indoor/config.yaml --output output/ --verbose
```

This runs a single drone through a lawnmower patrol pattern in a 20m x 20m indoor room, scanning the environment with a rangefinder sensor and building a 2.5D occupancy map.

**Output:**
- `output/map.png` — height-colored occupancy image (white = free, blue = unknown, grayscale = obstacles shaded by height)
- `output/map.npz` — raw data arrays (log-odds, probability, height, resolution, origin)

**Typical results:** 81 waypoints, 98% coverage, ~1 second runtime.

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
