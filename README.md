# auto-drone-mapping

Cooperative drone swarm for autonomous exploration and 2.5D mapping in MuJoCo.

## Design Decisions

### Ray Tracing Algorithm

The mapping module traces sensor rays through the occupancy grid to determine which cells are free (ray passed through) and which are occupied (ray hit an obstacle). We evaluated four algorithms:

| Algorithm | How it works | Pros | Cons |
|-----------|-------------|------|------|
| **Bresenham** (chosen) | Integer-only stepping along the line | Fast, deterministic, simple to implement and test | "Thin line" — may miss cells at shallow angles |
| DDA (Digital Differential Analyzer) | Float-based stepping with fixed increments | Slightly simpler code | Uses floating point — rounding can vary across platforms, less deterministic |
| Supercover / thick line | Visits ALL cells the mathematical line touches | Most complete coverage | Slower, more cells to process per ray |
| Amanatides & Woo | Steps through grid by tracking next axis crossing | Exact grid traversal — visits precisely the cells the ray intersects | More complex implementation |

**Why Bresenham:** Determinism is a hard requirement for this project (same config + same seed = identical run). Bresenham uses integer arithmetic only, guaranteeing identical results across platforms. At our grid scale (200×200 cells, 36 rays/scan), performance differences are negligible. The "thin line" approximation is acceptable at 10cm resolution — a ray that barely clips a cell corner does not meaningfully affect the occupancy map. If we find coverage artifacts later, we can upgrade to Amanatides & Woo without changing the Mapper's public API.
