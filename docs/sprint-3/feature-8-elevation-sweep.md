# Feature 8 — Elevation sweep: make the map actually 2.5D

Branch: `feat/elevation-sweep`
Decision log: [`docs/progress.md`](../progress.md), 2026-09-20 "The map is 2D, not 2.5D"

---

## The problem, measured

```
height layer: 3961 cells written, 58539 still -inf
  distinct height values: [1.]
```

1.0 m is the flight altitude. Every ray is built with body-frame `z = 0`, so
every `hit_point[2]` equals the drone's own altitude and `update_occupied`
stamps that constant into every cell it writes. CLAUDE.md commits to "2.5D
mapping only — 2D occupancy grid + per-cell height"; what ships is a 2D grid
plus a constant.

And anything below the flight plane is invisible. `large_indoor`'s four crates,
against a 1.0 m altitude:

```
crate_ne   0.00 .. 0.80   never hit
crate_sw   0.00 .. 0.80   never hit
crate_nw   0.00 .. 1.20   mapped
crate_se   0.00 .. 1.00   mapped
```

## Approach — fly low, look up

Two changes that solve different halves:

**Drop the flight altitude to 0.3 m.** A horizontal ray detects everything
taller than the altitude *at any range*, which an angled fan cannot: from 1.0 m
a -10° ray only reaches down to 0.47 m at 3 m range, and less further out.
Height-independent detection is the stronger guarantee and it is what a low
plane buys.

**Fan the rays upward.** Elevation layers from 0° up to `elevation_max_deg`
make `hit_point[2]` vary with what was struck, which is the only way the height
channel can carry information. From 0.3 m a +20° ray reaches 3.0 m at 7.4 m
horizontal — enough to measure this scene's walls.

**Upward, never downward.** A downward ray strikes the floor and the mapper
would record that as an obstacle; biasing the fan up means there is no ground
return to filter. Flying low is what makes an upward-only fan sufficient.

One pass, not the two-phase low-then-high sweep: re-flying the mission per
altitude band re-scans every wall each time.

## The rule that makes it safe

**Only navigation-plane rays write free space.**

Without this the feature erases the obstacles it exists to find. The mapper
projects every ray to 2D and marks all cells before the endpoint free. An
upward ray passing *over* a 0.8 m crate and striking a wall 10 m beyond would
mark the crate's own cell free — and with one occupied update (+0.847) against
four free ones (-1.62) per scan, the crate loses.

An elevated ray carries no information about what lies beneath it, so claiming
that space is free is wrong independently of this bug. `RayObservation` gains
`navigation_plane: bool`; the mapper applies free-space tracing only when it is
set, and occupancy/height updates from any ray that hit something.

## What this changes about the project

With the drone unable to climb over anything, **height stops being a navigation
input and becomes map output** — a property a consumer reads, not something the
planner consults. That is a narrowing of what "2.5D" means here, and worth
stating: this is 2.5D-for-mapping, not 2.5D-for-planning.

Every crate becomes an obstacle to route around. That is correct, and it means
the benchmark baselines recorded before this feature no longer describe the
same problem.

## Config

```yaml
drones:
  altitude: 0.3            # was 1.0
sensor:
  num_rays: 72             # azimuth samples, unchanged in meaning
  elevation_layers: 5      # 1 reproduces a flat sweep
  elevation_max_deg: 20.0  # layers span 0 deg .. this, upward only
```

Rays per scan become `num_rays x elevation_layers`. Ray-casting is already the
dominant per-tick cost, so this is the feature's real price.

## Test plan

**Geometry**

1. With one elevation layer the directions are exactly today's flat sweep.
2. With several, elevations span 0 to `elevation_max_deg` and none point down.
3. Every direction is a unit vector.

**The height channel — its first real test**

4. A tall obstacle and a short one produce **different** recorded heights.
   Every existing height assertion checks `-inf` or the flight altitude, so
   this channel has been green while measuring nothing.
5. A short crate below the old flight altitude is detected at all.

**The free-space rule**

6. An elevated ray that hits nothing writes nothing.
7. An elevated ray passing over an obstacle does **not** mark the cells beneath
   it free — verified by showing the obstacle survives a scan that includes
   such a ray, and is erased when the rule is removed.

**Regression**

8. Coverage on `small_indoor` still meets the KPI at the new altitude.
