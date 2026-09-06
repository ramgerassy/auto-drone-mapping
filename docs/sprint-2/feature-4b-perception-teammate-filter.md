# Feature 4b — Perception teammate filter (`perception.Rangefinder`)

Branch: `feat/perception-teammate-filter`
Depends on: Feature 4 (`SimulationEngine.drone_ids`)
Decision log: [`docs/progress.md`](../progress.md), 2026-08-15 entry

---

## Goal

Stop drones from mapping each other as obstacles. A ray that lands on a
teammate must contribute its free-space evidence and nothing else — no occupied
cell, no height write, and no false claim about the space behind the teammate.

## The bug, measured

Two drones 2.0 m apart, a ray from drone 0 along +x, in the real engine:

```
centre ray  -> drone_1_body @ 1.85 m     (= 2.00 - 0.15 half-extent)
ang +-3.44  -> drone_1_body @ 1.85 m
ang +-5.16  -> wall_east    @ 4.92 m
```

`mujoco.mj_ray`'s `bodyexclude` takes a **single** body id — the sensing drone's
own — and `raycaster.py` passes `geomgroup=None`, so every other drone is a
solid obstacle to every scan. Consequences are in `progress.md`; the one with no
recovery path is the height layer, because `update_occupied` does
`height = max(height, hit_z)` with no decay.

## Approach (settled)

Filter in `perception`, by comparing each hit point against teammates'
ground-truth positions. Rejected alternatives — a `geomgroup` mask at the
ray-cast layer, and filtering in `mapping` — are argued in `progress.md`.

**Geom-id filtering was considered and declined.** `RayHit` already carries
`geom_id`, so an exact identity match is available and would need no radius.
Declined because geom identity has no real-world analogue: a LiDAR return is not
tagged with what produced it. Filtering against known teammate positions is the
technique an actual swarm uses (shared telemetry), and keeping the simulator's
oracle out of the autonomy logic is worth a tuning parameter. The association
step here is idealized — ground-truth poses instead of telemetry — but it is the
same *shape* of computation.

## Public API

No change to the `Sensor` seam. `Rangefinder` gains one constructor argument:

```python
# src/swarm_mapping/perception/rangefinder.py
class Rangefinder:
    def __init__(
        self,
        engine: SimulationEngine,
        num_rays: int = 36,
        max_range: float = 10.0,
        angular_range: float = 2 * math.pi,
        exclusion_radius: float = DRONE_EXCLUSION_RADIUS,   # NEW
    ) -> None: ...
```

```python
# src/swarm_mapping/simulation/engine.py  (promoted from a literal)
DRONE_HALF_EXTENT = 0.15          # body geom half-extent, metres
DRONE_EXCLUSION_RADIUS = 0.30     # see "Choosing the radius"
```

`_build_drone_xml` reads `DRONE_HALF_EXTENT` instead of hardcoding `0.15`.
Feature 4c reuses both constants for planner clearance and the `min_separation`
floor — which is why 4b promotes them rather than 4c.

## Design

In `scan()`, after `cast_rays` returns and before an observation is built:

```
for each ray i with a hit within max_range:
    if any teammate t: |hit_point - t.position| < exclusion_radius:
        emit MISS with max_range = hit.distance
    else:
        emit HIT as today
```

Teammate poses are gathered once per scan from
`sorted(engine.drone_ids)` minus `drone_id`, via `engine.get_pose`. At <=5
drones and 72 rays that is at most 288 distance tests per scan — no vectorizing,
no caching.

**The encoding.** `RayObservation` already carries `max_range` beside
`distance`/`hit_point`, and `Mapper._integrate_observation` traces a MISS to
`origin + direction * max_range` marking every cell free (`mapper.py:73-75`).
So `distance=None, hit_point=None, max_range=<distance to the teammate>` gives
exactly the wanted semantics: free up to the teammate, nothing occupied, nothing
claimed beyond. **No change to `mapping`.**

Rejected encodings: dropping the ray discards legitimate free-space evidence up
to the teammate; a full-range MISS would falsely mark the occluded cells
*behind* the teammate as free.

`RayObservation.max_range`'s docstring becomes "distance to trace free space
along this ray" — the field is now the traced length, not the sensor's rating,
and the docstring should not claim otherwise.

**3D distance, not 2D.** Rays are horizontal at the sensing drone's altitude and
all drones fly the same `altitude`, so today the two agree exactly. 3D costs
nothing and stays correct if drones are ever separated vertically, where a 2D
test would discard a wall that merely shares an `(x, y)` with a teammate flying
above.

**Only hits within `max_range` are filtered.** A hit beyond `max_range` already
becomes a full-range MISS today; a teammate out of sensor range is the same
case, and re-deriving it would change existing behaviour for no gain.

**One hit per ray, and no re-cast.** `mj_ray` stops at the first geom, so a
filtered ray yields no second candidate — and deliberately so. Re-casting past
the teammate would map the wall behind it, which a real LiDAR cannot see. The
occlusion shadow stays unknown for this tick and fills in on a later pass.

## Choosing the radius

The test is radial against the teammate's **centre**, so the radius must cover
the farthest ray-visible point of the body.

- Hit points on the body box lie between **0.15** (face centre) and
  **0.212** (`0.15 * sqrt(2)`, corner) from the centre. A radius at or below
  0.212 lets corner-on hits through.
- Rotor geoms reach `0.15 * sqrt(2) + 0.08 = 0.292`. They carry
  `contype="0" conaffinity="0"`, but **`mj_ray` ignores contype** — it filters
  on `geomgroup` and `flg_static` only, so rotors are ray-visible in principle.
  They are missed today only because they sit 0.04-0.06 m *above* the sensor
  plane, a coincidence of every drone flying at one altitude. Not worth relying
  on.

Default **0.30 m**, which clears 0.292 with a small margin.

**Accepted cost:** a real wall within 0.30 m of a teammate's centre is
discarded too. That cell stays unknown for the tick and is mapped on a later
pass from a different vantage. Ground-truth poses are exact, so the radius
absorbs body extent only — it carries no localization error.

> **Corrected after the PR #11 review.** "Stays unknown" was only true once the
> free trace was changed to stop at the ray's entry into the exclusion sphere.
> As first written it stopped at the *hit*, and a MISS is traced free through
> its endpoint — so the discarded cell was claimed free rather than left
> unknown. See the 2026-09-06 addendum in `docs/progress.md`.

## Determinism

Teammates are iterated in sorted id order. The test is a boolean any-match, so
order cannot change the outcome; sorted is for legibility, matching
`movement.py`. No new float comparison enters a decision path that did not
already have one.

## Test plan (review these before I write them)

`tests/unit/test_perception/test_rangefinder.py`, tagged `sprint(2)`.
`FakeEngine` needs additive extension: a `drone_ids` property (default `[0]`)
and optional per-drone poses, both defaulting so Sprint-1 call sites are
unchanged.

**Filtering behaviour (unit, FakeEngine)**

1. A ray hitting a teammate yields `distance is None` and `hit_point is None`.
2. ...and `max_range` equals the hit distance, not the sensor rating.
3. A ray hitting a wall is untouched: `distance`, `hit_point` and `max_range`
   all as today.
4. A single-drone swarm filters nothing — no teammates, no behaviour change.
5. A corner-on hit at 0.212 m from a teammate's centre is still filtered
   (pins the radius above the box circumradius).
6. A hit beyond `max_range` stays a full-range MISS even if it is on a
   teammate.
7. A real wall within the exclusion radius of a teammate **is** discarded —
   pinning the accepted cost as known behaviour rather than a surprise.
8. Two identical scans produce identical observations.

**Pipeline behaviour (integration, real MuJoCo, two drones in line of sight)**

9. No occupied cell appears at the teammate's grid position.
10. The teammate's height cell stays `-inf` — the consequence with no recovery
    path.
11. Cells *between* the two drones are marked free.
12. Cells *beyond* the teammate stay unknown — the occlusion shadow is
    preserved, distinguishing this from a full-range MISS.
13. Without the filter the same scene produces an occupied cell at the
    teammate — the test that proves the others are not vacuous.

**Regression**

14. Existing Sprint-1 `test_rangefinder.py` tests pass unchanged.

## Files

| File | Change |
| --- | --- |
| `src/swarm_mapping/simulation/engine.py` | `DRONE_HALF_EXTENT`, `DRONE_EXCLUSION_RADIUS`; `_build_drone_xml` reads the constant |
| `src/swarm_mapping/perception/rangefinder.py` | `exclusion_radius` arg; teammate filter in `scan()` |
| `src/swarm_mapping/perception/types.py` | `max_range` docstring |
| `tests/conftest.py` | `FakeEngine.drone_ids` + per-drone poses (additive) |
| `tests/unit/test_perception/test_rangefinder.py` | tests 1-8 |
| `tests/integration/test_e2e.py` (or a new module) | tests 9-13 |

## Done when

- All 14 tests green, full suite green, ruff + mypy clean.
- `coordination`/`planning` coverage unchanged; `perception` covered via the
  new unit tests.
- `test_master.py`'s docstring caveat about map accuracy dropped, and a
  map-accuracy assertion added there — on the **height** layer, since a
  mission-level occupancy assertion proved vacuous (see `progress.md`).

## Not in this feature

Planner clearance and the `min_separation` guards (Feature 4c), even though 4b
introduces the constants they will use.
