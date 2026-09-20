"""Typed, validated scenario configuration.

`config` depends on nothing — not on `simulation`, not on `mapping` — so the
checks here are exactly those visible from inside a YAML document: required
keys, types, numeric ranges, and the swarm size CLAUDE.md commits to.

**Physical floors deliberately live elsewhere.** The body-diagonal floor on
`min_separation` is enforced in `CentralizedMaster.__init__` and the corner-radius
floor on `exclusion_radius` in `Rangefinder.__init__`, because both are stated in
terms of `DRONE_HALF_EXTENT` — a `simulation` constant. Re-deriving them here
would mean `config -> simulation`, which the dependency direction forbids. The
split is not an oversight: this module rejects values that are *meaningless*
(a negative radius), and the modules that can see the geometry reject values
that are *unsafe* (a radius smaller than the body).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# CLAUDE.md scopes the system at 1-5 drones: "Don't optimize for 50 or 500."
MIN_DRONES = 1
MAX_DRONES = 5

# A start position is (x, y, z) in world coordinates.
POSITION_LENGTH = 3


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required top-level section, or raise naming it."""
    if name not in config:
        msg = f"config is missing the required '{name}' section"
        raise ValueError(msg)
    value = config[name]
    if not isinstance(value, dict):
        msg = f"config section '{name}' must be a mapping, got {type(value).__name__}"
        raise ValueError(msg)
    return value


def _field(section: dict[str, Any], section_name: str, field: str) -> Any:
    """Return a required key from a section, or raise naming `section.field`."""
    if field not in section:
        msg = f"config is missing the required key '{section_name}.{field}'"
        raise ValueError(msg)
    return section[field]


def _number(section: dict[str, Any], section_name: str, field: str) -> float:
    """Return a required numeric field as a float.

    `bool` is rejected explicitly: it subclasses `int` in Python, so a plain
    isinstance check would accept `num_rays: true` and run the mission with a
    single ray. A typo that silently changes the sensor is worse than a crash.
    """
    value = _field(section, section_name, field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = (
            f"'{section_name}.{field}' must be a number, got "
            f"{value!r} ({type(value).__name__})"
        )
        raise ValueError(msg)
    return float(value)


def _integer(section: dict[str, Any], section_name: str, field: str) -> int:
    """Return a required integer field. Rejects `bool` — see `_number`."""
    value = _field(section, section_name, field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = (
            f"'{section_name}.{field}' must be an integer, got "
            f"{value!r} ({type(value).__name__})"
        )
        raise ValueError(msg)
    return value


def _positive(section: dict[str, Any], section_name: str, field: str) -> float:
    """Return a required numeric field that must be strictly greater than 0."""
    value = _number(section, section_name, field)
    if value <= 0.0:
        msg = f"'{section_name}.{field}' must be greater than 0, got {value}"
        raise ValueError(msg)
    return value


def _non_negative(section: dict[str, Any], section_name: str, field: str) -> float:
    """Return a required numeric field that must be 0 or greater.

    Zero is meaningful for every radius in the schema — it disables the filter
    or inflation deliberately — so the bound is `>= 0`, not `> 0`.
    """
    value = _number(section, section_name, field)
    if value < 0.0:
        msg = f"'{section_name}.{field}' must be 0 or greater, got {value}"
        raise ValueError(msg)
    return value


def _positive_int(section: dict[str, Any], section_name: str, field: str) -> int:
    """Return a required integer field that must be strictly greater than 0."""
    value = _integer(section, section_name, field)
    if value <= 0:
        msg = f"'{section_name}.{field}' must be greater than 0, got {value}"
        raise ValueError(msg)
    return value


def _non_negative_int(section: dict[str, Any], section_name: str, field: str) -> int:
    """Return a required integer field that must be 0 or greater."""
    value = _integer(section, section_name, field)
    if value < 0:
        msg = f"'{section_name}.{field}' must be 0 or greater, got {value}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class DroneSettings:
    """The swarm's starting layout.

    Attributes:
        start_positions: One (x, y, z) world position per drone, in order.
            Drone ids are the indices of this sequence.
        altitude: Flight height in metres. Drones stay in one horizontal plane.
    """

    start_positions: tuple[tuple[float, float, float], ...]
    altitude: float

    @property
    def count(self) -> int:
        """Number of drones in the swarm.

        Derived from `start_positions` rather than stored separately, so a
        count and a position list can never disagree.
        """
        return len(self.start_positions)


@dataclass(frozen=True)
class SensorSettings:
    """Rangefinder geometry.

    Attributes:
        num_rays: Angular samples per scan for the single modeled sensor.
        max_range: Maximum sensor range in metres.
        exclusion_radius: Radius around a teammate's centre within which a hit
            is attributed to that teammate rather than the environment.
    """

    num_rays: int
    max_range: float
    exclusion_radius: float


@dataclass(frozen=True)
class MapSettings:
    """Occupancy grid extent and export shading.

    Attributes:
        resolution: Metres per grid cell.
        origin_x: World x-coordinate of cell (0, 0).
        origin_y: World y-coordinate of cell (0, 0).
        grid_width: Number of cells in x.
        grid_height: Number of cells in y.
        max_height: Ceiling height used for PNG shading.
    """

    resolution: float
    origin_x: float
    origin_y: float
    grid_width: int
    grid_height: int
    max_height: float

    @property
    def max_x(self) -> float:
        """World x-coordinate just past the last column (exclusive bound)."""
        return self.origin_x + self.grid_width * self.resolution

    @property
    def max_y(self) -> float:
        """World y-coordinate just past the last row (exclusive bound)."""
        return self.origin_y + self.grid_height * self.resolution

    def contains(self, x: float, y: float) -> bool:
        """Whether a world point falls inside the grid.

        The upper bounds are exclusive because `world_to_grid` floors: a point
        exactly at `max_x` maps to column `grid_width`, one past the last
        valid column.
        """
        return self.origin_x <= x < self.max_x and self.origin_y <= y < self.max_y


@dataclass(frozen=True)
class PlanningSettings:
    """Path planning and frontier selection parameters.

    Attributes:
        strategy: Which `FrontierStrategy` to use — "nearest" (cheapest
            reachable frontier by true path cost) or "information_gain" (best
            unknown-space-seen per unit of travel, targeting a viewpoint rather
            than the frontier cell). Both are kept so the two can be compared
            on the same scenario; see docs/sprint-3/feature-7-*.md.
        clearance_radius: Drone half-extent plus safety margin, in metres.
            Obstacles are inflated by this before planning.
        spread_radius: Radius in metres within which a candidate frontier is
            penalized for sitting near one already claimed.
        spread_penalty: Cost added to a candidate inside `spread_radius`.
        min_frontier_size: Frontier regions with fewer cells are discarded as
            noise. Wall-surface cells collect both free and occupied evidence
            at grazing incidence and drift across the 0.4/0.6 classification
            bands, so even a finished map emits a churn of 2-3 cell regions no
            drone can clear. See `Mapper`.
    """

    strategy: str
    clearance_radius: float
    spread_radius: float
    spread_penalty: int
    min_frontier_size: int


@dataclass(frozen=True)
class CoordinationSettings:
    """Mission orchestration parameters.

    Attributes:
        min_separation: Required centre-to-centre spacing between drones, in
            metres.
        max_wait_ticks: Consecutive blocked ticks after which a drone abandons
            its frontier, breaking head-on deadlocks.
        max_ticks: Safety cap on mission length. A blocked mission must end.
        return_to_base_ticks: Consecutive ticks a drone may sit unassigned
            before it flies back to its start position. Expressed in ticks
            because ticks are the only clock a deterministic run has — no
            `mj_step` is called, so simulated seconds never advance, and
            wall-clock time would make the mission non-reproducible. 0 keeps
            idle drones parked where they stopped.
        no_progress_ticks: Consecutive ticks without a newly classified cell
            after which the mission stops. Wall-surface cells drift across the
            classification bands and keep emitting small frontier regions, a
            few of them transiently reachable, so a swarm with nothing left to
            find can hold assignments indefinitely and never satisfy "every
            drone unassigned". Measured on large_indoor: coverage is flat from
            tick 2000 while the mission runs to its cap. 0 disables the check.
    """

    min_separation: float
    max_wait_ticks: int
    max_ticks: int
    no_progress_ticks: int
    return_to_base_ticks: int


@dataclass(frozen=True)
class ScenarioConfig:
    """A fully validated scenario.

    Attributes:
        scene_path: MJCF filename, relative to the simulation assets directory.
        drones: Swarm starting layout.
        sensor: Rangefinder geometry.
        map: Occupancy grid extent and shading (the YAML `map:` section).
        planning: Planner and frontier-selection parameters.
        coordination: Mission orchestration parameters.
    """

    scene_path: str
    drones: DroneSettings
    sensor: SensorSettings
    map: MapSettings
    planning: PlanningSettings
    coordination: CoordinationSettings


def _parse_start_positions(
    section: dict[str, Any],
) -> tuple[tuple[float, float, float], ...]:
    """Parse and validate the swarm's start positions."""
    raw = _field(section, "drones", "start_positions")
    if not isinstance(raw, list):
        msg = (
            f"'drones.start_positions' must be a list of [x, y, z] positions, "
            f"got {type(raw).__name__}"
        )
        raise ValueError(msg)

    if not MIN_DRONES <= len(raw) <= MAX_DRONES:
        msg = (
            f"'drones.start_positions' must hold between {MIN_DRONES} and "
            f"{MAX_DRONES} positions (CLAUDE.md scopes the system at 1-5 "
            f"drones), got {len(raw)}"
        )
        raise ValueError(msg)

    positions: list[tuple[float, float, float]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, list | tuple) or len(entry) != POSITION_LENGTH:
            msg = (
                f"'drones.start_positions[{index}]' must have three "
                f"coordinates [x, y, z], got {entry!r}"
            )
            raise ValueError(msg)
        coordinates: list[float] = []
        for axis, value in zip("xyz", entry, strict=True):
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = (
                    f"'drones.start_positions[{index}]' coordinate {axis} must "
                    f"be a number, got {value!r}"
                )
                raise ValueError(msg)
            coordinates.append(float(value))
        positions.append((coordinates[0], coordinates[1], coordinates[2]))

    return tuple(positions)


def _check_positions_fit_grid(
    positions: tuple[tuple[float, float, float], ...],
    map_settings: MapSettings,
) -> None:
    """Reject any start position that falls outside the grid.

    The originally-planned check was "the grid covers the scene", but the
    scene's extent lives in the MJCF and reading it would mean
    `config -> simulation`. Containment of the start positions is the part of
    that check visible from inside `config` — and it catches the same class of
    mistake, a grid that does not fit the mission it is configured for.
    """
    for index, (x, y, _z) in enumerate(positions):
        if not map_settings.contains(x, y):
            msg = (
                f"'drones.start_positions[{index}]' at ({x}, {y}) is outside "
                f"the grid, which spans x [{map_settings.origin_x}, "
                f"{map_settings.max_x}) and y [{map_settings.origin_y}, "
                f"{map_settings.max_y}). A drone starting off-grid can never "
                f"be mapped or planned for."
            )
            raise ValueError(msg)


def parse_config(raw: Any) -> ScenarioConfig:
    """Validate a parsed YAML document and return a typed scenario config.

    Args:
        raw: The document as returned by `yaml.safe_load`.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If the document is empty, is not a mapping, is missing a
            required key, or holds a value of the wrong type or outside its
            permitted range. The message names the offending key.
    """
    if raw is None:
        msg = "config file is empty"
        raise ValueError(msg)
    if not isinstance(raw, dict):
        msg = f"config must be a mapping at the top level, got {type(raw).__name__}"
        raise ValueError(msg)

    scene = _section(raw, "scene")
    scene_path = _field(scene, "scene", "path")
    if not isinstance(scene_path, str):
        msg = f"'scene.path' must be a string, got {scene_path!r}"
        raise ValueError(msg)

    drones_section = _section(raw, "drones")
    sensor_section = _section(raw, "sensor")
    map_section = _section(raw, "map")
    planning_section = _section(raw, "planning")
    coordination_section = _section(raw, "coordination")

    map_settings = MapSettings(
        resolution=_positive(map_section, "map", "resolution"),
        origin_x=_number(map_section, "map", "origin_x"),
        origin_y=_number(map_section, "map", "origin_y"),
        grid_width=_positive_int(map_section, "map", "grid_width"),
        grid_height=_positive_int(map_section, "map", "grid_height"),
        max_height=_positive(map_section, "map", "max_height"),
    )

    start_positions = _parse_start_positions(drones_section)
    _check_positions_fit_grid(start_positions, map_settings)

    return ScenarioConfig(
        scene_path=scene_path,
        drones=DroneSettings(
            start_positions=start_positions,
            altitude=_number(drones_section, "drones", "altitude"),
        ),
        sensor=SensorSettings(
            num_rays=_positive_int(sensor_section, "sensor", "num_rays"),
            max_range=_positive(sensor_section, "sensor", "max_range"),
            exclusion_radius=_non_negative(
                sensor_section, "sensor", "exclusion_radius"
            ),
        ),
        map=map_settings,
        planning=PlanningSettings(
            strategy=_strategy(planning_section),
            clearance_radius=_non_negative(
                planning_section, "planning", "clearance_radius"
            ),
            spread_radius=_non_negative(planning_section, "planning", "spread_radius"),
            spread_penalty=_non_negative_int(
                planning_section, "planning", "spread_penalty"
            ),
            min_frontier_size=_non_negative_int(
                planning_section, "planning", "min_frontier_size"
            ),
        ),
        coordination=CoordinationSettings(
            min_separation=_non_negative(
                coordination_section, "coordination", "min_separation"
            ),
            max_wait_ticks=_non_negative_int(
                coordination_section, "coordination", "max_wait_ticks"
            ),
            max_ticks=_positive_int(coordination_section, "coordination", "max_ticks"),
            no_progress_ticks=_non_negative_int(
                coordination_section, "coordination", "no_progress_ticks"
            ),
            return_to_base_ticks=_non_negative_int(
                coordination_section, "coordination", "return_to_base_ticks"
            ),
        ),
    )


_STRATEGIES = ("nearest", "information_gain")


def _strategy(section: dict[str, Any]) -> str:
    """Read and validate `planning.strategy`.

    Args:
        section: The `planning` config section.

    Returns:
        The strategy name.

    Raises:
        ValueError: If the key is missing or names an unknown strategy. Listed
            explicitly rather than falling back to a default, because silently
            running the wrong exploration strategy is invisible in the output —
            the map looks plausible either way, just worse.
    """
    value = section.get("strategy")
    if value not in _STRATEGIES:
        msg = f"planning.strategy must be one of {_STRATEGIES}, got {value!r}"
        raise ValueError(msg)
    return str(value)
