"""YAML scenario loading.

Thin by design: read the file, hand the document to `schema.parse_config`, and
let validation raise. CLAUDE.md commits to "validated at startup, fail-fast on
errors", so nothing downstream ever sees a partially-checked config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from swarm_mapping.config.schema import ScenarioConfig, parse_config


def load_config(path: str | Path) -> ScenarioConfig:
    """Load, validate and return a scenario configuration.

    Args:
        path: Path to the YAML config file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or fails validation. The message
            names the offending key — see `schema.parse_config`.
    """
    config_path = Path(path)
    with config_path.open() as f:
        raw: Any = yaml.safe_load(f)
    try:
        return parse_config(raw)
    except ValueError as exc:
        # Which file failed matters as much as which key: the acceptance suite
        # loads several scenarios, and a bare key name does not say which one.
        msg = f"{config_path}: {exc}"
        raise ValueError(msg) from exc
