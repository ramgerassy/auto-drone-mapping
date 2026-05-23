"""Minimal YAML config loader for scenario files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return scenario configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
