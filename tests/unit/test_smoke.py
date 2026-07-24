"""Smoke test to verify package installation."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sprint(1)  # tests introduced in Sprint 1


@pytest.mark.sanity
def test_import() -> None:
    """Package can be imported and has a version string."""
    import swarm_mapping

    assert hasattr(swarm_mapping, "__version__")
    assert isinstance(swarm_mapping.__version__, str)
