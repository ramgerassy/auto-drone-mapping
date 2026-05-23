"""End-to-end integration test for the mapping pipeline.

Uses the real MuJoCo scene — not a unit test. Verifies the full
pipeline: load scene, patrol, scan, map, export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swarm_mapping.cli import generate_patrol, run_pipeline

SCENARIO_CONFIG = (
    Path(__file__).resolve().parents[2] / "scenarios" / "small_indoor" / "config.yaml"
)


class TestEndToEnd:
    """Integration tests for the full pipeline."""

    def test_pipeline_produces_output(self, tmp_path: Path) -> None:
        """Full pipeline runs without error and produces output files."""
        run_pipeline(SCENARIO_CONFIG, tmp_path)

        assert (tmp_path / "map.npz").exists()
        assert (tmp_path / "map.png").exists()

    def test_npz_has_expected_content(self, tmp_path: Path) -> None:
        """NPZ output contains valid arrays with expected shapes."""
        run_pipeline(SCENARIO_CONFIG, tmp_path)

        data = np.load(str(tmp_path / "map.npz"))
        assert data["log_odds"].shape == (200, 200)
        assert data["probability"].shape == (200, 200)
        assert data["height"].shape == (200, 200)
        assert float(data["resolution"]) == 0.1

    def test_coverage_above_threshold(self, tmp_path: Path) -> None:
        """Pipeline achieves reasonable coverage (>80% for small indoor)."""
        run_pipeline(SCENARIO_CONFIG, tmp_path)

        data = np.load(str(tmp_path / "map.npz"))
        prob = data["probability"]
        mapped = np.sum((prob < 0.4) | (prob > 0.6))
        coverage = mapped / prob.size
        assert coverage > 0.80

    def test_png_is_valid_image(self, tmp_path: Path) -> None:
        """PNG output is a valid image with correct dimensions."""
        from PIL import Image

        run_pipeline(SCENARIO_CONFIG, tmp_path)

        img = Image.open(str(tmp_path / "map.png"))
        assert img.size == (200, 200)
        assert img.mode == "RGB"


class TestGeneratePatrol:
    """Tests for the patrol pattern generator."""

    def test_lawnmower_covers_area(self) -> None:
        """Patrol waypoints cover the specified area."""
        waypoints = generate_patrol(
            x_min=-4, x_max=4, y_min=-4, y_max=4, spacing=2.0, altitude=1.0
        )

        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]

        assert min(xs) == -4.0
        assert max(xs) == 4.0
        assert min(ys) == -4.0
        assert max(ys) == 4.0

    def test_all_waypoints_at_correct_altitude(self) -> None:
        """All waypoints have the specified z-coordinate."""
        waypoints = generate_patrol(
            x_min=0, x_max=4, y_min=0, y_max=4, spacing=2.0, altitude=1.5
        )

        for wp in waypoints:
            assert wp[2] == 1.5

    def test_waypoint_count(self) -> None:
        """Waypoint count matches expected grid size."""
        waypoints = generate_patrol(
            x_min=-8, x_max=8, y_min=-8, y_max=8, spacing=2.0, altitude=1.0
        )
        # 9 x-steps * 9 y-steps = 81
        assert len(waypoints) == 81
