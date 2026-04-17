import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.behavior_metrics import compute_behavior_metrics
from analysis.daily_activity_scan import create_daily_scan_manifest
from analysis.ephys_metrics import (
    binary_entropy,
    center_of_activity,
    cluster_binary_entropy,
    cross_correlation,
    exclusive_motor_event_percentage,
    functional_plasticity_distance,
)
from analysis.extract_spikes import detect_threshold_crossings
from analysis.validate_session import validate_session_dir


class BehaviorMetricsTest(unittest.TestCase):
    def test_behavior_metrics_use_events_and_window_samples_after_exclusion(self):
        windows = [
            {"phase": "game", "elapsed_ms": 0, "paddle_y": 10},
            {"phase": "game", "elapsed_ms": 10000, "paddle_y": 15},
            {"phase": "game", "elapsed_ms": 10010, "paddle_y": 20},
            {"phase": "game", "elapsed_ms": 10020, "paddle_y": 12},
        ]
        events = [
            {"event": "miss", "phase": "game", "detail": "bounces=0,rally_id=0"},
            {"event": "miss", "phase": "game", "detail": "bounces=4,rally_id=1"},
            {"event": "miss", "phase": "pre_rest", "detail": "bounces=99,rally_id=2"},
        ]

        metrics = compute_behavior_metrics(windows, events, exclude_initial_game_seconds=10)

        self.assertEqual(metrics["rally_lengths"], [0, 4])
        self.assertEqual(metrics["average_rally_length"], 2.0)
        self.assertEqual(metrics["aces"], 1)
        self.assertEqual(metrics["long_rallies"], 1)
        self.assertEqual(metrics["paddle_movement"], 13)

    def test_behavior_metrics_reconstruct_rally_lengths_from_hits_when_miss_bounces_are_zero(self):
        windows = [
            {"phase": "game", "elapsed_ms": 10000, "paddle_y": 10},
            {"phase": "game", "elapsed_ms": 10010, "paddle_y": 12},
        ]
        events = [
            {"event": "miss", "phase": "game", "detail": "bounces=0,rally_id=0"},
            {"event": "hit", "phase": "game", "detail": "bounces=1"},
            {"event": "hit", "phase": "game", "detail": "bounces=2"},
            {"event": "miss", "phase": "game", "detail": "bounces=0,rally_id=1"},
        ]

        metrics = compute_behavior_metrics(windows, events, exclude_initial_game_seconds=10)

        self.assertEqual(metrics["rally_lengths"], [0, 2])
        self.assertEqual(metrics["average_rally_length"], 1.0)
        self.assertEqual(metrics["aces"], 1)
        self.assertEqual(metrics["long_rallies"], 0)

    def test_behavior_metrics_handles_per_window_elapsed_ms_logs(self):
        windows = []
        for index in range(1200):
            paddle_y = 10 if index < 1000 else 10 + (index - 999)
            windows.append(
                {
                    "phase": "game",
                    "elapsed_ms": 10,
                    "frame_start": index * 200,
                    "frame_end": (index + 1) * 200,
                    "paddle_y": paddle_y,
                }
            )

        metrics = compute_behavior_metrics(windows, [], exclude_initial_game_seconds=10)

        self.assertGreater(metrics["paddle_movement"], 0)


class EphysMetricsTest(unittest.TestCase):
    def test_entropy_ca_and_exclusive_motor_metrics(self):
        self.assertEqual(binary_entropy(0.0), 0.0)
        self.assertEqual(binary_entropy(1.0), 0.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)

        ca = center_of_activity(
            [
                {"spikes": 1, "x": 0, "y": 0},
                {"spikes": 3, "x": 4, "y": 8},
            ]
        )
        self.assertEqual(ca, (3.0, 6.0))

        pct = exclusive_motor_event_percentage([
            {"motor_1_spikes": 2, "motor_2_spikes": 0},
            {"motor_1_spikes": 0, "motor_2_spikes": 3},
            {"motor_1_spikes": 1, "motor_2_spikes": 1},
            {"motor_1_spikes": 0, "motor_2_spikes": 0},
        ])
        self.assertEqual(pct, 50.0)

    def test_cross_correlation_entropy_and_plasticity_helpers(self):
        self.assertAlmostEqual(cross_correlation([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(cross_correlation([1, 2, 3], [3, 2, 1]), -1.0)

        self.assertAlmostEqual(cluster_binary_entropy([0, 1, 1, 0]), 1.0)
        self.assertEqual(cluster_binary_entropy([1, 1, 1, 1]), 0.0)

        self.assertEqual(functional_plasticity_distance((3.0, 4.0), (0.0, 0.0)), 5.0)


class DailyActivityScanTest(unittest.TestCase):
    def test_daily_scan_manifest_records_checkerboard_requirements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = create_daily_scan_manifest(
                root=tmpdir,
                culture_id="culture_a",
                div=14,
                scan_paths=[f"checker_{i}.h5" for i in range(14)],
                timestamp="20260416",
            )
            manifest = json.loads(Path(manifest_path).read_text())

        self.assertEqual(manifest["culture_id"], "culture_a")
        self.assertEqual(manifest["div"], 14)
        self.assertEqual(manifest["checkerboard_configurations"], 14)
        self.assertEqual(manifest["record_seconds_per_configuration"], 15)
        self.assertEqual(manifest["highpass_hz"], 300)
        self.assertEqual(manifest["gain"], 512)
        self.assertEqual(manifest["threshold_sigma"], 6)
        self.assertEqual(len(manifest["scan_paths"]), 14)


class ExtractionAndValidationTest(unittest.TestCase):
    def test_extract_spikes_detects_negative_threshold_crossings(self):
        traces = {
            3: [0.0, -1.0, -6.0, -2.0, -7.0],
            4: [0.0, 0.0, -2.0, -3.0, -4.0],
        }

        spikes = detect_threshold_crossings(traces, threshold=-5.0, refractory_samples=2)

        self.assertEqual(spikes, [{"channel": 3, "sample": 2, "amplitude": -6.0}])

    def test_validate_session_dir_requires_raw_layout_and_runtime_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            for name in (
                "session_manifest.json",
                "session_config.json",
                "resolved_layout.json",
                "runtime_events.jsonl",
                "window_samples.csv",
                "quality_summary.json",
                "session.raw.h5",
            ):
                (session_dir / name).write_text("{}")

            report = validate_session_dir(session_dir)

        self.assertTrue(report["valid"])
        self.assertEqual(report["missing"], [])


if __name__ == "__main__":
    unittest.main()
