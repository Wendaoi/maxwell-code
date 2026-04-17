import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import h5py
import numpy as np


def load_pong_setup_module():
    module_name = "pong_setup_test_module"
    module_path = Path(__file__).resolve().parents[1] / "pong_setup.py"

    fake_maxlab = types.ModuleType("maxlab")
    placeholder = type("Placeholder", (), {})
    fake_maxlab.Array = placeholder
    fake_maxlab.Sequence = placeholder
    fake_maxlab.Saving = placeholder
    fake_maxlab.StimulationUnit = placeholder
    fake_maxlab.DAC = placeholder
    fake_maxlab.Core = placeholder
    fake_maxlab.Timing = types.SimpleNamespace(waitAfterRecording=0)

    sys.modules["maxlab"] = fake_maxlab

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PongSetupAnalysisTest(unittest.TestCase):
    def test_analyze_recording_falls_back_to_runtime_events_when_h5_events_are_empty(self):
        pong_setup = load_pong_setup_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            recording_path = session_dir / "session.raw.h5"
            runtime_events_path = session_dir / "runtime_events.jsonl"

            with h5py.File(recording_path, "w") as f:
                raw = f.create_dataset(
                    "wells/well000/rec0000/groups/all_channels/raw",
                    data=np.zeros((2, 20000), dtype=np.int16),
                )
                f.create_dataset("wells/well000/rec0000/events", data=np.array([], dtype=np.int32))

            runtime_events_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "phase_start",
                                "frame": 0,
                                "phase": "game",
                                "detail": "game",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "miss",
                                "frame": 100,
                                "phase": "game",
                                "detail": "bounces=0,rally_id=0",
                            }
                        ),
                    ]
                )
                + "\n"
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pong_setup.analyze_recording(str(recording_path))

            text = output.getvalue()
            self.assertIn("2 runtime events", text)
            self.assertIn("miss: 1", text)
            self.assertNotIn("No events recorded", text)


if __name__ == "__main__":
    unittest.main()
