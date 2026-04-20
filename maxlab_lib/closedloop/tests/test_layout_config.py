import sys
import unittest
from unittest import mock
from contextlib import contextmanager
import importlib
import json
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layout_config import (
    ELECTRODE_COLUMNS,
    ENTROPY_CLUSTER_COUNT,
    ENTROPY_CLUSTER_SIZE,
    MOTOR_DOWN_LEFT_ELECTRODES,
    MOTOR_DOWN_RECORDING_ELECTRODES,
    MOTOR_DOWN_RECTS,
    MOTOR_DOWN_RIGHT_ELECTRODES,
    MOTOR_UP_LEFT_ELECTRODES,
    MOTOR_UP_RECORDING_ELECTRODES,
    MOTOR_UP_RECTS,
    MOTOR_UP_RIGHT_ELECTRODES,
    RECORDING_ELECTRODES,
    SENSORY_RECORDING_ELECTRODES,
    SENSORY_STIM_COORDS,
    SENSORY_STIM_ELECTRODES,
    build_electrode_metadata,
    coord_to_electrode,
    sample_rect_1_in_4,
)
from runtime_args import generate_cpp_args, normalize_condition


@contextmanager
def import_pong_setup_with_fake_maxlab():
    class FakeCommand:
        def __init__(self, kind, *args):
            self.kind = kind
            self.args = args

        def connect(self, value):
            return FakeCommand("StimulationUnit.connect", self.args[0], value)

        def power_up(self, value):
            return FakeCommand("StimulationUnit.power_up", self.args[0], value)

    class FakeSequence:
        instances = {}

        def __init__(self, name, persistent=False):
            self.name = name
            self.persistent = persistent
            self.commands = []
            self.sent = False
            FakeSequence.instances[name] = self

        def append(self, command):
            self.commands.append(command)

        def send(self):
            self.sent = True

    class FakeSaving:
        instances = []
        fail_group_name = None
        fail_methods = set()

        def __init__(self):
            self.calls = []
            FakeSaving.instances.append(self)

        def _maybe_fail(self, method, detail=None):
            if method in FakeSaving.fail_methods:
                raise RuntimeError(f"{method} failed")
            if method == "group_define" and detail == FakeSaving.fail_group_name:
                raise RuntimeError(f"group_define {detail} failed")

        def open_directory(self, path):
            self.calls.append(("open_directory", path))
            self._maybe_fail("open_directory")

        def start_file(self, name):
            self.calls.append(("start_file", name))
            self._maybe_fail("start_file")

        def group_delete_all(self):
            self.calls.append(("group_delete_all",))
            self._maybe_fail("group_delete_all")

        def group_define(self, well, name, channels):
            self.calls.append(("group_define", well, name, channels))
            self._maybe_fail("group_define", name)

        def start_recording(self, wells):
            self.calls.append(("start_recording", wells))
            self._maybe_fail("start_recording")

        def stop_recording(self):
            self.calls.append(("stop_recording",))
            self._maybe_fail("stop_recording")

        def stop_file(self):
            self.calls.append(("stop_file",))
            self._maybe_fail("stop_file")

    fake_maxlab = types.SimpleNamespace(
        Array=object,
        Sequence=FakeSequence,
        Saving=FakeSaving,
        StimulationUnit=lambda unit_id: FakeCommand("StimulationUnit", unit_id),
        Event=lambda *args: FakeCommand("Event", *args),
        DAC=lambda *args: FakeCommand("DAC", *args),
        DelaySamples=lambda samples: FakeCommand("DelaySamples", samples),
        query_DAC_lsb_mV=lambda: 1.0,
        Timing=types.SimpleNamespace(waitAfterRecording=0),
    )

    fake_numpy = types.SimpleNamespace(
        random=types.SimpleNamespace(
            Generator=object,
            default_rng=lambda seed: None,
        )
    )
    fake_h5py = types.SimpleNamespace(File=object)

    module_names = ("maxlab", "numpy", "h5py", "pong_setup")
    old_modules = {name: sys.modules.get(name) for name in module_names}
    missing = {name for name in module_names if name not in sys.modules}

    try:
        sys.modules["maxlab"] = fake_maxlab
        sys.modules["numpy"] = fake_numpy
        sys.modules["h5py"] = fake_h5py
        sys.modules.pop("pong_setup", None)

        pong_setup = importlib.import_module("pong_setup")
        yield pong_setup, FakeSaving
    finally:
        for name in module_names:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_modules[name]


class LayoutConfigTest(unittest.TestCase):
    def test_coord_to_electrode_uses_220_columns(self):
        self.assertEqual(ELECTRODE_COLUMNS, 220)
        self.assertEqual(coord_to_electrode(20, 25), 4425)
        self.assertEqual(coord_to_electrode(40, 200), 9000)

    def test_stimulation_electrodes_match_user_layout(self):
        expected = [4425, 8850, 4475, 8900, 4525, 8950, 4575, 9000]
        self.assertEqual(SENSORY_STIM_COORDS[0], (20, 25))
        self.assertEqual(SENSORY_STIM_COORDS[-1], (40, 200))
        self.assertEqual(SENSORY_STIM_ELECTRODES, expected)

    def test_one_in_four_sampling_is_upper_left_phase(self):
        sampled = sample_rect_1_in_4((80, 20), (109, 49))
        self.assertEqual(len(sampled), 225)
        self.assertEqual(sampled[0], coord_to_electrode(80, 20))
        self.assertEqual(sampled[1], coord_to_electrode(80, 22))
        self.assertEqual(sampled[15], coord_to_electrode(82, 20))
        self.assertEqual(sampled[-1], coord_to_electrode(108, 48))
        self.assertNotIn(coord_to_electrode(80, 21), sampled)
        self.assertNotIn(coord_to_electrode(109, 49), sampled)

    def test_motor_regions_are_900_recording_electrodes(self):
        self.assertEqual(MOTOR_DOWN_RECTS, [((80, 20), (109, 49)), ((80, 140), (109, 169))])
        self.assertEqual(MOTOR_UP_RECTS, [((80, 50), (109, 79)), ((80, 170), (109, 199))])

        for region in (
            MOTOR_DOWN_LEFT_ELECTRODES,
            MOTOR_DOWN_RIGHT_ELECTRODES,
            MOTOR_UP_LEFT_ELECTRODES,
            MOTOR_UP_RIGHT_ELECTRODES,
        ):
            self.assertEqual(len(region), 225)

        self.assertEqual(len(MOTOR_DOWN_RECORDING_ELECTRODES), 450)
        self.assertEqual(len(MOTOR_UP_RECORDING_ELECTRODES), 450)
        self.assertEqual(len(MOTOR_DOWN_RECORDING_ELECTRODES + MOTOR_UP_RECORDING_ELECTRODES), 900)

    def test_recording_layout_includes_sensory_area_and_entropy_clusters(self):
        self.assertEqual(len(SENSORY_RECORDING_ELECTRODES), 124)
        self.assertEqual(len(RECORDING_ELECTRODES), 1024)
        self.assertEqual(len(set(RECORDING_ELECTRODES)), 1024)
        self.assertTrue(set(SENSORY_STIM_ELECTRODES).isdisjoint(RECORDING_ELECTRODES))

        metadata = build_electrode_metadata(
            channel_lookup={electrode: idx for idx, electrode in enumerate(RECORDING_ELECTRODES)},
            stim_unit_by_electrode={electrode: unit for unit, electrode in enumerate(SENSORY_STIM_ELECTRODES)},
        )

        sensory = [entry for entry in metadata if entry["region"] == "sensory"]
        motor = [entry for entry in metadata if entry["region"].startswith("motor_")]
        clusters = {}
        for entry in metadata:
            if entry["entropy_cluster"] is not None:
                clusters.setdefault(entry["entropy_cluster"], []).append(entry)

        self.assertEqual(len(sensory), 124)
        self.assertEqual(len(motor), 900)
        self.assertEqual(len(clusters), ENTROPY_CLUSTER_COUNT)
        self.assertTrue(all(len(entries) == ENTROPY_CLUSTER_SIZE for entries in clusters.values()))


class RuntimeArgsTest(unittest.TestCase):
    def test_normalize_condition_aliases(self):
        self.assertEqual(normalize_condition("STIM"), "STIM")
        self.assertEqual(normalize_condition("STIMULUS"), "STIM")
        self.assertEqual(normalize_condition("silent"), "SILENT")
        self.assertEqual(normalize_condition("NO_FEEDBACK"), "NO_FEEDBACK")
        self.assertEqual(normalize_condition("NO-FEEDBACK"), "NO_FEEDBACK")
        self.assertEqual(normalize_condition("NOFEEDBACK"), "NO_FEEDBACK")
        self.assertEqual(normalize_condition("REST"), "REST")

        with self.assertRaises(ValueError):
            normalize_condition("RANDOM_STIMULUS")

    def test_generate_cpp_args_orders_runtime_fields(self):
        args = generate_cpp_args([0], "/tmp/session_config.json")
        self.assertEqual(
            args[:4],
            [
                "/tmp/session_config.json",
                "0",
                "1",
                "1",
            ],
        )

    def test_generate_cpp_args_rejects_multiple_wells(self):
        with self.assertRaises(ValueError):
            generate_cpp_args([0, 1], "/tmp/session_config.json")


class PongSetupMetadataTest(unittest.TestCase):
    def test_create_session_context_writes_metadata_and_paths(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, _):
            with tempfile.TemporaryDirectory() as tmpdir:
                runtime_params = dict(pong_setup.RUNTIME_PARAMS)
                runtime_params["pre_rest_seconds"] = 0
                runtime_params["game_seconds"] = 180
                context = pong_setup.create_session_context(
                    recording_root=tmpdir,
                    condition="STIM",
                    culture_id="culture_a",
                    cell_type="MCC",
                    replicate_id="mea_1",
                    experiment_day=2,
                    session_index=3,
                    operator="tester",
                    notes="dry run",
                    timestamp="20260416_120000",
                    runtime_params=runtime_params,
                )

                manifest = json.loads(Path(context["manifest_path"]).read_text())

        self.assertEqual(context["session_id"], "culture_a_day02_s03_STIM_20260416_120000")
        self.assertTrue(context["session_dir"].endswith(context["session_id"]))
        self.assertEqual(manifest["phase_durations"], {"pre_rest_seconds": 0, "game_seconds": 180})
        self.assertEqual(manifest["analysis_defaults"]["exclude_initial_game_seconds"], 10)
        self.assertEqual(manifest["metadata"]["cell_type"], "MCC")

    def test_ball_sequences_use_one_pulse_per_frequency_variant(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, _):
            sequences = pong_setup.prepare_decoupled_ball_sequences(
                {101: 1},
                [101],
                ["pos0"],
            )

            seq_40hz = sequences["pos0_40hz"]
            events = [cmd for cmd in seq_40hz.commands if cmd.kind == "Event"]
            delays = [cmd.args[0] for cmd in seq_40hz.commands if cmd.kind == "DelaySamples"]

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].args[3], "pos0_40hz_pulse")
            self.assertNotIn(int(pong_setup.RUNTIME_PARAMS["sample_rate_hz"] / 40), delays)

    def test_start_recording_defines_metadata_groups_for_target_well(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, fake_saving):
            cpp_config = {
                "channels": {
                    "motor_down_channels": [11, 12],
                    "motor_up_channels": [21, 22],
                    "stim_channels": [31, 32],
                    "sensory_channels": [41, 42],
                }
            }

            saving = pong_setup.start_recording("session_a", cpp_config, wells=[3])

            self.assertIs(saving, fake_saving.instances[-1])
            self.assertEqual(
                saving.calls,
                [
                    ("open_directory", pong_setup.RECORDING_DIR),
                    ("start_file", "session_a"),
                    ("group_delete_all",),
                    ("group_define", 3, "all_channels", list(range(1024))),
                    ("group_define", 3, "motor_down", [11, 12]),
                    ("group_define", 3, "motor_up", [21, 22]),
                    ("group_define", 3, "sensory", [41, 42]),
                    ("group_define", 3, "stim_channels", [31, 32]),
                    ("start_recording", [3]),
                ],
            )

    def test_start_recording_cleans_up_after_group_setup_failure(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, fake_saving):
            fake_saving.fail_group_name = "motor_down"
            cpp_config = {
                "channels": {
                    "motor_down_channels": [11, 12],
                    "motor_up_channels": [21, 22],
                    "stim_channels": [31, 32],
                    "sensory_channels": [41, 42],
                }
            }

            with self.assertRaisesRegex(RuntimeError, "group_define motor_down failed"):
                pong_setup.start_recording("session_a", cpp_config, wells=[3])

            self.assertEqual(
                fake_saving.instances[-1].calls,
                [
                    ("open_directory", pong_setup.RECORDING_DIR),
                    ("start_file", "session_a"),
                    ("group_delete_all",),
                    ("group_define", 3, "all_channels", list(range(1024))),
                    ("group_define", 3, "motor_down", [11, 12]),
                    ("stop_file",),
                    ("group_delete_all",),
                ],
            )

    def test_stop_recording_deletes_groups_after_stop_recording_failure(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, fake_saving):
            saving = fake_saving()
            fake_saving.fail_methods = {"stop_recording"}

            with self.assertRaisesRegex(RuntimeError, "stop_recording failed"):
                pong_setup.stop_recording(saving)

            self.assertEqual(
                saving.calls,
                [
                    ("stop_recording",),
                    ("stop_file",),
                    ("group_delete_all",),
                ],
            )

    def test_export_cpp_config_writes_recording_block(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, _):
            class FakeArrayConfig:
                def get_channels_for_electrodes(self, electrodes):
                    return [electrode % 1024 for electrode in electrodes]

            class FakeArray:
                def get_config(self):
                    return FakeArrayConfig()

            with tempfile.TemporaryDirectory() as tmpdir:
                old_recording_dir = pong_setup.RECORDING_DIR
                pong_setup.RECORDING_DIR = tmpdir
                config_path = Path(tmpdir) / "session_b_config.json"
                runtime_params = dict(pong_setup.RUNTIME_PARAMS)
                runtime_params["pre_rest_seconds"] = 0
                try:
                    cpp_config = pong_setup.export_cpp_config(
                        FakeArray(),
                        "STIM",
                        {4425: 1},
                        [4425],
                        ["pos0"],
                        {"pos0_4hz": object(), "hit_feedback": object()},
                        str(config_path),
                        session_name="session_b",
                        runtime_params=runtime_params,
                    )
                finally:
                    pong_setup.RECORDING_DIR = old_recording_dir

        self.assertEqual(
            cpp_config["recording"],
            {
                "session_name": "session_b",
                "raw_h5": str(Path(tmpdir) / "session_b.raw.h5"),
                "resolved_layout": str(Path(tmpdir) / "resolved_layout.json"),
                "runtime_events": str(Path(tmpdir) / "runtime_events.jsonl"),
                "window_samples": str(Path(tmpdir) / "window_samples.csv"),
                "quality_summary": str(Path(tmpdir) / "quality_summary.json"),
            },
        )
        self.assertEqual(cpp_config["runtime"]["pre_rest_seconds"], 0)
        self.assertEqual(cpp_config["runtime"]["sensory_blinding_ms"], 5)
        self.assertEqual(cpp_config["runtime"]["hit_feedback_blinding_ms"], 105)
        self.assertEqual(cpp_config["runtime"]["miss_feedback_blinding_ms"], 4005)

    def test_main_passes_pre_rest_seconds_to_run_pong_experiment(self):
        with import_pong_setup_with_fake_maxlab() as (pong_setup, _):
            captured = {}

            def fake_run_pong_experiment(**kwargs):
                captured.update(kwargs)

            with mock.patch.object(pong_setup, "run_pong_experiment", side_effect=fake_run_pong_experiment):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["pong_setup.py", "--pre-rest-seconds", "0", "--duration", "1"],
                ):
                    rc = pong_setup.main()

        self.assertEqual(rc, 0)
        self.assertEqual(captured["pre_rest_seconds"], 0)
        self.assertEqual(captured["duration_minutes"], 1)


if __name__ == "__main__":
    unittest.main()
