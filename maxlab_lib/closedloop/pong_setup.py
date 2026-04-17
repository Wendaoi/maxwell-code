#!/usr/bin/env python3
"""
pong_experiment_setup.py
------------------------

Complete setup script for the Pong closed-loop experiment with MaxLab system.

Modified to support fully decoupled position × frequency stimulation:
- 8 stimulation positions (ball positions)
- 10 stimulation frequencies (4-40Hz in 4Hz steps)
- 80 pre-generated sequences with embedded unit configuration
"""

import maxlab as mx
import numpy as np
import json
import time
import argparse
import subprocess
import threading
import os
import sys

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from h5py import File

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layout_config import (
    ENTROPY_CLUSTER_COUNT,
    ENTROPY_CLUSTER_SIZE,
    ELECTRODE_COLUMNS,
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

# ============================================================================
# ELECTRODE LAYOUT DEFINITION
# ============================================================================

STIM_ELECTRODES = SENSORY_STIM_ELECTRODES
MOTOR_1_UP_RECORDING_ELECTRODES = MOTOR_UP_LEFT_ELECTRODES
MOTOR_1_DOWN_RECORDING_ELECTRODES = MOTOR_DOWN_LEFT_ELECTRODES
MOTOR_2_UP_RECORDING_ELECTRODES = MOTOR_UP_RIGHT_ELECTRODES
MOTOR_2_DOWN_RECORDING_ELECTRODES = MOTOR_DOWN_RIGHT_ELECTRODES

# Position names for semantic mapping
POSITION_NAMES = [
    "pos0",
    "pos1",
    "pos2",
    "pos3",
    "pos4",
    "pos5",
    "pos6",
    "pos7"
]

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

RECORDING_DIR = str(Path.home() / "pong_experiments")
Path(RECORDING_DIR).mkdir(parents=True, exist_ok=True)

# C++ executable configuration
CPP_EXECUTABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/maxone_with_filter"
)

# Stimulation parameters
STIM_PARAMS = {
    "ball_position": {
        "amplitude_mV": 75,
        "phase_us": 200,
        "frequencies_Hz": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],  # 10 frequencies
    },
    "hit_feedback": {
        "amplitude_mV": 75,
        "phase_us": 200,
        "burst_frequency_Hz": 100,
        "burst_duration_ms": 100,
    },
    "miss_feedback": {
        "amplitude_mV": 150,
        "phase_us": 200,
        "burst_frequency_Hz": 5,
        "burst_duration_ms": 4000,
        "variant_count": 8,
    }
}

RUNTIME_PARAMS = {
    "sample_rate_hz": 20000,
    "window_ms": 10,
    "pre_rest_seconds": 600,
    "game_seconds": 1200,
    "exclude_initial_game_seconds": 10,
    "artifact_blanking_samples": 200,
    "miss_feedback_duration_ms": STIM_PARAMS["miss_feedback"]["burst_duration_ms"],
    "miss_pause_ms": 4000,
    "hit_sensory_suppression_ms": 100,
    "motor_gain_target_hz": 20.0,
    "stream_mode": "raw",
}

event_counter = 0  # Global event counter
STIM_NEIGHBOR_SEARCH_RADIUS = 2


def create_session_context(
    recording_root: str,
    condition: str,
    culture_id: str,
    cell_type: str,
    replicate_id: str,
    experiment_day: int,
    session_index: int,
    operator: str = "",
    notes: str = "",
    timestamp: Optional[str] = None,
    runtime_params: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Create a session directory and write metadata needed for later analysis."""
    runtime_params = runtime_params or RUNTIME_PARAMS
    normalized_condition = normalize_condition(condition)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_culture = culture_id.strip().replace(" ", "_")
    session_id = (
        f"{safe_culture}_day{experiment_day:02d}_s{session_index:02d}_"
        f"{normalized_condition}_{timestamp}"
    )
    session_dir = Path(recording_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "session_id": session_id,
        "metadata": {
            "culture_id": culture_id,
            "cell_type": cell_type,
            "replicate_id": replicate_id,
            "experiment_day": experiment_day,
            "session_index": session_index,
            "condition": normalized_condition,
            "operator": operator,
            "notes": notes,
            "timestamp": timestamp,
        },
        "phase_durations": {
            "pre_rest_seconds": runtime_params["pre_rest_seconds"],
            "game_seconds": runtime_params["game_seconds"],
        },
        "analysis_defaults": {
            "exclude_initial_game_seconds": runtime_params["exclude_initial_game_seconds"],
            "t1_minutes": [0, 5],
            "t2_minutes": [6, 20],
        },
        "files": {
            "raw_h5": str(session_dir / f"{session_id}.raw.h5"),
            "config": str(session_dir / "session_config.json"),
            "resolved_layout": str(session_dir / "resolved_layout.json"),
            "runtime_events": str(session_dir / "runtime_events.jsonl"),
            "window_samples": str(session_dir / "window_samples.csv"),
            "quality_summary": str(session_dir / "quality_summary.json"),
        },
    }

    manifest_path = session_dir / "session_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "manifest_path": str(manifest_path),
        **manifest["files"],
    }


# ============================================================================
# UTILITY FUNCTIONS FOR PROGRESS TRACKING
# ============================================================================

def print_step_header(step_number: int, step_name: str, total_steps: int = 14) -> None:
    """Print a formatted step header"""
    print("\n" + "=" * 70)
    print(f"STEP {step_number}/{total_steps}: {step_name}")
    print("=" * 70)


def print_substep(message: str, indent: int = 2) -> None:
    """Print a substep with indentation"""
    print(" " * indent + "→ " + message)


def print_success(message: str, indent: int = 2) -> None:
    """Print a success message"""
    print(" " * indent + "✓ " + message)


def print_info(message: str, indent: int = 4) -> None:
    """Print an info message"""
    print(" " * indent + "• " + message)


def print_warning(message: str, indent: int = 2) -> None:
    """Print a warning message"""
    print(" " * indent + "⚠ " + message)


def print_error(message: str, indent: int = 2) -> None:
    """Print an error message"""
    print(" " * indent + "✗ " + message)


# ============================================================================
# C++ PROCESS MANAGEMENT
# ============================================================================

class CPPProcessManager:
    """管理C++游戏进程的类"""

    def __init__(self, executable, args):
        self.executable = executable
        self.args = args
        self.process = None
        self.output_thread = None
        self.running = False
        self.ready_event = threading.Event()

    def start(self):
        """启动C++进程，使用stdin/stdout管道"""
        try:
            self.process = subprocess.Popen(
                [self.executable] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start C++ process: {e}")

        self.running = True
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()

        self._wait_for_ready()

    def _read_output(self):
        """后台线程持续读取C++输出"""
        ready_marker = "[SYNC] Waiting for start signal"
        while self.running and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    print(f"[C++ OUT] {line.strip()}")
                    if ready_marker in line and not self.ready_event.is_set():
                        self.ready_event.set()
            except Exception as e:
                break

    def _wait_for_ready(self):
        """等待C++进程启动并进入等待状态"""
        timeout = 10
        start_time = time.time()

        while time.time() - start_time < timeout and not self.ready_event.is_set():
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"C++ process exited before ready marker (exit code {self.process.returncode})"
                )
            time.sleep(0.1)

        if not self.ready_event.is_set():
            raise RuntimeError(f"C++ ready marker not seen within {timeout}s")

    def send_start_signal(self):
        """向C++进程发送启动信号"""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("C++ process is not running")

        try:
            self.process.stdin.write("start\n")
            self.process.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to send start signal: {e}")

    def raise_if_exited(self, context: str = "experiment"):
        """Raise if the C++ process has exited before Python requested shutdown."""
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"C++ process exited during {context} (exit code {self.process.returncode})"
            )

    def stop(self):
        """停止C++进程"""
        if self.process is None:
            return

        import signal
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.running = False
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1)

        self.process = None


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system() -> None:
    """Initialize system into a defined state"""
    print_step_header(2, "INITIALIZING MAXLAB SYSTEM")

    try:
        mx.initialize()
        result = mx.send(mx.Core().enable_stimulation_power(True))
        if result != "Ok":
            raise RuntimeError(f"Stimulation power enable returned: {result}")
        print_success("System initialized")
    except Exception as e:
        print_error(f"Initialization failed: {str(e)}")
        raise RuntimeError("The system didn't initialize correctly.")


# ============================================================================
# ARRAY CONFIGURATION
# ============================================================================
def configure_array(
    electrodes: List[int],
    stim_electrodes: List[int],
    stim_position_electrodes: Optional[List[int]] = None,
) -> mx.Array:
    """Configure array with recording and stimulation electrodes"""
    print_step_header(3, "CONFIGURING ELECTRODE ARRAY")

    print_substep("Creating and configuring array...")
    try:
        array = mx.Array("pong_experiment")
        array.reset()
        array.clear_selected_electrodes()
        array.select_electrodes(electrodes)
        array.select_stimulation_electrodes(stim_electrodes)
    except Exception as e:
        print_error(f"Array configuration failed: {str(e)}")
        raise

    print_substep("Routing electrodes (10-30s)...")
    try:
        start_time = time.time()
        array.route()
        elapsed = time.time() - start_time
        print_success(f"Routing completed ({elapsed:.1f}s): {len(electrodes)} rec + {len(stim_electrodes)} stim")
    except Exception as e:
        print_error(f"Routing failed: {str(e)}")
        raise

    return array


def build_stim_candidate_electrodes(
    stim_electrodes: List[int], max_radius: int = STIM_NEIGHBOR_SEARCH_RADIUS
) -> List[int]:
    """Expand stimulation electrodes with nearby neighbors for fallback routing"""
    if max_radius < 1:
        return list(stim_electrodes)

    candidates: List[int] = []
    seen = set()

    def add_candidate(electrode: int) -> None:
        if electrode not in seen:
            seen.add(electrode)
            candidates.append(electrode)

    for electrode in stim_electrodes:
        add_candidate(electrode)

    for electrode in stim_electrodes:
        for radius in range(1, max_radius + 1):
            for neighbor in mx.electrode_neighbors(electrode, radius):
                add_candidate(neighbor)

    return candidates


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int],
    array: mx.Array,
    candidate_electrodes: Optional[List[int]] = None,
    neighbor_search_radius: int = STIM_NEIGHBOR_SEARCH_RADIUS,
) -> Tuple[Dict[int, int], List[int]]:
    """Connect stimulation units to electrodes and return electrode->unit mapping

    Parameters
    ----------
    stim_electrodes : List[int]
        Requested stimulation electrodes (one per position).
    array : mx.Array
        Configured array with routing applied.
    candidate_electrodes : Optional[List[int]]
        Optional list of routed electrodes to consider as fallbacks.
    neighbor_search_radius : int
        Maximum radius for neighbor search when resolving conflicts.

    Returns
    -------
    Tuple[Dict[int, int], List[int]]
        Mapping from electrode ID to stimulation unit ID, and resolved
        stimulation electrode list (one per position)
    """
    print_step_header(4, "CONNECTING STIMULATION UNITS")

    electrode_to_unit: Dict[int, int] = {}
    used_units: List[int] = []
    used_electrodes = set()
    resolved_stim_electrodes: List[int] = []
    candidate_set = set(candidate_electrodes) if candidate_electrodes else None

    def disconnect_electrode(electrode: int) -> None:
        try:
            array.disconnect_electrode_from_stimulation(electrode)
        except Exception:
            pass

    def try_connect(electrode: int) -> Tuple[Optional[int], str]:
        try:
            array.connect_electrode_to_stimulation(electrode)
            stim = array.query_stimulation_at_electrode(electrode)
        except Exception as e:
            disconnect_electrode(electrode)
            raise e

        if len(stim) == 0:
            disconnect_electrode(electrode)
            return None, "no_unit"

        stim_unit_int = int(stim)
        if stim_unit_int in used_units:
            disconnect_electrode(electrode)
            return stim_unit_int, "unit_in_use"

        return stim_unit_int, ""

    def iter_neighbor_candidates(electrode: int) -> List[int]:
        if neighbor_search_radius < 1:
            return []
        neighbors: List[int] = []
        seen = set()
        for radius in range(1, neighbor_search_radius + 1):
            for neighbor in mx.electrode_neighbors(electrode, radius):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                if candidate_set is not None and neighbor not in candidate_set:
                    continue
                neighbors.append(neighbor)
        return neighbors

    for idx, stim_el in enumerate(stim_electrodes, 1):
        position_name = POSITION_NAMES[idx - 1]

        try:
            selected_electrode = stim_el
            attempted = [stim_el]
            stim_unit_int, reason = try_connect(stim_el)

            if reason:
                for neighbor in iter_neighbor_candidates(stim_el):
                    if neighbor in used_electrodes:
                        continue
                    attempted.append(neighbor)
                    stim_unit_int, reason = try_connect(neighbor)
                    if not reason:
                        selected_electrode = neighbor
                        break

                if reason and candidate_electrodes:
                    for candidate in candidate_electrodes:
                        if candidate in used_electrodes or candidate in attempted:
                            continue
                        attempted.append(candidate)
                        stim_unit_int, reason = try_connect(candidate)
                        if not reason:
                            selected_electrode = candidate
                            break

            if reason:
                tried = ", ".join(str(e) for e in attempted)
                if reason == "unit_in_use":
                    detail = (
                        f"Two electrodes connected to the same stim unit {stim_unit_int}.\n"
                        f"This is not allowed. Please select a neighboring electrode of {stim_el}!"
                    )
                else:
                    detail = (
                        f"No stimulation channel can connect to electrode: {stim_el}\n"
                        f"Please select a neighboring electrode instead."
                    )
                print_error(f"Connection failed after trying: {tried}")
                raise RuntimeError(detail)

            electrode_to_unit[selected_electrode] = stim_unit_int
            used_units.append(stim_unit_int)
            used_electrodes.add(selected_electrode)
            resolved_stim_electrodes.append(selected_electrode)

            if selected_electrode != stim_el:
                print_info(f"{position_name}: using electrode {selected_electrode} (instead of {stim_el})")

        except Exception as e:
            print_error(f"Connection failed for {position_name}: {str(e)}")
            raise

    mapping_str = ", ".join([f"E{e}:U{u}" for e, u in electrode_to_unit.items()])
    print_success(f"Connected {len(electrode_to_unit)} units: {mapping_str}")

    return electrode_to_unit, resolved_stim_electrodes


def configure_and_powerup_stim_units(stim_units: List[int]) -> None:
    """Power up and configure all stimulation units"""
    print_step_header(8, "POWERING UP STIMULATION UNITS")

    for idx, stim_unit in enumerate(stim_units, 1):
        try:
            stim = (
                mx.StimulationUnit(stim_unit)
                .power_up(True)
                .connect(True)
                .set_voltage_mode()
                .dac_source(0)
            )
            mx.send(stim)
        except Exception as e:
            print_error(f"Failed to power up unit {stim_unit}: {str(e)}")
            raise

    print_success(f"Powered up {len(stim_units)} units (voltage mode)")


# ============================================================================
# STIMULATION SEQUENCE PREPARATION (DECOUPLED ARCHITECTURE)
# ============================================================================

def amplitude_mV_to_DAC_bits(amplitude_mV: float) -> int:
    """Convert mV to DAC bits"""
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    
    if abs(amplitude_mV) > 1000.0:
        raise ValueError(f"Amplitude {amplitude_mV}mV exceeds 1.0V limit")
    
    return int(512 - (amplitude_mV / dac_lsb_mV))


def create_biphasic_pulse(
    seq: mx.Sequence,
    amplitude_mV: float,
    phase_us: int,
    event_label: str = ""
) -> mx.Sequence:
    """Create a single biphasic stimulation pulse"""
    global event_counter
    
    phase_samples = int(phase_us / 50)
    amplitude_bits = amplitude_mV_to_DAC_bits(amplitude_mV)
    
    if event_label:
        event_counter += 1
        seq.append(mx.Event(0, 1, event_counter, event_label))
    
    # Positive phase
    seq.append(mx.DAC(0, amplitude_bits))
    seq.append(mx.DelaySamples(phase_samples))
    
    # Negative phase
    seq.append(mx.DAC(0, amplitude_mV_to_DAC_bits(-amplitude_mV)))
    seq.append(mx.DelaySamples(phase_samples))
    
    # Return to baseline
    seq.append(mx.DAC(0, 512))
    
    return seq


def create_single_unit_configuration_commands(
    target_unit_id: int,
    all_unit_ids: List[int]
) -> List:
    return [
        mx.StimulationUnit(unit_id).connect(False)
        for unit_id in all_unit_ids
    ] + [
        mx.StimulationUnit(target_unit_id).connect(True).power_up(True)
    ]


def create_all_unit_configuration_commands(all_unit_ids: List[int]) -> List:
    return [
        mx.StimulationUnit(unit_id).connect(True).power_up(True)
        for unit_id in all_unit_ids
    ]


def _append_clamped_miss_interval(
    seq: mx.Sequence,
    delay_samples: int,
    elapsed_samples: int,
    total_samples: int,
    future_pulse_samples: int,
) -> int:
    """Append a miss interval without exceeding the total burst duration."""
    max_allowed = max(0, total_samples - elapsed_samples - future_pulse_samples)
    clamped_delay = min(max(0, delay_samples), max_allowed)
    if clamped_delay > 0:
        seq.append(mx.DelaySamples(clamped_delay))
    return clamped_delay


def _prepare_single_miss_feedback_sequence(
    seq_name: str,
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    rng: np.random.Generator,
    all_unit_ids: List[int],
) -> mx.Sequence:
    params = STIM_PARAMS["miss_feedback"]
    seq = mx.Sequence(name=seq_name, persistent=True)

    burst_duration_ms = params["burst_duration_ms"]
    base_frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * base_frequency_Hz)
    base_interval = int(RUNTIME_PARAMS["sample_rate_hz"] / base_frequency_Hz)
    total_samples = int(RUNTIME_PARAMS["sample_rate_hz"] * (burst_duration_ms / 1000.0))
    phase_samples = int(params["phase_us"] / 50)
    pulse_samples = 2 * phase_samples
    elapsed_samples = 0

    for pulse_idx in range(num_pulses):
        target_electrode = stim_electrodes[int(rng.integers(0, len(stim_electrodes)))]
        target_unit = electrode_to_unit[target_electrode]

        for cmd in create_single_unit_configuration_commands(target_unit, all_unit_ids):
            seq.append(cmd)

        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"{seq_name}_pulse_{pulse_idx+1}"
        )
        elapsed_samples += pulse_samples

        if pulse_idx < num_pulses - 1:
            jitter = int(base_interval * rng.uniform(-0.5, 0.5))
            delay_samples = base_interval + jitter
            future_pulse_samples = (num_pulses - pulse_idx - 1) * pulse_samples
            elapsed_samples += _append_clamped_miss_interval(
                seq,
                delay_samples=delay_samples,
                elapsed_samples=elapsed_samples,
                total_samples=total_samples,
                future_pulse_samples=future_pulse_samples,
            )

    if elapsed_samples < total_samples:
        seq.append(mx.DelaySamples(total_samples - elapsed_samples))

    seq.send()
    return seq


def prepare_decoupled_ball_sequences(
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    position_names: List[str],
) -> Dict[str, mx.Sequence]:
    """Prepare all position × frequency combinations (80 sequences)

    Each sequence contains:
    1. Unit configuration (activate target position, deactivate others)
    2. One short biphasic stimulation pulse.

    The frequency remains in the sequence name for C++ lookup; C++ schedules
    when each short sequence is triggered.

    Parameters
    ----------
    electrode_to_unit : Dict[int, int]
        Mapping from electrode ID to stimulation unit ID
    stim_electrodes : List[int]
        Resolved stimulation electrodes (one per position)
    position_names : List[str]
        Names for each stimulation position

    Returns
    -------
    Dict[str, mx.Sequence]
        Dictionary mapping sequence names to sequence objects
        Keys: "pos0_top_4hz", "pos0_top_8hz", ..., "pos7_bottom_40hz"
    """
    params = STIM_PARAMS["ball_position"]
    frequencies = params["frequencies_Hz"]
    total_sequences = len(stim_electrodes) * len(frequencies)

    sequences = {}
    all_unit_ids = list(electrode_to_unit.values())

    for pos_idx, (electrode, position_name) in enumerate(zip(stim_electrodes, position_names)):
        target_unit = electrode_to_unit[electrode]

        for freq_hz in frequencies:
            seq_name = f"{position_name}_{freq_hz}hz"
            seq = mx.Sequence(name=seq_name, persistent=True)

            unit_config_commands = create_single_unit_configuration_commands(
                target_unit_id=target_unit,
                all_unit_ids=all_unit_ids
            )

            for cmd in unit_config_commands:
                seq.append(cmd)

            create_biphasic_pulse(
                seq,
                amplitude_mV=params["amplitude_mV"],
                phase_us=params["phase_us"],
                event_label=f"{seq_name}_pulse"
            )

            seq.send()
            sequences[seq_name] = seq

    return sequences


def prepare_hit_feedback_sequence(all_unit_ids: List[int]) -> mx.Sequence:
    """Prepare burst sequence for successful paddle hit"""
    seq = mx.Sequence(name="hit_feedback", persistent=True)
    params = STIM_PARAMS["hit_feedback"]

    for cmd in create_all_unit_configuration_commands(all_unit_ids):
        seq.append(cmd)

    seq.append(mx.DelaySamples(100))

    burst_duration_ms = params["burst_duration_ms"]
    frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * frequency_Hz)
    interval_samples = int(RUNTIME_PARAMS["sample_rate_hz"] / frequency_Hz)

    for i in range(num_pulses):
        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"hit_feedback_pulse_{i+1}"
        )
        if i < num_pulses - 1:
            seq.append(mx.DelaySamples(interval_samples))

    seq.send()
    return seq


def prepare_miss_feedback_sequences(
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
) -> Dict[str, mx.Sequence]:
    """Prepare unpredictable sequences for missed balls"""
    params = STIM_PARAMS["miss_feedback"]
    sequences = {}
    all_unit_ids = list(electrode_to_unit.values())
    rng = np.random.default_rng(42)

    sequences["miss_feedback"] = _prepare_single_miss_feedback_sequence(
        "miss_feedback",
        electrode_to_unit,
        stim_electrodes,
        rng,
        all_unit_ids,
    )

    for variant_idx in range(params["variant_count"]):
        seq_name = f"miss_feedback_{variant_idx}"
        sequences[seq_name] = _prepare_single_miss_feedback_sequence(
            seq_name,
            electrode_to_unit,
            stim_electrodes,
            rng,
            all_unit_ids,
        )

    return sequences


def prepare_all_sequences(
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    position_names: List[str],
) -> Dict[str, mx.Sequence]:
    """Prepare all stimulation sequences for the experiment

    Parameters
    ----------
    electrode_to_unit : Dict[int, int]
        Mapping from electrode ID to stimulation unit ID
    stim_electrodes : List[int]
        Resolved stimulation electrodes (one per position)
    position_names : List[str]
        Names for each stimulation position

    Returns
    -------
    Dict[str, mx.Sequence]
        Dictionary of all named sequences (82 total)
    """
    print_step_header(9, "PREPARING STIMULATION SEQUENCES")

    position_count = len(position_names)
    frequency_count = len(STIM_PARAMS["ball_position"]["frequencies_Hz"])
    ball_sequence_count = position_count * frequency_count

    ball_sequences = prepare_decoupled_ball_sequences(
        electrode_to_unit, stim_electrodes, position_names
    )

    all_unit_ids = list(electrode_to_unit.values())
    hit_sequence = prepare_hit_feedback_sequence(all_unit_ids)
    miss_sequences = prepare_miss_feedback_sequences(electrode_to_unit, stim_electrodes)

    sequences = ball_sequences.copy()
    sequences["hit_feedback"] = hit_sequence
    sequences.update(miss_sequences)

    print_success(
        f"Prepared {len(sequences)} sequences ({ball_sequence_count} ball + {1 + STIM_PARAMS['miss_feedback']['variant_count']} feedback variants)"
    )

    return sequences


# ============================================================================
# RECORDING CONTROL
# ============================================================================

def _cleanup_saving(
    saving: mx.Saving,
    stop_active_recording: bool,
    close_file: bool,
    delete_groups: bool,
) -> Optional[Exception]:
    """Best-effort cleanup for MaxLab saving state; returns the first cleanup error."""
    first_error = None

    cleanup_steps = []
    if stop_active_recording:
        cleanup_steps.append(("stop recording", saving.stop_recording))
    if close_file:
        cleanup_steps.append(("stop file", saving.stop_file))
    if delete_groups:
        cleanup_steps.append(("delete groups", saving.group_delete_all))

    for step_name, cleanup in cleanup_steps:
        try:
            cleanup()
        except Exception as cleanup_error:
            if first_error is None:
                first_error = cleanup_error
            print_warning(f"Saving cleanup failed during {step_name}: {cleanup_error}")

    return first_error


def start_recording(
    recording_name: str,
    cpp_config: Dict,
    wells: List[int] = [0],
    recording_dir: str = RECORDING_DIR,
) -> mx.Saving:
    """Start recording data to HDF5 file"""
    print_step_header(10, "STARTING DATA RECORDING")

    recording_path = Path(recording_dir) / f"{recording_name}.raw.h5"
    target_well = wells[0]
    s = None
    file_started = False
    recording_attempted = False

    try:
        s = mx.Saving()
        s.open_directory(recording_dir)
        s.start_file(recording_name)
        file_started = True
        s.group_delete_all()
        s.group_define(target_well, "all_channels", list(range(1024)))
        s.group_define(target_well, "motor_down", cpp_config["channels"]["motor_down_channels"])
        s.group_define(target_well, "motor_up", cpp_config["channels"]["motor_up_channels"])
        s.group_define(target_well, "sensory", cpp_config["channels"]["sensory_channels"])
        s.group_define(target_well, "stim_channels", cpp_config["channels"]["stim_channels"])
        recording_attempted = True
        s.start_recording(wells)
        print_success(f"Recording: {recording_path.name}")
    except Exception as e:
        print_error(f"Recording failed: {str(e)}")
        if s is not None:
            _cleanup_saving(
                s,
                stop_active_recording=recording_attempted,
                close_file=file_started,
                delete_groups=file_started,
            )
        raise

    return s


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close file"""
    print_step_header(15, "STOPPING DATA RECORDING")

    cleanup_error = _cleanup_saving(
        saving,
        stop_active_recording=True,
        close_file=True,
        delete_groups=True,
    )

    try:
        time.sleep(mx.Timing.waitAfterRecording)
    except Exception as sleep_error:
        if cleanup_error is None:
            cleanup_error = sleep_error

    if cleanup_error is not None:
        print_error(f"Failed to stop recording: {cleanup_error}")
        raise cleanup_error

    print_success("Recording stopped")


# ============================================================================
# CONFIGURATION EXPORT FOR C++ MODULE
# ============================================================================

def export_cpp_config(
    array: mx.Array,
    condition: str,
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    position_names: List[str],
    sequences: Dict[str, mx.Sequence],
    config_path: str,
    session_name: str,
    runtime_params: Optional[Dict[str, object]] = None,
) -> Dict:
    """Export configuration for C++ closed-loop module

    Parameters
    ----------
    array : mx.Array
        Configured array
    electrode_to_unit : Dict[int, int]
        Mapping from electrode ID to stimulation unit ID
    stim_electrodes : List[int]
        Resolved stimulation electrodes (one per position)
    position_names : List[str]
        Names for each stimulation position
    sequences : Dict[str, mx.Sequence]
        Dictionary of prepared sequences
    config_path : str
        Path to save JSON config

    Returns
    -------
    Dict
        Configuration dictionary
    """
    print_step_header(11, "EXPORTING C++ CONFIGURATION")
    runtime_params = runtime_params or RUNTIME_PARAMS

    try:
        config = array.get_config()

        motor_1_up_channels = config.get_channels_for_electrodes(MOTOR_1_UP_RECORDING_ELECTRODES)
        motor_1_down_channels = config.get_channels_for_electrodes(MOTOR_1_DOWN_RECORDING_ELECTRODES)
        motor_2_up_channels = config.get_channels_for_electrodes(MOTOR_2_UP_RECORDING_ELECTRODES)
        motor_2_down_channels = config.get_channels_for_electrodes(MOTOR_2_DOWN_RECORDING_ELECTRODES)
        motor_down_channels = motor_1_down_channels + motor_2_down_channels
        motor_up_channels = motor_1_up_channels + motor_2_up_channels
        sensory_channels = config.get_channels_for_electrodes(SENSORY_RECORDING_ELECTRODES)
        recording_channels = config.get_channels_for_electrodes(RECORDING_ELECTRODES)
        stim_channels = config.get_channels_for_electrodes(stim_electrodes)

    except Exception as e:
        print_error(f"Failed to retrieve channel mappings: {str(e)}")
        raise

    ball_sequences = [k for k in sequences.keys() if any(pos in k for pos in position_names)]
    frequencies = STIM_PARAMS["ball_position"]["frequencies_Hz"]

    position_to_unit = {
        position_names[idx]: electrode_to_unit[electrode]
        for idx, electrode in enumerate(stim_electrodes)
    }
    channel_lookup = {
        electrode: channel
        for electrode, channel in zip(RECORDING_ELECTRODES, recording_channels)
    }
    electrode_metadata = build_electrode_metadata(channel_lookup, electrode_to_unit)
    resolved_layout_path = Path(config_path).with_name("resolved_layout.json")

    sequence_lookup = {}
    for pos_name in position_names:
        sequence_lookup[pos_name] = {}
        for freq in frequencies:
            seq_name = f"{pos_name}_{freq}hz"
            sequence_lookup[pos_name][freq] = seq_name

    cpp_config = {
        "experiment": {
            "name": "pong_closed_loop_4motor",
            "description": "Pong with 4 motor groups (Motor 1/2 × UP/DOWN)",
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "architecture": "4-channel motor control with position × frequency stimulation"
        },

        "condition": condition,
        "runtime": runtime_params,
        "recording": {
            "session_name": session_name,
            "raw_h5": str(Path(config_path).with_name(f"{session_name}.raw.h5")),
            "resolved_layout": str(resolved_layout_path),
            "runtime_events": str(Path(config_path).with_name("runtime_events.jsonl")),
            "window_samples": str(Path(config_path).with_name("window_samples.csv")),
            "quality_summary": str(Path(config_path).with_name("quality_summary.json")),
        },

        "electrodes": {
            "motor_down_left_recording": MOTOR_1_DOWN_RECORDING_ELECTRODES,
            "motor_down_right_recording": MOTOR_2_DOWN_RECORDING_ELECTRODES,
            "motor_up_left_recording": MOTOR_1_UP_RECORDING_ELECTRODES,
            "motor_up_right_recording": MOTOR_2_UP_RECORDING_ELECTRODES,
            "motor_down_recording": MOTOR_DOWN_RECORDING_ELECTRODES,
            "motor_up_recording": MOTOR_UP_RECORDING_ELECTRODES,
            "sensory_recording": SENSORY_RECORDING_ELECTRODES,
            "sensory_stimulation": stim_electrodes
        },

        "channels": {
            "motor_down_left_channels": motor_1_down_channels,
            "motor_down_right_channels": motor_2_down_channels,
            "motor_up_left_channels": motor_1_up_channels,
            "motor_up_right_channels": motor_2_up_channels,
            "motor_down_channels": motor_down_channels,
            "motor_up_channels": motor_up_channels,
            "sensory_channels": sensory_channels,
            "stim_channels": stim_channels
        },

        "analysis_layout": {
            "entropy_cluster_count": ENTROPY_CLUSTER_COUNT,
            "entropy_cluster_size": ENTROPY_CLUSTER_SIZE,
            "electrode_metadata": electrode_metadata,
        },

        "stimulation_units": {
            "electrode_to_unit": electrode_to_unit,
            "position_to_unit": position_to_unit,
            "positions": position_names
        },

        "stimulation_parameters": STIM_PARAMS,

        "sequences": {
            "ball_position": {
                "positions": position_names,
                "frequencies": frequencies,
                "sequence_lookup": sequence_lookup
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback"
            },
            "miss_feedback": {
                "sequence_names": sorted(k for k in sequences if k.startswith("miss_feedback_"))
            }
        },

        "cpp_usage": {
            "triggering": {
                "method": "maxlab::sendSequence(sequence_name)",
                "example_1": 'maxlab::sendSequence("pos0_40hz")',
                "example_2": 'maxlab::sendSequence("pos7_4hz")',
                "example_3": 'maxlab::sendSequence("hit_feedback")'
            },
            "sequence_selection": {
                "description": "Map ball position and distance to sequence name",
                "position_index": f"0-{len(position_names)-1} (top to bottom)",
                "frequency_index": "0-9 (4-40Hz in 4Hz steps)",
                "formula": f'std::string seq = positions[pos_idx] + "_" + std::to_string(frequencies[freq_idx]) + "hz"'
            }
        },

        "game_parameters": {
            "ball_speed": 0.5,
            "paddle_speed": 1.0,
            "paddle_height": 0.2,
            "update_rate_ms": 10,
            "num_positions": len(position_names),
            "position_mapping": {
                "description": f"Normalize ball Y position [0,1] to position index [0,{len(position_names)-1}]",
                "formula": f"pos_idx = int(ball_y * {len(position_names) - 0.01})"
            },
            "distance_to_frequency": {
                "description": "Map ball distance [0,1] to frequency index [0,9]",
                "formula": "freq_idx = int((1.0 - distance) * 9.99)"
            }
        },

        "motor_control": {
            "motor_1": {
                "up_channels": motor_1_up_channels,
                "down_channels": motor_1_down_channels,
                "electrodes_up": MOTOR_1_UP_RECORDING_ELECTRODES,
                "electrodes_down": MOTOR_1_DOWN_RECORDING_ELECTRODES,
                "control_logic": "spike_count(UP) - spike_count(DOWN)"
            },
            "motor_2": {
                "up_channels": motor_2_up_channels,
                "down_channels": motor_2_down_channels,
                "electrodes_up": MOTOR_2_UP_RECORDING_ELECTRODES,
                "electrodes_down": MOTOR_2_DOWN_RECORDING_ELECTRODES,
                "control_logic": "spike_count(UP) - spike_count(DOWN)"
            }
        },

        "spike_detection": {
            "threshold_std": 5.0,
            "refractory_period_ms": 2.0
        }
    }

    try:
        config_file = Path(config_path)
        with open(config_file, 'w') as f:
            json.dump(cpp_config, f, indent=2)
        with open(resolved_layout_path, 'w') as f:
            json.dump(cpp_config["analysis_layout"], f, indent=2)
        print_success(f"Config: {config_file.name}")
    except Exception as e:
        print_error(f"Failed to write config file: {str(e)}")
        raise

    return cpp_config


# ============================================================================
# POST-EXPERIMENT ANALYSIS
# ============================================================================

def analyze_recording(recording_path: str) -> None:
    """Perform basic analysis of recorded data"""
    print_step_header(16, "ANALYZING RECORDED DATA")

    try:
        with File(recording_path, 'r') as f:
            well_path = "wells/well000/rec0000"
            traces = f[f"{well_path}/groups/all_channels/raw"]
            events = f[f"{well_path}/events"]

            duration = traces.shape[1] / 20000
            runtime_events_path = Path(recording_path).with_name("runtime_events.jsonl")
            event_types = {}

            if len(events) > 0:
                print_success(f"Duration: {duration:.2f}s, {len(events)} events")
                for event in events:
                    event_desc = event[3].decode('utf-8') if isinstance(event[3], bytes) else str(event[3])
                    event_type = event_desc.split('_')[0] if '_' in event_desc else event_desc
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            elif runtime_events_path.exists():
                with open(runtime_events_path) as runtime_events_file:
                    runtime_events = [
                        json.loads(line)
                        for line in runtime_events_file
                        if line.strip()
                    ]
                print_success(f"Duration: {duration:.2f}s, {len(runtime_events)} runtime events")
                for event in runtime_events:
                    event_type = event.get("event", "unknown")
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            else:
                print_success(f"Duration: {duration:.2f}s, {len(events)} events")
                print_warning("No events recorded")

            for event_type, count in sorted(event_types.items()):
                print_info(f"  {event_type}: {count}")

    except Exception as e:
        print_error(f"Analysis failed: {str(e)}")
        raise


# ============================================================================
# MAIN EXPERIMENT WORKFLOW
# ============================================================================

def run_pong_experiment(
    duration_minutes: int = 20,
    condition: str = "STIM",
    wells: List[int] = [0],
    pre_rest_seconds: Optional[int] = None,
    culture_id: str = "unknown_culture",
    cell_type: str = "unknown",
    replicate_id: str = "unknown_replicate",
    experiment_day: int = 0,
    session_index: int = 1,
    operator: str = "",
    notes: str = "",
    recording_root: str = RECORDING_DIR,
) -> None:
    """Run complete Pong experiment with decoupled architecture"""
    condition = normalize_condition(condition)
    runtime_params = dict(RUNTIME_PARAMS)
    runtime_params["game_seconds"] = duration_minutes * 60
    if pre_rest_seconds is not None:
        runtime_params["pre_rest_seconds"] = pre_rest_seconds
    if int(runtime_params["pre_rest_seconds"]) < 0:
        raise ValueError("pre_rest_seconds must be non-negative")
    print("\n" + "=" * 70)
    print("PONG CLOSED-LOOP EXPERIMENT (DECOUPLED ARCHITECTURE)")
    print("=" * 70)
    print(f"Condition: {condition} | Duration: {duration_minutes}min | Wells: {wells}")

    session_context = create_session_context(
        recording_root=recording_root,
        condition=condition,
        culture_id=culture_id,
        cell_type=cell_type,
        replicate_id=replicate_id,
        experiment_day=experiment_day,
        session_index=session_index,
        operator=operator,
        notes=notes,
        runtime_params=runtime_params,
    )
    session_name = session_context["session_id"]
    config_path = Path(session_context["config"])
    # Validate the single-well C++ contract before touching MaxLab hardware.
    cpp_args = generate_cpp_args(wells, str(config_path))
    cpp_manager = None
    saving = None
    recording_active = False

    # Step 0: Check C++ executable
    print_step_header(0, "CHECKING C++ EXECUTABLE")
    if not os.path.exists(CPP_EXECUTABLE):
        print_error(f"C++ executable not found: {CPP_EXECUTABLE}")
        print_error("Please build the C++ program first:")
        print_error("  cd maxlab_lib && make maxone_with_filter")
        raise RuntimeError("C++ executable not found")
    print_success("C++ executable found")

    # Step 2: Initialize system
    initialize_system()
    print_substep(f"Waiting {mx.Timing.waitInit}s for system stabilization...")
    time.sleep(mx.Timing.waitInit)
    print_success("System ready")
    
    # Step 3: Configure array
    all_recording = RECORDING_ELECTRODES
    
    stim_candidates = build_stim_candidate_electrodes(SENSORY_STIM_ELECTRODES)
    array = configure_array(
        all_recording,
        stim_candidates,
        stim_position_electrodes=SENSORY_STIM_ELECTRODES,
    )
    
    # Step 4: Connect stimulation (returns electrode->unit mapping)
    electrode_to_unit, resolved_stim_electrodes = connect_stim_units_to_stim_electrodes(
        SENSORY_STIM_ELECTRODES,
        array,
        candidate_electrodes=stim_candidates,
        neighbor_search_radius=STIM_NEIGHBOR_SEARCH_RADIUS,
    )
    
    # Step 5: Activate wells
    print_step_header(5, "ACTIVATING WELLS")
    try:
        mx.activate(wells)
        print_success(f"Wells {wells} activated")
    except Exception as e:
        print_error(f"Failed to activate wells: {str(e)}")
        raise

    # Step 6: Download configuration
    print_step_header(6, "DOWNLOADING CONFIGURATION")
    print_substep("Transferring to chip (may take several seconds)...")
    try:
        array.download(wells)
        time.sleep(mx.Timing.waitAfterDownload)
        print_success("Configuration downloaded")
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        raise

    # Step 7: Calibration
    print_step_header(7, "CALIBRATION")
    print_substep("Running offset compensation (15s)...")
    try:
        mx.offset()
        for i in range(15):
            time.sleep(1)
            if (i + 1) % 5 == 0:
                print_info(f"{15 - i}s remaining")
        print_success("Calibration complete")
    except Exception as e:
        print_error(f"Calibration failed: {str(e)}")
        raise
    
    # Step 8: Power up stimulation units
    all_units = list(electrode_to_unit.values())
    configure_and_powerup_stim_units(all_units)

    # Step 9: Clear event buffer
    print_step_header(9, "CLEARING EVENT BUFFER")
    try:
        mx.clear_events()
        print_success("Event buffer cleared")
    except Exception as e:
        print_error(f"Failed to clear events: {str(e)}")
        raise

    # Step 10: Prepare all decoupled sequences
    sequences = prepare_all_sequences(
        electrode_to_unit, resolved_stim_electrodes, POSITION_NAMES
    )

    # Step 11: Export configuration for C++
    cpp_config = export_cpp_config(
        array,
        condition,
        electrode_to_unit,
        resolved_stim_electrodes,
        POSITION_NAMES,
        sequences,
        str(config_path),
        session_name,
        runtime_params=runtime_params,
    )

    # Step 12: Start recording
    saving = start_recording(session_name, cpp_config, wells, recording_dir=session_context["session_dir"])
    recording_active = True

    try:
        # Step 12b: Start C++ game process after MaxLab setup and config export.
        print_step_header(12, "STARTING C++ GAME PROCESS")
        try:
            cpp_manager = CPPProcessManager(CPP_EXECUTABLE, cpp_args)
            cpp_manager.start()
            print_success("C++ game process started successfully")
        except Exception as e:
            print_error(f"Failed to start C++ process: {e}")
            raise RuntimeError(f"C++ process startup failed: {e}")

        # Step 13: Baseline recording
        print_step_header(13, "BASELINE RECORDING")
        baseline_duration = int(runtime_params["pre_rest_seconds"])
        if baseline_duration == 0:
            print_substep("Skipping baseline recording (pre_rest_seconds=0)")
        else:
            print_substep(f"Recording {baseline_duration}s baseline...")
        for i in range(baseline_duration):
            cpp_manager.raise_if_exited("baseline recording")
            if (i + 1) % 10 == 0:
                print_info(f"{baseline_duration - i}s remaining")
            time.sleep(1)
        print_success("Baseline complete")

        # Step 14: C++ module integration
        print_step_header(14, "C++ MODULE INTEGRATION")
        print_substep("Sending start signal to C++ game...")
        try:
            cpp_manager.send_start_signal()
            print_success("C++ game loop started")
        except Exception as e:
            print_error(f"Failed to send start signal: {e}")
            print_error("Aborting experiment due to sync failure")
            raise RuntimeError(f"C++ sync failed: {e}")

        time.sleep(2)
        cpp_manager.raise_if_exited("game startup")

        print_substep(f"Running experiment for {duration_minutes} minutes...")
        total_seconds = duration_minutes * 60
        last_update = 0

        for i in range(total_seconds):
            cpp_manager.raise_if_exited("experiment")

            current_minute = i // 60
            if current_minute != last_update:
                minutes_remaining = (total_seconds - i) // 60
                print_info(f"{minutes_remaining} min remaining")
                last_update = current_minute
            time.sleep(1)

        cpp_manager.raise_if_exited("experiment completion")
        print_success("Experiment complete")

        # Step 15: Stop recording
        try:
            stop_recording(saving)
        finally:
            recording_active = False

        # Step 16: Analyze results
        recording_path = Path(session_context["raw_h5"])
        if recording_path.exists():
            analyze_recording(str(recording_path))
        else:
            print_warning(f"Recording file not found: {recording_path}")
    finally:
        # Cleanup: Stop C++ process if still running
        if cpp_manager is not None:
            print("\n[INFO] Stopping C++ process...")
            cpp_manager.stop()
        if recording_active and saving is not None:
            print("\n[INFO] Stopping recording after error...")
            try:
                stop_recording(saving)
            except Exception as cleanup_error:
                print_error(f"Recording cleanup failed: {cleanup_error}")
            finally:
                recording_active = False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaxLab Pong Experiment (Decoupled Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 20 --condition STIM
  %(prog)s --duration 30 --condition NO_FEEDBACK --wells 0 1
        """
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Experiment duration in minutes (default: 20)"
    )
    parser.add_argument(
        "--pre-rest-seconds",
        type=int,
        default=RUNTIME_PARAMS["pre_rest_seconds"],
        help="Baseline duration before the game starts in seconds; use 0 to skip (default: 600)",
    )

    parser.add_argument(
        "--condition",
        type=str,
        choices=["STIM", "STIMULUS", "SILENT", "NO_FEEDBACK", "NO-FEEDBACK", "REST"],
        default="STIM",
        help="Experimental condition (default: STIM)"
    )

    parser.add_argument(
        "--wells",
        type=int,
        nargs='+',
        default=[0],
        help="Well indices to use (default: 0)"
    )
    parser.add_argument("--culture-id", default="unknown_culture", help="Culture identifier")
    parser.add_argument("--cell-type", default="unknown", help="Cell type, e.g. MCC/HCC_DSI/HCC_NGN2/HEK/CTL")
    parser.add_argument("--replicate-id", default="unknown_replicate", help="MEA or biological replicate identifier")
    parser.add_argument("--experiment-day", type=int, default=0, help="Experiment day index")
    parser.add_argument("--session-index", type=int, default=1, help="Session index within the experiment day")
    parser.add_argument("--operator", default="", help="Operator name or initials")
    parser.add_argument("--notes", default="", help="Free-text session notes")
    parser.add_argument("--recording-root", default=RECORDING_DIR, help="Root directory for session outputs")

    args = parser.parse_args()

    try:
        run_pong_experiment(
            duration_minutes=args.duration,
            condition=args.condition,
            wells=args.wells,
            pre_rest_seconds=args.pre_rest_seconds,
            culture_id=args.culture_id,
            cell_type=args.cell_type,
            replicate_id=args.replicate_id,
            experiment_day=args.experiment_day,
            session_index=args.session_index,
            operator=args.operator,
            notes=args.notes,
            recording_root=args.recording_root,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("EXPERIMENT INTERRUPTED BY USER")
        print("=" * 70)
        return 130
    except Exception as e:
        print("\n\n" + "=" * 70)
        print("EXPERIMENT FAILED")
        print("=" * 70)
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
