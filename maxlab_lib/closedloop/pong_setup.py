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

# ============================================================================
# ELECTRODE LAYOUT DEFINITION
# ============================================================================

# Recording electrodes pool (reusing from stimulate.py example)
# fmt: off
STIM_ELECTRODES = [
    3344, 3388, 3432, 3476, 6644, 6688, 6732, 6776,     
]
# fmt: on

# ============================================================================
# STIMULATION POSITION CONFIGURATION (8 positions)
# ============================================================================

# 8 distinct stimulation positions corresponding to 8 ball vertical positions
SENSORY_STIM_ELECTRODES = [
    STIM_ELECTRODES[0],  # Position 0 (top)
    STIM_ELECTRODES[1],  # Position 1
    STIM_ELECTRODES[2],  # Position 2
    STIM_ELECTRODES[3],  # Position 3 (middle)
    STIM_ELECTRODES[4],  # Position 4
    STIM_ELECTRODES[5],  # Position 5
    STIM_ELECTRODES[6],  # Position 6
    STIM_ELECTRODES[7],  # Position 7 (bottom)
]

"""
Generate electrode configuration with vertical expansion
"""

def generate_vertical_electrodes(base_electrode, num_rows=10, row_offset=220):
    """Generate vertical array of electrodes"""
    return [base_electrode + i * row_offset for i in range(num_rows)]

def generate_electrode_pool():
    """Generate full electrode pool with vertical expansion"""
    
    # Generate base electrode ranges
    base = 13200
    range1_base = list(range(base + 15, base + 55, 2))   # 13210 to 13230
    range2_base = list(range(base + 56, base + 95, 2))   # 13230 to 13250
    range3_base = list(range(base + 125, base + 165, 2))   # 13270 to 13290
    range4_base = list(range(base + 166, base + 205, 2))  # 13290 to 13310
    
    # Combine and remove duplicates
    all_base = range1_base + range2_base + range3_base + range4_base
    unique_base = list(dict.fromkeys(all_base))  # 42 unique base electrodes
    
    # Expand each base into 10 vertical rows
    recording_electrodes = []
    for base_electrode in unique_base:
        vertical_array = generate_vertical_electrodes(base_electrode, num_rows=10, row_offset=220)
        recording_electrodes.extend(vertical_array)
    
    return recording_electrodes, unique_base

def print_electrode_array(electrodes, name="RECORDING_ELECTRODES"):
    """Print electrode array in formatted Python code"""
    print(f"\n# fmt: off")
    print(f"{name} = [")
    
    # Print in groups of 10 (each vertical column)
    for i in range(0, len(electrodes), 10):
        chunk = electrodes[i:i+10]
        base = chunk[0]
        print(f"    # Base {base} (10 rows)")
        print(f"    {', '.join(map(str, chunk))},")
        if i < len(electrodes) - 10:
            print()
    
    print(f"]")
    print(f"# fmt: on")
    print(f"\n# Total: {len(electrodes)} electrodes")

# Generate electrodes (silent)
RECORDING_ELECTRODES, base_electrodes = generate_electrode_pool()

# Validate
assert len(RECORDING_ELECTRODES) == 800, "Should have 800 electrodes"
assert len(set(RECORDING_ELECTRODES)) == 800, "All electrodes should be unique"

# Generate functional groups
MOTOR_1_UP_RECORDING_ELECTRODES = RECORDING_ELECTRODES[0:200]      # 20 base × 10
MOTOR_1_DOWN_RECORDING_ELECTRODES = RECORDING_ELECTRODES[200:400]  # 20 base × 10
MOTOR_2_UP_RECORDING_ELECTRODES = RECORDING_ELECTRODES[400:600]    # 20 base × 10
MOTOR_2_DOWN_RECORDING_ELECTRODES = RECORDING_ELECTRODES[600:800]  # 20 base × 10

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

# C++ program arguments (will be dynamically generated based on experiment wells)
# Format: [target_well, window_ms, blanking_frames, show_gui, sample_rate_hz, threshold, min_threshold, refractory_samples, channel_count, wait_for_sync]
_CPP_ARGS_TEMPLATE = {
    "window_ms": 5,
    "blanking_frames": 8000,
    "show_gui": 1,
    "sample_rate_hz": 20000,
    "threshold_multiplier": 5.0,
    "min_threshold": -20,
    "refractory_samples": 1000,
    "channel_count": 1024,
    "wait_for_sync": 1
}

def generate_cpp_args(wells: List[int]) -> List[str]:
    """Generate C++ program arguments based on experiment configuration"""
    target_well = wells[0] if wells else 0
    return [
        str(target_well),
        str(_CPP_ARGS_TEMPLATE["window_ms"]),
        str(_CPP_ARGS_TEMPLATE["blanking_frames"]),
        str(_CPP_ARGS_TEMPLATE["show_gui"]),
        str(_CPP_ARGS_TEMPLATE["sample_rate_hz"]),
        str(_CPP_ARGS_TEMPLATE["threshold_multiplier"]),
        str(_CPP_ARGS_TEMPLATE["min_threshold"]),
        str(_CPP_ARGS_TEMPLATE["refractory_samples"]),
        str(_CPP_ARGS_TEMPLATE["channel_count"]),
        str(_CPP_ARGS_TEMPLATE["wait_for_sync"])
    ]

# Stimulation parameters
STIM_PARAMS = {
    "ball_position": {
        "amplitude_mV": 50,
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
    }
}

event_counter = 0  # Global event counter
STIM_NEIGHBOR_SEARCH_RADIUS = 2


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
            time.sleep(0.1)

        if not self.ready_event.is_set():
            print(f"[C++] Warning: Did not see ready marker within {timeout}s")

    def send_start_signal(self):
        """向C++进程发送启动信号"""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("C++ process is not running")

        try:
            self.process.stdin.write("start\n")
            self.process.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to send start signal: {e}")

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


def create_unit_configuration_commands(
    target_unit_id: int,
    all_unit_ids: List[int]
) -> List:
    """Generate commands to activate only the target stimulation unit
    
    This ensures the sequence only stimulates at the desired position.
    
    Parameters
    ----------
    target_unit_id : int
        The stimulation unit to activate
    all_unit_ids : List[int]
        All available stimulation unit IDs
        
    Returns
    -------
    List
        List of mx.StimulationUnit configuration commands
    """
    commands = []
    
    # Disconnect all units first
    for unit_id in all_unit_ids:
        commands.append(
            mx.StimulationUnit(unit_id).connect(False)
        )
    
    # Connect and activate only the target unit
    commands.append(
        mx.StimulationUnit(target_unit_id)
        .connect(True)
        .power_up(True)
    )
    
    return commands


def prepare_decoupled_ball_sequences(
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    position_names: List[str],
) -> Dict[str, mx.Sequence]:
    """Prepare all position × frequency combinations (80 sequences)

    Each sequence contains:
    1. Unit configuration (activate target position, deactivate others)
    2. Stimulation pulses at specified frequency

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

            unit_config_commands = create_unit_configuration_commands(
                target_unit_id=target_unit,
                all_unit_ids=all_unit_ids
            )

            for cmd in unit_config_commands:
                seq.append(cmd)

            seq.append(mx.DelaySamples(100))

            num_pulses = freq_hz
            interval_samples = int(20000 / freq_hz)

            for pulse_idx in range(num_pulses):
                create_biphasic_pulse(
                    seq,
                    amplitude_mV=params["amplitude_mV"],
                    phase_us=params["phase_us"],
                    event_label=f"{seq_name}_pulse_{pulse_idx+1}"
                )

                if pulse_idx < num_pulses - 1:
                    seq.append(mx.DelaySamples(interval_samples))

            seq.send()
            sequences[seq_name] = seq

    return sequences


def prepare_hit_feedback_sequence() -> mx.Sequence:
    """Prepare burst sequence for successful paddle hit"""
    seq = mx.Sequence(name="hit_feedback", persistent=True)
    params = STIM_PARAMS["hit_feedback"]

    burst_duration_ms = params["burst_duration_ms"]
    frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * frequency_Hz)
    interval_samples = int(20000 / frequency_Hz)

    for i in range(num_pulses):
        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"hit_feedback_pulse_{i+1}"
        )
        seq.append(mx.DelaySamples(interval_samples))

    seq.send()
    return seq


def prepare_miss_feedback_sequence() -> mx.Sequence:
    """Prepare unpredictable sequence for missed ball"""
    seq = mx.Sequence(name="miss_feedback", persistent=True)
    params = STIM_PARAMS["miss_feedback"]

    burst_duration_ms = params["burst_duration_ms"]
    base_frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * base_frequency_Hz)

    np.random.seed(42)

    for i in range(num_pulses):
        amplitude = params["amplitude_mV"] * np.random.uniform(0.8, 1.2)

        create_biphasic_pulse(
            seq,
            amplitude_mV=amplitude,
            phase_us=params["phase_us"],
            event_label=f"miss_feedback_pulse_{i+1}"
        )

        base_interval = int(20000 / base_frequency_Hz)
        jitter = int(base_interval * np.random.uniform(0.5, 1.5))
        seq.append(mx.DelaySamples(base_interval + jitter))

    seq.send()
    return seq


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

    hit_sequence = prepare_hit_feedback_sequence()
    miss_sequence = prepare_miss_feedback_sequence()

    sequences = ball_sequences.copy()
    sequences["hit_feedback"] = hit_sequence
    sequences["miss_feedback"] = miss_sequence

    print_success(f"Prepared {len(sequences)} sequences ({ball_sequence_count} ball + 2 feedback)")

    return sequences


# ============================================================================
# RECORDING CONTROL
# ============================================================================

def start_recording(
    recording_name: str,
    wells: List[int] = [0]
) -> mx.Saving:
    """Start recording data to HDF5 file"""
    print_step_header(10, "STARTING DATA RECORDING")

    recording_path = Path(RECORDING_DIR) / f"{recording_name}.raw.h5"

    try:
        s = mx.Saving()
        s.open_directory(RECORDING_DIR)
        s.start_file(recording_name)
        s.group_define(0, "all_channels", list(range(1024)))
        s.start_recording(wells)
        print_success(f"Recording: {recording_path.name}")
    except Exception as e:
        print_error(f"Recording failed: {str(e)}")
        raise

    return s


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close file"""
    print_step_header(15, "STOPPING DATA RECORDING")

    try:
        saving.stop_recording()
        saving.stop_file()
        saving.group_delete_all()
        time.sleep(mx.Timing.waitAfterRecording)
        print_success("Recording stopped")
    except Exception as e:
        print_error(f"Failed to stop recording: {str(e)}")
        raise


# ============================================================================
# CONFIGURATION EXPORT FOR C++ MODULE
# ============================================================================

def export_cpp_config(
    array: mx.Array,
    electrode_to_unit: Dict[int, int],
    stim_electrodes: List[int],
    position_names: List[str],
    sequences: Dict[str, mx.Sequence],
    config_path: str
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

    try:
        config = array.get_config()

        motor_1_up_channels = config.get_channels_for_electrodes(MOTOR_1_UP_RECORDING_ELECTRODES)
        motor_1_down_channels = config.get_channels_for_electrodes(MOTOR_1_DOWN_RECORDING_ELECTRODES)
        motor_2_up_channels = config.get_channels_for_electrodes(MOTOR_2_UP_RECORDING_ELECTRODES)
        motor_2_down_channels = config.get_channels_for_electrodes(MOTOR_2_DOWN_RECORDING_ELECTRODES)
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

        "electrodes": {
            "motor_1_up_recording": MOTOR_1_UP_RECORDING_ELECTRODES,
            "motor_1_down_recording": MOTOR_1_DOWN_RECORDING_ELECTRODES,
            "motor_2_up_recording": MOTOR_2_UP_RECORDING_ELECTRODES,
            "motor_2_down_recording": MOTOR_2_DOWN_RECORDING_ELECTRODES,
            "sensory_stimulation": stim_electrodes
        },

        "channels": {
            "motor_1_up_channels": motor_1_up_channels,
            "motor_1_down_channels": motor_1_down_channels,
            "motor_2_up_channels": motor_2_up_channels,
            "motor_2_down_channels": motor_2_down_channels,
            "stim_channels": stim_channels
        },

        "stimulation_units": {
            "electrode_to_unit": electrode_to_unit,
            "position_to_unit": position_to_unit,
            "positions": position_names
        },

        "stimulation_parameters": STIM_PARAMS,

        "sequences": {
            "ball_position": {
                "total_sequences": len(ball_sequences),
                "positions": position_names,
                "frequencies": frequencies,
                "sequence_lookup": sequence_lookup,
                "usage_example": "pos3_20hz triggers 20Hz at position 3"
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback",
                "trigger": "on_paddle_hit"
            },
            "miss_feedback": {
                "sequence_name": "miss_feedback",
                "trigger": "on_ball_miss"
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
            print_success(f"Duration: {duration:.2f}s, {len(events)} events")

            if len(events) > 0:
                event_types = {}
                for event in events:
                    event_desc = event[3].decode('utf-8') if isinstance(event[3], bytes) else str(event[3])
                    event_type = event_desc.split('_')[0] if '_' in event_desc else event_desc
                    event_types[event_type] = event_types.get(event_type, 0) + 1

                for event_type, count in sorted(event_types.items()):
                    print_info(f"  {event_type}: {count}")
            else:
                print_warning("No events recorded")

    except Exception as e:
        print_error(f"Analysis failed: {str(e)}")
        raise


# ============================================================================
# MAIN EXPERIMENT WORKFLOW
# ============================================================================

def run_pong_experiment(
    duration_minutes: int = 20,
    condition: str = "STIMULUS",
    wells: List[int] = [0]
) -> None:
    """Run complete Pong experiment with decoupled architecture"""
    print("\n" + "=" * 70)
    print("PONG CLOSED-LOOP EXPERIMENT (DECOUPLED ARCHITECTURE)")
    print("=" * 70)
    print(f"Condition: {condition} | Duration: {duration_minutes}min | Wells: {wells}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"pong_{condition}_{timestamp}"
    config_path = Path(RECORDING_DIR) / f"{session_name}_config.json"

    # Step 0: Check C++ executable
    print_step_header(0, "CHECKING C++ EXECUTABLE")
    if not os.path.exists(CPP_EXECUTABLE):
        print_error(f"C++ executable not found: {CPP_EXECUTABLE}")
        print_error("Please build the C++ program first:")
        print_error("  cd maxlab_lib && make maxone_with_filter")
        raise RuntimeError("C++ executable not found")
    print_success("C++ executable found")

    # Step 1: Start C++ game process
    print_step_header(1, "STARTING C++ GAME PROCESS")
    cpp_args = generate_cpp_args(wells)
    cpp_manager = None
    try:
        cpp_manager = CPPProcessManager(CPP_EXECUTABLE, cpp_args)
        cpp_manager.start()
        print_success("C++ game process started successfully")
    except Exception as e:
        print_error(f"Failed to start C++ process: {e}")
        raise RuntimeError(f"C++ process startup failed: {e}")

    # Step 2: Initialize system
    initialize_system()
    print_substep(f"Waiting {mx.Timing.waitInit}s for system stabilization...")
    time.sleep(mx.Timing.waitInit)
    print_success("System ready")
    
    # Step 3: Configure array
    all_recording = list(set(
        MOTOR_1_UP_RECORDING_ELECTRODES +
        MOTOR_1_DOWN_RECORDING_ELECTRODES +
        MOTOR_2_UP_RECORDING_ELECTRODES +
        MOTOR_2_DOWN_RECORDING_ELECTRODES 
    ))
    
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
        electrode_to_unit,
        resolved_stim_electrodes,
        POSITION_NAMES,
        sequences,
        str(config_path),
    )

    # Step 12: Start recording
    saving = start_recording(session_name, wells)

    # Step 13: Baseline recording
    print_step_header(13, "BASELINE RECORDING")
    baseline_duration = 30
    print_substep(f"Recording {baseline_duration}s baseline...")
    for i in range(baseline_duration):
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

    print_substep(f"Running experiment for {duration_minutes} minutes...")
    total_seconds = duration_minutes * 60
    last_update = 0

    for i in range(total_seconds):
        if cpp_manager.process.poll() is not None:
            print_error("C++ process exited early")
            print_error(f"Exit code: {cpp_manager.process.returncode}")
            break

        current_minute = i // 60
        if current_minute != last_update:
            minutes_remaining = (total_seconds - i) // 60
            print_info(f"{minutes_remaining} min remaining")
            last_update = current_minute
        time.sleep(1)

    if cpp_manager.process.poll() is None:
        print_success("Experiment complete")
    else:
        print_error("Experiment terminated early")

    # Step 15: Stop recording
    stop_recording(saving)

    # Step 16: Analyze results
    recording_path = Path(RECORDING_DIR) / f"{session_name}.raw.h5"
    if recording_path.exists():
        analyze_recording(str(recording_path))
    else:
        print_warning(f"Recording file not found: {recording_path}")

    # Cleanup: Stop C++ process if still running
    if 'cpp_manager' in locals() and cpp_manager is not None:
        print("\n[INFO] Stopping C++ process...")
        cpp_manager.stop()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaxLab Pong Experiment (Decoupled Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 20 --condition STIMULUS
  %(prog)s --duration 30 --condition NO_STIMULUS --wells 0 1
        """
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Experiment duration in minutes (default: 20)"
    )

    parser.add_argument(
        "--condition",
        type=str,
        choices=["STIMULUS", "RANDOM_STIMULUS", "NO_STIMULUS"],
        default="STIMULUS",
        help="Experimental condition (default: STIMULUS)"
    )

    parser.add_argument(
        "--wells",
        type=int,
        nargs='+',
        default=[0],
        help="Well indices to use (default: 0)"
    )

    args = parser.parse_args()

    try:
        run_pong_experiment(
            duration_minutes=args.duration,
            condition=args.condition,
            wells=args.wells
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
