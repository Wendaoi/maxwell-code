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

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from h5py import File

# ============================================================================
# ELECTRODE LAYOUT DEFINITION
# ============================================================================

# Recording electrodes pool (reusing from stimulate.py example)
# fmt: off
RECORDING_ELECTRODES = [
    3624, 3648, 3672, 3696, 3748, 5328, 5106, 5326, 3138, 3140, 2919,
    5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
    3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3142, 3797, 3361,
    4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
    3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
    3800, 3579, 3581,
]
# fmt: on

# Functional electrode groups for Pong experiment
SENSORY_RECORDING_ELECTRODES = RECORDING_ELECTRODES[0:8]   # 8 electrodes for ball position
MOTOR_1_RECORDING_ELECTRODES = RECORDING_ELECTRODES[8:20]  # 12 electrodes for "paddle up"
MOTOR_2_RECORDING_ELECTRODES = RECORDING_ELECTRODES[20:32] # 12 electrodes for "paddle down"

# ============================================================================
# STIMULATION POSITION CONFIGURATION (8 positions)
# ============================================================================

# 8 distinct stimulation positions corresponding to 8 ball vertical positions
SENSORY_STIM_ELECTRODES = [
    RECORDING_ELECTRODES[0],  # Position 0 (top)
    RECORDING_ELECTRODES[1],  # Position 1
    RECORDING_ELECTRODES[2],  # Position 2
    RECORDING_ELECTRODES[3],  # Position 3 (middle)
    RECORDING_ELECTRODES[4],  # Position 4
    RECORDING_ELECTRODES[5],  # Position 5
    RECORDING_ELECTRODES[6],  # Position 6
    RECORDING_ELECTRODES[7],  # Position 7 (bottom)
]

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
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system() -> None:
    """Initialize system into a defined state"""
    print_step_header(1, "INITIALIZING MAXLAB SYSTEM")
    
    print_substep("Connecting to MaxLab server...")
    try:
        mx.initialize()
        print_success("Connected to MaxLab server")
    except Exception as e:
        print_error(f"Failed to connect: {str(e)}")
        raise
    
    print_substep("Enabling stimulation power...")
    try:
        result = mx.send(mx.Core().enable_stimulation_power(True))
        if result != "Ok":
            raise RuntimeError(f"Stimulation power enable returned: {result}")
        print_success("Stimulation power enabled")
    except Exception as e:
        print_error(f"Failed to enable stimulation power: {str(e)}")
        raise RuntimeError("The system didn't initialize correctly.")
    
    print_success("System initialization complete")


# ============================================================================
# ARRAY CONFIGURATION
# ============================================================================

def configure_array(
    electrodes: List[int],
    stim_electrodes: List[int],
    stim_position_electrodes: Optional[List[int]] = None,
) -> mx.Array:
    """Configure array with recording and stimulation electrodes"""
    print_step_header(2, "CONFIGURING ELECTRODE ARRAY")
    
    print_info(f"Total recording electrodes: {len(electrodes)}")
    position_count = len(stim_position_electrodes or stim_electrodes)
    print_info(f"Stimulation positions: {position_count}")
    if stim_position_electrodes is not None and len(stim_electrodes) != position_count:
        print_info(f"Stimulation candidates (with neighbors): {len(stim_electrodes)}")
    print_info(f"Sensory recording: {len(SENSORY_RECORDING_ELECTRODES)} electrodes")
    print_info(f"Motor 1 recording: {len(MOTOR_1_RECORDING_ELECTRODES)} electrodes")
    print_info(f"Motor 2 recording: {len(MOTOR_2_RECORDING_ELECTRODES)} electrodes")
    
    print_substep("Creating array configuration...")
    try:
        array = mx.Array("pong_experiment")
        print_success("Array object created")
    except Exception as e:
        print_error(f"Failed to create array: {str(e)}")
        raise
    
    print_substep("Resetting previous configuration...")
    try:
        array.reset()
        array.clear_selected_electrodes()
        print_success("Previous configuration cleared")
    except Exception as e:
        print_error(f"Failed to reset: {str(e)}")
        raise
    
    print_substep("Selecting recording electrodes...")
    try:
        array.select_electrodes(electrodes)
        print_success(f"Selected {len(electrodes)} recording electrodes")
    except Exception as e:
        print_error(f"Failed to select electrodes: {str(e)}")
        raise
    
    print_substep("Selecting stimulation electrodes...")
    try:
        array.select_stimulation_electrodes(stim_electrodes)
        print_success(f"Selected {len(stim_electrodes)} stimulation electrodes")
    except Exception as e:
        print_error(f"Failed to select stimulation electrodes: {str(e)}")
        raise
    
    print_substep("Routing electrodes (this may take 10-30 seconds)...")
    try:
        start_time = time.time()
        array.route()
        elapsed = time.time() - start_time
        print_success(f"Routing completed in {elapsed:.1f} seconds")
    except Exception as e:
        print_error(f"Routing failed: {str(e)}")
        raise
    
    print_success("Array configuration complete")
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
    print_step_header(3, "CONNECTING STIMULATION UNITS")
    
    print_info(f"Connecting {len(stim_electrodes)} positions to stimulation units")
    
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
        print_substep(f"Connecting {position_name} (electrode {stim_el})...")
        
        try:
            selected_electrode = stim_el
            attempted = [stim_el]
            stim_unit_int, reason = try_connect(stim_el)

            if reason:
                print_warning(
                    f"Electrode {stim_el} unavailable or conflicts; trying neighbors"
                )

                for neighbor in iter_neighbor_candidates(stim_el):
                    if neighbor in used_electrodes:
                        continue
                    attempted.append(neighbor)
                    stim_unit_int, reason = try_connect(neighbor)
                    if not reason:
                        selected_electrode = neighbor
                        break

                # If still conflicting, broaden search to any routed candidate electrode
                if reason and candidate_electrodes:
                    print_warning(
                        "No free neighbor; trying additional routed candidates..."
                    )
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
                print_warning(
                    f"{position_name}: using neighbor electrode {selected_electrode} instead of {stim_el}"
                )
            print_success(
                f"{position_name}: Electrode {selected_electrode} -> Unit {stim_unit_int}"
            )
        
        except Exception as e:
            print_error(f"Connection failed: {str(e)}")
            raise
    
    print_success(f"All {len(electrode_to_unit)} stimulation units connected")
    print_info(f"Electrode->Unit mapping: {electrode_to_unit}")
    
    return electrode_to_unit, resolved_stim_electrodes


def configure_and_powerup_stim_units(stim_units: List[int]) -> None:
    """Power up and configure all stimulation units"""
    print_step_header(6, "POWERING UP STIMULATION UNITS")
    
    print_info(f"Configuring {len(stim_units)} stimulation units")
    print_info("Mode: Voltage mode, DAC source 0")
    
    for idx, stim_unit in enumerate(stim_units, 1):
        print_substep(f"Powering up unit {stim_unit} ({idx}/{len(stim_units)})...")
        
        try:
            stim = (
                mx.StimulationUnit(stim_unit)
                .power_up(True)
                .connect(True)
                .set_voltage_mode()
                .dac_source(0)
            )
            mx.send(stim)
            print_success(f"Unit {stim_unit} ready")
        except Exception as e:
            print_error(f"Failed to power up unit {stim_unit}: {str(e)}")
            raise
    
    print_success("All stimulation units powered up")


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
    print_step_header(8, "PREPARING DECOUPLED POSITION × FREQUENCY SEQUENCES")
    
    params = STIM_PARAMS["ball_position"]
    frequencies = params["frequencies_Hz"]
    
    total_sequences = len(stim_electrodes) * len(frequencies)
    print_info(f"Generating {total_sequences} sequences:")
    print_info(f"  - {len(stim_electrodes)} positions × {len(frequencies)} frequencies")
    
    sequences = {}
    all_unit_ids = list(electrode_to_unit.values())
    
    sequence_count = 0
    
    for pos_idx, (electrode, position_name) in enumerate(zip(stim_electrodes, position_names)):
        target_unit = electrode_to_unit[electrode]
        
        print_substep(f"Generating sequences for {position_name} (unit {target_unit})...")
        
        for freq_hz in frequencies:
            sequence_count += 1
            
            # Create unique sequence name
            seq_name = f"{position_name}_{freq_hz}hz"
            
            # Create persistent sequence
            seq = mx.Sequence(name=seq_name, persistent=True)
            
            # Step 1: Configure stimulation unit routing
            # Deactivate all other units, activate only this position
            unit_config_commands = create_unit_configuration_commands(
                target_unit_id=target_unit,
                all_unit_ids=all_unit_ids
            )
            
            for cmd in unit_config_commands:
                seq.append(cmd)
            
            # Add small delay to ensure routing takes effect
            seq.append(mx.DelaySamples(100))  # 5ms
            
            # Step 2: Generate stimulation pulses
            num_pulses = freq_hz  # 1 second burst
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
            
            # Send sequence to mxwserver
            seq.send()
            sequences[seq_name] = seq
            
            if sequence_count % 10 == 0:
                print_info(f"Progress: {sequence_count}/{total_sequences} sequences generated")
    
    print_success(f"All {len(sequences)} sequences generated and uploaded")
    print_info(f"Sequence naming: <position>_<frequency>hz")
    print_info(f"Example: pos0_20hz, pos3_40hz")
    print_info(f"Amplitude: {params['amplitude_mV']}mV")
    print_info(f"Phase width: {params['phase_us']}µs")
    
    return sequences


def prepare_hit_feedback_sequence() -> mx.Sequence:
    """Prepare burst sequence for successful paddle hit"""
    print_substep("Preparing hit feedback sequence...")
    
    seq = mx.Sequence(name="hit_feedback", persistent=True)
    params = STIM_PARAMS["hit_feedback"]
    
    burst_duration_ms = params["burst_duration_ms"]
    frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * frequency_Hz)
    interval_samples = int(20000 / frequency_Hz)
    
    print_info(f"Generating {num_pulses} pulses...")
    
    for i in range(num_pulses):
        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"hit_feedback_pulse_{i+1}"
        )
        seq.append(mx.DelaySamples(interval_samples))
    
    seq.send()
    
    print_success("Hit feedback sequence ready")
    print_info(f"Burst: {num_pulses} pulses at {frequency_Hz}Hz")
    
    return seq


def prepare_miss_feedback_sequence() -> mx.Sequence:
    """Prepare unpredictable sequence for missed ball"""
    print_substep("Preparing miss feedback sequence...")
    
    seq = mx.Sequence(name="miss_feedback", persistent=True)
    params = STIM_PARAMS["miss_feedback"]
    
    burst_duration_ms = params["burst_duration_ms"]
    base_frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * base_frequency_Hz)
    
    print_info(f"Generating {num_pulses} irregular pulses...")
    
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
    
    print_success("Miss feedback sequence ready")
    
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
    print_step_header(8, "PREPARING ALL STIMULATION SEQUENCES")
    
    print_info("Architecture: Fully decoupled position × frequency")
    position_count = len(position_names)
    frequency_count = len(STIM_PARAMS["ball_position"]["frequencies_Hz"])
    ball_sequence_count = position_count * frequency_count
    print_info(
        f"  - {position_count} positions × {frequency_count} frequencies = {ball_sequence_count} ball sequences"
    )
    print_info(f"  - 2 feedback sequences (hit/miss)")
    print_info(f"  - Total: {ball_sequence_count + 2} sequences")
    
    # Prepare decoupled ball position sequences
    ball_sequences = prepare_decoupled_ball_sequences(
        electrode_to_unit, stim_electrodes, position_names
    )
    
    # Prepare feedback sequences
    hit_sequence = prepare_hit_feedback_sequence()
    miss_sequence = prepare_miss_feedback_sequence()
    
    # Combine all sequences
    sequences = ball_sequences.copy()
    sequences["hit_feedback"] = hit_sequence
    sequences["miss_feedback"] = miss_sequence
    
    print_success(f"All {len(sequences)} sequences prepared and uploaded to mxwserver")
    print_info("Sequences are persistent and ready for C++ module triggering")
    
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
    
    print_substep("Creating recording file...")
    print_info(f"File path: {recording_path}")
    
    try:
        s = mx.Saving()
        s.open_directory(RECORDING_DIR)
        s.start_file(recording_name)
        print_success("Recording file created")
    except Exception as e:
        print_error(f"Failed to create file: {str(e)}")
        raise
    
    print_substep("Defining recording channels...")
    try:
        s.group_define(0, "all_channels", list(range(1024)))
        print_success("All 1024 channels configured")
    except Exception as e:
        print_error(f"Failed to define channels: {str(e)}")
        raise
    
    print_substep(f"Starting recording on wells {wells}...")
    try:
        s.start_recording(wells)
        print_success("Recording started successfully")
    except Exception as e:
        print_error(f"Failed to start recording: {str(e)}")
        raise
    
    print_success("Data recording active")
    
    return s


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close file"""
    print_step_header(13, "STOPPING DATA RECORDING")
    
    print_substep("Stopping recording...")
    try:
        saving.stop_recording()
        print_success("Recording stopped")
    except Exception as e:
        print_error(f"Failed to stop recording: {str(e)}")
        raise
    
    print_substep("Closing file...")
    try:
        saving.stop_file()
        saving.group_delete_all()
        print_success("File closed")
    except Exception as e:
        print_error(f"Failed to close file: {str(e)}")
        raise
    
    print_substep(f"Waiting {mx.Timing.waitAfterRecording}s for file finalization...")
    time.sleep(mx.Timing.waitAfterRecording)
    
    print_success("Recording stopped and file closed successfully")


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
    print_step_header(9, "EXPORTING CONFIGURATION FOR C++ MODULE")
    
    print_substep("Retrieving channel mappings from array...")
    
    try:
        config = array.get_config()
        
        sensory_channels = config.get_channels_for_electrodes(SENSORY_RECORDING_ELECTRODES)
        motor_1_channels = config.get_channels_for_electrodes(MOTOR_1_RECORDING_ELECTRODES)
        motor_2_channels = config.get_channels_for_electrodes(MOTOR_2_RECORDING_ELECTRODES)
        stim_channels = config.get_channels_for_electrodes(stim_electrodes)
        
        print_success("Channel mappings retrieved")
        print_info(f"Sensory channels: {len(sensory_channels)}")
        print_info(f"Motor 1 channels: {len(motor_1_channels)}")
        print_info(f"Motor 2 channels: {len(motor_2_channels)}")
        print_info(f"Stimulation channels: {len(stim_channels)}")
        
    except Exception as e:
        print_error(f"Failed to retrieve channel mappings: {str(e)}")
        raise
    
    print_substep("Building configuration structure...")
    
    # Extract ball position sequence names
    ball_sequences = [k for k in sequences.keys() if any(pos in k for pos in position_names)]
    frequencies = STIM_PARAMS["ball_position"]["frequencies_Hz"]
    
    # Create position->unit mapping
    position_to_unit = {
        position_names[idx]: electrode_to_unit[electrode]
        for idx, electrode in enumerate(stim_electrodes)
    }
    
    # Create sequence lookup table for C++
    sequence_lookup = {}
    for pos_name in position_names:
        sequence_lookup[pos_name] = {}
        for freq in frequencies:
            seq_name = f"{pos_name}_{freq}hz"
            sequence_lookup[pos_name][freq] = seq_name
    
    cpp_config = {
        "experiment": {
            "name": "pong_closed_loop_decoupled",
            "description": "Pong with 8 positions × 10 frequencies (80 sequences)",
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "architecture": "Fully decoupled position × frequency"
        },
        
        "electrodes": {
            "sensory_recording": SENSORY_RECORDING_ELECTRODES,
            "motor_1_recording": MOTOR_1_RECORDING_ELECTRODES,
            "motor_2_recording": MOTOR_2_RECORDING_ELECTRODES,
            "sensory_stimulation": stim_electrodes
        },
        
        "channels": {
            "sensory_channels": sensory_channels,
            "motor_1_channels": motor_1_channels,
            "motor_2_channels": motor_2_channels,
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
                "usage_example": "pos3_mid_20hz triggers 20Hz at position 3"
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
                "example_1": 'maxlab::sendSequence("pos0_top_40hz")',
                "example_2": 'maxlab::sendSequence("pos7_bottom_4hz")',
                "example_3": 'maxlab::sendSequence("hit_feedback")'
            },
            "sequence_selection": {
                "description": "Map ball position and distance to sequence name",
                "position_index": "0-7 (top to bottom)",
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
                "description": "Normalize ball Y position [0,1] to position index [0,7]",
                "formula": "pos_idx = int(ball_y * 7.99)"
            },
            "distance_to_frequency": {
                "description": "Map ball distance [0,1] to frequency index [0,9]",
                "formula": "freq_idx = int((1.0 - distance) * 9.99)"
            }
        },
        
        "spike_detection": {
            "threshold_std": 5.0,
            "refractory_period_ms": 2.0
        }
    }
    
    print_success("Configuration structure built")
    
    print_substep(f"Writing configuration to JSON file...")
    
    try:
        config_file = Path(config_path)
        with open(config_file, 'w') as f:
            json.dump(cpp_config, f, indent=2)
        print_success(f"Configuration exported to: {config_file}")
    except Exception as e:
        print_error(f"Failed to write config file: {str(e)}")
        raise
    
    print_info("Decoupled architecture summary:")
    print_info(f"  - Positions: {len(position_names)}")
    print_info(f"  - Frequencies: {len(frequencies)}")
    print_info(f"  - Total ball sequences: {len(ball_sequences)}")
    print_info(f"  - Position->Unit mapping: {position_to_unit}")
    
    print_success("Configuration export complete")
    
    return cpp_config


# ============================================================================
# POST-EXPERIMENT ANALYSIS
# ============================================================================

def analyze_recording(recording_path: str) -> None:
    """Perform basic analysis of recorded data"""
    print_step_header(14, "ANALYZING RECORDED DATA")
    
    print_substep(f"Opening recording file: {recording_path}")
    
    try:
        with File(recording_path, 'r') as f:
            well_path = "wells/well000/rec0000"
            traces = f[f"{well_path}/groups/all_channels/raw"]
            events = f[f"{well_path}/events"]
            
            print_success("Recording file opened successfully")
            
            print_info(f"Trace shape: {traces.shape}")
            print_info(f"Number of events: {len(events)}")
            duration = traces.shape[1] / 20000
            print_info(f"Recording duration: {duration:.2f} seconds")
            
            if len(events) > 0:
                print_substep("Analyzing event types...")
                event_types = {}
                for event in events:
                    event_desc = event[3].decode('utf-8') if isinstance(event[3], bytes) else str(event[3])
                    event_type = event_desc.split('_')[0] if '_' in event_desc else event_desc
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                print_success("Event analysis complete:")
                for event_type, count in sorted(event_types.items()):
                    print_info(f"  - {event_type}: {count} events")
            else:
                print_warning("No events recorded")
    
    except Exception as e:
        print_error(f"Failed to analyze recording: {str(e)}")
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
    print(f"\nCondition: {condition}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Wells: {wells}")
    print(f"Architecture: 8 positions × 10 frequencies = 80 sequences")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"pong_{condition}_{timestamp}"
    config_path = Path(RECORDING_DIR) / f"{session_name}_config.json"
    
    print(f"Session name: {session_name}")
    print(f"Output directory: {RECORDING_DIR}")
    
    # Step 1: Initialize system
    initialize_system()
    print_substep(f"Waiting {mx.Timing.waitInit}s for system stabilization...")
    time.sleep(mx.Timing.waitInit)
    print_success("System ready")
    
    # Step 2: Configure array
    all_recording = list(set(
        SENSORY_RECORDING_ELECTRODES +
        MOTOR_1_RECORDING_ELECTRODES +
        MOTOR_2_RECORDING_ELECTRODES
    ))
    
    stim_candidates = build_stim_candidate_electrodes(SENSORY_STIM_ELECTRODES)
    array = configure_array(
        all_recording,
        stim_candidates,
        stim_position_electrodes=SENSORY_STIM_ELECTRODES,
    )
    
    # Step 3: Connect stimulation (returns electrode->unit mapping)
    electrode_to_unit, resolved_stim_electrodes = connect_stim_units_to_stim_electrodes(
        SENSORY_STIM_ELECTRODES,
        array,
        candidate_electrodes=stim_candidates,
        neighbor_search_radius=STIM_NEIGHBOR_SEARCH_RADIUS,
    )
    
    # Step 4: Activate wells
    print_step_header(4, "ACTIVATING WELLS")
    print_substep(f"Activating wells: {wells}")
    try:
        mx.activate(wells)
        print_success(f"Wells {wells} activated")
    except Exception as e:
        print_error(f"Failed to activate wells: {str(e)}")
        raise
    
    # Step 5: Download configuration
    print_step_header(5, "DOWNLOADING CONFIGURATION TO HARDWARE")
    print_substep("Transferring array configuration to chip...")
    try:
        array.download(wells)
        print_success("Configuration downloaded to hardware")
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        raise
    
    print_substep(f"Waiting {mx.Timing.waitAfterDownload}s for hardware to stabilize...")
    time.sleep(mx.Timing.waitAfterDownload)
    print_success("Hardware ready")
    
    # Step 6: Calibration
    print_step_header(5, "PERFORMING CALIBRATION")
    print_substep("Running offset compensation...")
    try:
        mx.offset()
        print_success("Offset compensation initiated")
    except Exception as e:
        print_error(f"Offset compensation failed: {str(e)}")
        raise
    
    print_substep("Waiting for calibration to complete...")
    for i in range(15):
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print_info(f"{15 - (i + 1)} seconds remaining...")
    print_success("Calibration complete")
    
    # Step 7: Power up stimulation units
    all_units = list(electrode_to_unit.values())
    configure_and_powerup_stim_units(all_units)
    
    # Step 8: Clear event buffer
    print_step_header(7, "CLEARING EVENT BUFFER")
    print_substep("Clearing any previous events from buffer...")
    try:
        mx.clear_events()
        print_success("Event buffer cleared")
    except Exception as e:
        print_error(f"Failed to clear events: {str(e)}")
        raise
    
    # Step 9: Prepare all decoupled sequences
    sequences = prepare_all_sequences(
        electrode_to_unit, resolved_stim_electrodes, POSITION_NAMES
    )
    
    # Step 10: Export configuration for C++
    cpp_config = export_cpp_config(
        array,
        electrode_to_unit,
        resolved_stim_electrodes,
        POSITION_NAMES,
        sequences,
        str(config_path),
    )
    
    # Step 11: Start recording
    saving = start_recording(session_name, wells)
    
    # Step 12: Baseline recording
    print_step_header(11, "BASELINE RECORDING")
    baseline_duration = 30
    print_substep(f"Recording {baseline_duration}s baseline...")
    for i in range(baseline_duration):
        if (i + 1) % 10 == 0:
            print_info(f"{baseline_duration - (i + 1)} seconds remaining...")
        time.sleep(1)
    print_success("Baseline recording complete")
    
    # Step 13: C++ module integration
    print_step_header(12, "C++ MODULE INTEGRATION POINT")
    print_warning("C++ module would be loaded here")
    print_info(f"Command: mx.load_module('libpong_closed_loop.so', '{config_path}')")
    print_info("\nC++ usage example:")
    print_info('  maxlab::sendSequence("pos3_mid_20hz");  // Trigger 20Hz at middle position')
    print_info('  maxlab::sendSequence("pos0_top_40hz");   // Trigger 40Hz at top')
    print_info('  maxlab::sendSequence("hit_feedback");    // Positive feedback')
    
    print_substep(f"Running experiment for {duration_minutes} minutes...")
    total_seconds = duration_minutes * 60
    last_update = 0
    
    for i in range(total_seconds):
        current_minute = i // 60
        if current_minute != last_update:
            minutes_remaining = (total_seconds - i) // 60
            print_info(f"Time remaining: {minutes_remaining} minutes...")
            last_update = current_minute
        time.sleep(1)
    
    print_success("Experiment duration complete")
    
    # Step 14: Stop recording
    stop_recording(saving)
    
    # Step 15: Analyze results
    recording_path = Path(RECORDING_DIR) / f"{session_name}.raw.h5"
    if recording_path.exists():
        analyze_recording(str(recording_path))
    else:
        print_warning(f"Recording file not found: {recording_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print_success("All steps completed successfully!")
    print_info("\nDecoupled architecture summary:")
    print_info(f"  - Total sequences: {len(sequences)}")
    print_info(f"  - Ball sequences: 80 (8 positions × 10 frequencies)")
    print_info(f"  - Feedback sequences: 2 (hit/miss)")
    print_info("\nOutput files:")
    print_info(f"  - Recording: {recording_path}")
    print_info(f"  - Configuration: {config_path}")


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
    
    print("\n" + "=" * 70)
    print("MAXLAB PONG EXPERIMENT - DECOUPLED ARCHITECTURE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Duration: {args.duration} minutes")
    print(f"  Condition: {args.condition}")
    print(f"  Wells: {args.wells}")
    print(f"  Architecture: 8 positions × 10 frequencies")
    print("\nStarting experiment...")
    
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
