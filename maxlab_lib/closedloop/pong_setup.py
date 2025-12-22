#!/usr/bin/env python3
"""
pong_experiment_setup.py
------------------------

Complete setup script for the Pong closed-loop experiment with MaxLab system.

This script performs:
1. System initialization and calibration (Offset/Gain compensation)
2. Electrode layout configuration (sensory + motor recording, sensory stimulation)
3. Pre-configured stimulation sequences for C++ real-time module
4. Recording control and data export
5. Configuration export for C++ module integration

The C++ module will handle real-time game logic and stimulation delivery.
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
    4885, 4666, 4886, 4022, 5327, 5328, 5106, 5326, 3138, 3140, 2919,
    5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
    3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3142, 3797, 3361,
    4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
    3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
    3800, 3579, 3581,
]
# fmt: on

# Functional electrode groups for Pong experiment
# These will be subdivided from the recording pool
SENSORY_RECORDING_ELECTRODES = RECORDING_ELECTRODES[0:8]   # 8 electrodes for ball position
MOTOR_1_RECORDING_ELECTRODES = RECORDING_ELECTRODES[8:20]  # 12 electrodes for "paddle up"
MOTOR_2_RECORDING_ELECTRODES = RECORDING_ELECTRODES[20:32] # 12 electrodes for "paddle down"

# Stimulation electrodes (subset of sensory electrodes)
# Must be carefully chosen to avoid routing conflicts
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

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

RECORDING_DIR = str(Path.home() / "pong_experiments")
Path(RECORDING_DIR).mkdir(parents=True, exist_ok=True)

# Stimulation parameters for different game events
STIM_PARAMS = {
    "ball_position": {
        "amplitude_mV": 50,          # Sensory feedback for ball position
        "phase_us": 200,             # Biphasic pulse width
        "frequency_min_Hz": 4,       # When ball is far
        "frequency_max_Hz": 40,      # When ball is close
    },
    "hit_feedback": {
        "amplitude_mV": 75,          # Positive feedback on paddle hit
        "phase_us": 200,
        "burst_frequency_Hz": 100,
        "burst_duration_ms": 100,
    },
    "miss_feedback": {
        "amplitude_mV": 150,         # Aversive feedback on miss
        "phase_us": 200,
        "burst_frequency_Hz": 5,
        "burst_duration_ms": 4000,
    }
}

event_counter = 0  # Global event counter


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
# SYSTEM INITIALIZATION (Reusing from stimulate.py)
# ============================================================================

def initialize_system() -> None:
    """Initialize system into a defined state
    
    Reused from stimulate.py example.
    Powers on stimulation units and ensures clean state.
    """
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
# ARRAY CONFIGURATION (Reusing from stimulate.py)
# ============================================================================

def configure_array(electrodes: List[int], stim_electrodes: List[int]) -> mx.Array:
    """Configure array with recording and stimulation electrodes
    
    Reused from stimulate.py example.
    
    Parameters
    ----------
    electrodes : List[int]
        Recording electrodes
    stim_electrodes : List[int]
        Stimulation electrodes
        
    Returns
    -------
    mx.Array
        Configured array (not yet downloaded to hardware)
    """
    print_step_header(2, "CONFIGURING ELECTRODE ARRAY")
    
    print_info(f"Total recording electrodes: {len(electrodes)}")
    print_info(f"Stimulation electrodes: {len(stim_electrodes)}")
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


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Connect stimulation units to electrodes
    
    Reused from stimulate.py example with added validation.
    """
    print_step_header(3, "CONNECTING STIMULATION UNITS")
    
    print_info(f"Attempting to connect {len(stim_electrodes)} electrodes to stimulation units")
    
    stim_units: List[int] = []
    
    for idx, stim_el in enumerate(stim_electrodes, 1):
        print_substep(f"Connecting electrode {stim_el} ({idx}/{len(stim_electrodes)})...")
        
        try:
            array.connect_electrode_to_stimulation(stim_el)
            stim = array.query_stimulation_at_electrode(stim_el)
            
            if len(stim) == 0:
                print_error(f"No stimulation unit available for electrode {stim_el}")
                raise RuntimeError(
                    f"No stimulation channel can connect to electrode: {stim_el}\n"
                    f"Please select a neighboring electrode instead."
                )
            
            stim_unit_int = int(stim)
            
            if stim_unit_int in stim_units:
                print_error(f"Electrode {stim_el} conflicts with existing connection")
                raise RuntimeError(
                    f"Two electrodes connected to the same stim unit {stim_unit_int}.\n"
                    f"This is not allowed. Please select a neighboring electrode of {stim_el}!"
                )
            else:
                stim_units.append(stim_unit_int)
                print_success(f"Electrode {stim_el} → Stim Unit {stim_unit_int}")
        
        except Exception as e:
            print_error(f"Connection failed: {str(e)}")
            raise
    
    print_success(f"All {len(stim_units)} stimulation units connected successfully")
    print_info(f"Stimulation unit IDs: {stim_units}")
    
    return stim_units


def configure_and_powerup_stim_units(stim_units: List[int]) -> None:
    """Power up and configure stimulation units
    
    Reused from stimulate.py example.
    All units use DAC source 0 for synchronized control.
    """
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
                .dac_source(0)  # All use DAC 0 for C++ control
            )
            mx.send(stim)
            print_success(f"Stim Unit {stim_unit} powered up and ready")
        except Exception as e:
            print_error(f"Failed to power up unit {stim_unit}: {str(e)}")
            raise
    
    print_success("All stimulation units powered up successfully")


# ============================================================================
# STIMULATION SEQUENCE PREPARATION
# ============================================================================

def amplitude_mV_to_DAC_bits(amplitude_mV: float) -> int:
    """Convert mV to DAC bits
    
    DAC range: 0-1023 bits
    512 bits = 0V
    Inverting amplifier: 512 - bits = positive voltage
    
    Parameters
    ----------
    amplitude_mV : float
        Desired amplitude in millivolts
        
    Returns
    -------
    int
        DAC bit value
    """
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
    """Create a single biphasic stimulation pulse
    
    Parameters
    ----------
    seq : mx.Sequence
        Sequence to append to
    amplitude_mV : float
        Pulse amplitude in millivolts
    phase_us : int
        Duration of each phase in microseconds
    event_label : str
        Optional label for event tracking
        
    Returns
    -------
    mx.Sequence
        Updated sequence
    """
    global event_counter
    
    # Calculate phase duration in samples (50us per sample)
    phase_samples = int(phase_us / 50)
    
    # Convert amplitude to DAC bits
    amplitude_bits = amplitude_mV_to_DAC_bits(amplitude_mV)
    
    # Add event marker if label provided
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


def prepare_ball_position_sequences() -> Dict[str, mx.Sequence]:
    """Prepare multiple ball position feedback sequences at different frequencies
    
    Creates discrete sequences from 4Hz to 40Hz in 4Hz steps.
    The C++ module will select the appropriate sequence based on ball distance.
    
    Frequency mapping:
    - 4Hz: Ball very far from paddle
    - 8Hz: Ball far
    - 12Hz: Ball moderately far
    - 16Hz: Ball moderate distance
    - 20Hz: Ball moderately close
    - 24Hz: Ball close
    - 28Hz: Ball very close
    - 32Hz: Ball extremely close
    - 36Hz: Ball approaching
    - 40Hz: Ball about to hit
    
    Returns
    -------
    Dict[str, mx.Sequence]
        Dictionary mapping frequency to sequence
        Keys: "ball_4hz", "ball_8hz", ..., "ball_40hz"
    """
    print_substep("Preparing ball position feedback sequences...")
    
    params = STIM_PARAMS["ball_position"]
    sequences = {}
    
    # Generate sequences from 4Hz to 40Hz in 4Hz steps
    frequencies = range(4, 44, 4)  # 4, 8, 12, ..., 40
    
    print_info(f"Generating {len(frequencies)} discrete frequency sequences...")
    
    for freq_hz in frequencies:
        seq = mx.Sequence()
        
        # Create a 1-second burst at this frequency
        # This gives the C++ module a repeatable pattern
        num_pulses = freq_hz  # For 1 second of stimulation
        
        # Inter-pulse interval in samples (20kHz sampling rate)
        # For X Hz, interval = 20000/X samples
        interval_samples = int(20000 / freq_hz)
        
        for pulse_idx in range(num_pulses):
            create_biphasic_pulse(
                seq,
                amplitude_mV=params["amplitude_mV"],
                phase_us=params["phase_us"],
                event_label=f"ball_{freq_hz}hz_pulse_{pulse_idx+1}"
            )
            
            # Add inter-pulse delay (except after last pulse)
            if pulse_idx < num_pulses - 1:
                seq.append(mx.DelaySamples(interval_samples))
        
        # Store sequence with descriptive key
        seq_key = f"ball_{freq_hz}hz"
        sequences[seq_key] = seq
        
        print_success(f"Sequence {freq_hz}Hz ready ({num_pulses} pulses)")
    
    print_success(f"All {len(sequences)} ball position sequences ready")
    print_info(f"Frequency range: 4-40Hz in 4Hz steps")
    print_info(f"Amplitude: {params['amplitude_mV']}mV")
    print_info(f"Phase width: {params['phase_us']}µs")
    
    return sequences


def prepare_hit_feedback_sequence() -> mx.Sequence:
    """Prepare burst sequence for successful paddle hit
    
    Short, high-frequency burst as positive reinforcement.
    
    Returns
    -------
    mx.Sequence
        Burst stimulation sequence
    """
    print_substep("Preparing hit feedback sequence...")
    
    seq = mx.Sequence()
    params = STIM_PARAMS["hit_feedback"]
    
    # Calculate number of pulses in burst
    burst_duration_ms = params["burst_duration_ms"]
    frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * frequency_Hz)
    
    # Inter-pulse interval in samples
    interval_samples = int(20000 / frequency_Hz)  # 20kHz sampling
    
    print_info(f"Generating {num_pulses} pulses...")
    
    for i in range(num_pulses):
        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"hit_feedback_pulse_{i+1}"
        )
        seq.append(mx.DelaySamples(interval_samples))
    
    print_success("Hit feedback sequence ready")
    print_info(f"Burst: {num_pulses} pulses at {frequency_Hz}Hz")
    print_info(f"Amplitude: {params['amplitude_mV']}mV")
    print_info(f"Duration: {burst_duration_ms}ms")
    
    return seq


def prepare_miss_feedback_sequence() -> mx.Sequence:
    """Prepare unpredictable sequence for missed ball
    
    Long, low-frequency, irregular stimulation as negative feedback.
    
    Returns
    -------
    mx.Sequence
        Aversive stimulation sequence
    """
    print_substep("Preparing miss feedback sequence...")
    
    seq = mx.Sequence()
    params = STIM_PARAMS["miss_feedback"]
    
    # Calculate parameters
    burst_duration_ms = params["burst_duration_ms"]
    base_frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * base_frequency_Hz)
    
    print_info(f"Generating {num_pulses} irregular pulses...")
    
    # Add jitter to make it unpredictable
    np.random.seed(42)  # Reproducible randomness
    
    for i in range(num_pulses):
        # Random amplitude variation (±20%)
        amplitude = params["amplitude_mV"] * np.random.uniform(0.8, 1.2)
        
        create_biphasic_pulse(
            seq,
            amplitude_mV=amplitude,
            phase_us=params["phase_us"],
            event_label=f"miss_feedback_pulse_{i+1}"
        )
        
        # Random inter-pulse interval (unpredictable timing)
        base_interval = int(20000 / base_frequency_Hz)
        jitter = int(base_interval * np.random.uniform(0.5, 1.5))
        seq.append(mx.DelaySamples(base_interval + jitter))
    
    print_success("Miss feedback sequence ready")
    print_info(f"Unpredictable burst: {num_pulses} pulses")
    print_info(f"Amplitude: {params['amplitude_mV']}mV ± 20% jitter")
    print_info(f"Duration: {burst_duration_ms}ms")
    
    return seq


def prepare_all_sequences() -> Dict[str, mx.Sequence]:
    """Prepare all stimulation sequences for the experiment
    
    These sequences are pre-loaded and can be triggered by the C++ module.
    
    Returns
    -------
    Dict[str, mx.Sequence]
        Dictionary of named sequences
    """
    print_step_header(8, "PREPARING STIMULATION SEQUENCES")
    
    print_info("Preparing multiple sequence types for C++ module")
    
    # Get ball position sequences (multiple frequencies)
    ball_sequences = prepare_ball_position_sequences()
    
    # Prepare feedback sequences
    print_substep("Preparing hit feedback sequence...")
    hit_sequence = prepare_hit_feedback_sequence()
    
    print_substep("Preparing miss feedback sequence...")
    miss_sequence = prepare_miss_feedback_sequence()
    
    # Combine all sequences into one dictionary
    sequences = ball_sequences.copy()
    sequences["hit_feedback"] = hit_sequence
    sequences["miss_feedback"] = miss_sequence
    
    print_success(f"All stimulation sequences prepared successfully")
    print_info(f"Total sequences: {len(sequences)}")
    print_info(f"  - Ball position: {len(ball_sequences)} frequencies (4-40Hz)")
    print_info(f"  - Hit feedback: 1 sequence")
    print_info(f"  - Miss feedback: 1 sequence")
    
    return sequences


# ============================================================================
# RECORDING CONTROL
# ============================================================================

def start_recording(
    recording_name: str,
    wells: List[int] = [0]
) -> mx.Saving:
    """Start recording data to HDF5 file
    
    Parameters
    ----------
    recording_name : str
        Name of the recording file (without extension)
    wells : List[int]
        List of well indices to record
        
    Returns
    -------
    mx.Saving
        Saving object for controlling recording
    """
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
    print_info(f"Wells: {wells}")
    print_info(f"Channels: 1024")
    print_info(f"Sampling rate: 20 kHz")
    
    return s


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close file
    
    Parameters
    ----------
    saving : mx.Saving
        Saving object to stop
    """
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
    stim_units: List[int],
    sequences: Dict[str, mx.Sequence],  # 新增参数
    config_path: str
) -> Dict:
    """Export configuration for C++ closed-loop module
    
    The C++ module will read this JSON file to know:
    - Which electrodes to monitor for spikes
    - Which stimulation units to control
    - Stimulation parameters
    - Available stimulation sequences
    
    Parameters
    ----------
    array : mx.Array
        Configured array
    stim_units : List[int]
        List of stimulation unit indices
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
        # Get channel mappings from array config
        config = array.get_config()
        
        # Map electrodes to channels
        sensory_channels = config.get_channels_for_electrodes(SENSORY_RECORDING_ELECTRODES)
        motor_1_channels = config.get_channels_for_electrodes(MOTOR_1_RECORDING_ELECTRODES)
        motor_2_channels = config.get_channels_for_electrodes(MOTOR_2_RECORDING_ELECTRODES)
        stim_channels = config.get_channels_for_electrodes(SENSORY_STIM_ELECTRODES)
        
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
    ball_sequences = [k for k in sequences.keys() if k.startswith("ball_")]
    ball_frequencies = sorted([int(k.split("_")[1].replace("hz", "")) for k in ball_sequences])
    
    cpp_config = {
        "experiment": {
            "name": "pong_closed_loop",
            "description": "Real-time Pong game with neuronal control",
            "version": "1.0",
            "timestamp": datetime.now().isoformat()
        },
        
        "electrodes": {
            "sensory_recording": SENSORY_RECORDING_ELECTRODES,
            "motor_1_recording": MOTOR_1_RECORDING_ELECTRODES,
            "motor_2_recording": MOTOR_2_RECORDING_ELECTRODES,
            "sensory_stimulation": SENSORY_STIM_ELECTRODES
        },
        
        "channels": {
            "sensory_channels": sensory_channels,
            "motor_1_channels": motor_1_channels,
            "motor_2_channels": motor_2_channels,
            "stim_channels": stim_channels
        },
        
        "stimulation_units": stim_units,
        
        "stimulation_parameters": STIM_PARAMS,
        
        # New: Sequence mapping for C++ module
        "sequences": {
            "ball_position": {
                "available_frequencies": ball_frequencies,
                "sequence_names": ball_sequences,
                "mapping": {
                    freq: f"ball_{freq}hz" 
                    for freq in ball_frequencies
                }
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
        
        "game_parameters": {
            "ball_speed": 0.5,
            "paddle_speed": 1.0,
            "paddle_height": 0.2,
            "update_rate_ms": 10,
            
            # Ball distance to frequency mapping
            "distance_to_frequency": {
                "description": "Map normalized ball distance [0,1] to stimulation frequency",
                "min_frequency_hz": 4,
                "max_frequency_hz": 40,
                "step_hz": 4,
                "formula": "freq = 4 * round((40 - 36 * distance) / 4)"
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
    
    print_info("Channel mappings summary:")
    print_info(f"  - Sensory (ball position): {sensory_channels}")
    print_info(f"  - Motor 1 (paddle up): {motor_1_channels}")
    print_info(f"  - Motor 2 (paddle down): {motor_2_channels}")
    print_info(f"  - Stimulation: {stim_channels}")
    print_info(f"Stimulation units: {stim_units}")
    print_info(f"Available ball sequences: {ball_sequences}")
    
    print_success("Configuration export complete")
    
    return cpp_config




# ============================================================================
# POST-EXPERIMENT ANALYSIS
# ============================================================================

def analyze_recording(recording_path: str) -> None:
    """Perform basic analysis of recorded data
    
    Parameters
    ----------
    recording_path : str
        Path to the .raw.h5 file
    """
    print_step_header(14, "ANALYZING RECORDED DATA")
    
    print_substep(f"Opening recording file: {recording_path}")
    
    try:
        with File(recording_path, 'r') as f:
            # Get spike data
            well_path = "wells/well000/rec0000"
            traces = f[f"{well_path}/groups/all_channels/raw"]
            events = f[f"{well_path}/events"]
            
            print_success("Recording file opened successfully")
            
            print_info(f"Trace shape: {traces.shape}")
            print_info(f"Number of events: {len(events)}")
            duration = traces.shape[1] / 20000
            print_info(f"Recording duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            # Count events by type
            if len(events) > 0:
                print_substep("Analyzing event types...")
                event_types = {}
                for event in events:
                    event_id = event[2]
                    event_desc = event[3].decode('utf-8') if isinstance(event[3], bytes) else str(event[3])
                    
                    # Extract event type from description
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
    """Run complete Pong experiment
    
    Parameters
    ----------
    duration_minutes : int
        Experiment duration in minutes
    condition : str
        Experimental condition ("STIMULUS", "RANDOM_STIMULUS", "NO_STIMULUS")
    wells : List[int]
        List of well indices to use
    """
    print("\n" + "=" * 70)
    print("PONG CLOSED-LOOP EXPERIMENT")
    print("=" * 70)
    print(f"\nCondition: {condition}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Wells: {wells}")
    
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
    
    array = configure_array(all_recording, SENSORY_STIM_ELECTRODES)
    
    # Step 3: Connect stimulation
    stim_units = connect_stim_units_to_stim_electrodes(SENSORY_STIM_ELECTRODES, array)
    
    # Activate wells
    print_step_header(4, "ACTIVATING WELLS")
    print_substep(f"Activating wells: {wells}")
    try:
        mx.activate(wells)
        print_success(f"Wells {wells} activated")
    except Exception as e:
        print_error(f"Failed to activate wells: {str(e)}")
        raise
    
    # Step 4: Download configuration
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
    
    # Step 5: Calibration
    print_step_header(5, "PERFORMING CALIBRATION")
    print_substep("Running offset compensation (this will take ~15 seconds)...")
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
    
    # Step 6: Power up stimulation units
    configure_and_powerup_stim_units(stim_units)
    
    # Step 7: Clear event buffer
    print_step_header(7, "CLEARING EVENT BUFFER")
    print_substep("Clearing any previous events from buffer...")
    try:
        mx.clear_events()
        print_success("Event buffer cleared")
    except Exception as e:
        print_error(f"Failed to clear events: {str(e)}")
        raise
    
    # Step 8: Prepare stimulation sequences
    sequences = prepare_all_sequences()
    
    # Step 9: Export configuration for C++ module
    cpp_config = export_cpp_config(array, stim_units, str(config_path))
    
    # Step 10: Start recording
    saving = start_recording(session_name, wells)
    
    # Step 11: Baseline recording
    print_step_header(11, "BASELINE RECORDING")
    baseline_duration = 30
    print_substep(f"Recording {baseline_duration}s baseline before experiment...")
    for i in range(baseline_duration):
        if (i + 1) % 10 == 0:
            print_info(f"{baseline_duration - (i + 1)} seconds remaining...")
        time.sleep(1)
    print_success("Baseline recording complete")
    
    # Step 12: C++ module integration point
    print_step_header(12, "C++ MODULE INTEGRATION POINT")
    print_warning("C++ module loading would happen here")
    print_info("Command: mx.load_module('libpong_closed_loop.so', '{config_path}')")
    print_info("\nC++ module responsibilities:")
    print_info("  - Monitor spike events from sensory and motor channels")
    print_info("  - Update game physics at 10ms intervals")
    print_info("  - Trigger pre-configured sequences based on game state")
    print_info("  - Log game events to CSV file")
    
    # Simulate experiment duration
    print_substep(f"Running experiment for {duration_minutes} minutes...")
    print_info("(In production environment, C++ module would be running)")
    
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
    
    # Step 13: Stop recording
    stop_recording(saving)
    
    # Step 14: Analyze results
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
    print_info("\nOutput files:")
    print_info(f"  - Recording: {recording_path}")
    print_info(f"  - Configuration: {config_path}")
    print_info("\nNext steps:")
    print_info("  1. Review recorded data using MaxLab analysis tools")
    print_info("  2. Analyze game performance from C++ module logs")
    print_info("  3. Compare spike activity across experimental conditions")
    print_info("  4. Generate figures and statistics for publication")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaxLab Pong Closed-Loop Experiment Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 20 --condition STIMULUS
  %(prog)s --duration 30 --condition NO_STIMULUS --wells 0 1
  %(prog)s --duration 10 --condition RANDOM_STIMULUS
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
    print("MAXLAB PONG EXPERIMENT - INITIALIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Duration: {args.duration} minutes")
    print(f"  Condition: {args.condition}")
    print(f"  Wells: {args.wells}")
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
        print_warning("Experiment stopped by user (Ctrl+C)")
        return 130
    except Exception as e:
        print("\n\n" + "=" * 70)
        print("EXPERIMENT FAILED")
        print("=" * 70)
        print_error(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())