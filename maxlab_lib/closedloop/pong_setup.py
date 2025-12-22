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
# SYSTEM INITIALIZATION (Reusing from stimulate.py)
# ============================================================================

def initialize_system() -> None:
    """Initialize system into a defined state
    
    Reused from stimulate.py example.
    Powers on stimulation units and ensures clean state.
    """
    print("=" * 60)
    print("INITIALIZING MAXLAB SYSTEM")
    print("=" * 60)
    
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")
    
    print("✓ System initialized")
    print("✓ Stimulation power enabled")


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
    print("\nConfiguring electrode array...")
    print(f"  - Recording electrodes: {len(electrodes)}")
    print(f"  - Stimulation electrodes: {len(stim_electrodes)}")
    
    array = mx.Array("pong_experiment")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    
    print("  - Routing electrodes (this may take a moment)...")
    array.route()
    
    print("✓ Array configured successfully")
    return array


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Connect stimulation units to electrodes
    
    Reused from stimulate.py example with added validation.
    """
    print("\nConnecting stimulation units to electrodes...")
    stim_units: List[int] = []
    
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        
        if len(stim) == 0:
            raise RuntimeError(
                f"No stimulation channel can connect to electrode: {stim_el}\n"
                f"Please select a neighboring electrode instead."
            )
        
        stim_unit_int = int(stim)
        
        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit {stim_unit_int}.\n"
                f"This is not allowed. Please select a neighboring electrode of {stim_el}!"
            )
        else:
            stim_units.append(stim_unit_int)
            print(f"  ✓ Electrode {stim_el} → Stim Unit {stim_unit_int}")
    
    return stim_units


def configure_and_powerup_stim_units(stim_units: List[int]) -> None:
    """Power up and configure stimulation units
    
    Reused from stimulate.py example.
    All units use DAC source 0 for synchronized control.
    """
    print("\nPowering up stimulation units...")
    
    for stim_unit in stim_units:
        stim = (
            mx.StimulationUnit(stim_unit)
            .power_up(True)
            .connect(True)
            .set_voltage_mode()
            .dac_source(0)  # All use DAC 0 for C++ control
        )
        mx.send(stim)
        print(f"  ✓ Stim Unit {stim_unit} powered up")


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


def prepare_ball_position_sequence() -> mx.Sequence:
    """Prepare a single pulse for ball position feedback
    
    This sequence represents ONE pulse at a given position.
    The C++ module will trigger this at varying frequencies.
    
    Returns
    -------
    mx.Sequence
        Single biphasic pulse sequence
    """
    seq = mx.Sequence()
    params = STIM_PARAMS["ball_position"]
    
    create_biphasic_pulse(
        seq,
        amplitude_mV=params["amplitude_mV"],
        phase_us=params["phase_us"],
        event_label="ball_position_pulse"
    )
    
    return seq


def prepare_hit_feedback_sequence() -> mx.Sequence:
    """Prepare burst sequence for successful paddle hit
    
    Short, high-frequency burst as positive reinforcement.
    
    Returns
    -------
    mx.Sequence
        Burst stimulation sequence
    """
    seq = mx.Sequence()
    params = STIM_PARAMS["hit_feedback"]
    
    # Calculate number of pulses in burst
    burst_duration_ms = params["burst_duration_ms"]
    frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * frequency_Hz)
    
    # Inter-pulse interval in samples
    interval_samples = int(20000 / frequency_Hz)  # 20kHz sampling
    
    for i in range(num_pulses):
        create_biphasic_pulse(
            seq,
            amplitude_mV=params["amplitude_mV"],
            phase_us=params["phase_us"],
            event_label=f"hit_feedback_pulse_{i+1}"
        )
        seq.append(mx.DelaySamples(interval_samples))
    
    return seq


def prepare_miss_feedback_sequence() -> mx.Sequence:
    """Prepare unpredictable sequence for missed ball
    
    Long, low-frequency, irregular stimulation as negative feedback.
    
    Returns
    -------
    mx.Sequence
        Aversive stimulation sequence
    """
    seq = mx.Sequence()
    params = STIM_PARAMS["miss_feedback"]
    
    # Calculate parameters
    burst_duration_ms = params["burst_duration_ms"]
    base_frequency_Hz = params["burst_frequency_Hz"]
    num_pulses = int((burst_duration_ms / 1000.0) * base_frequency_Hz)
    
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
    
    return seq


def prepare_all_sequences() -> Dict[str, mx.Sequence]:
    """Prepare all stimulation sequences for the experiment
    
    These sequences are pre-loaded and can be triggered by the C++ module.
    
    Returns
    -------
    Dict[str, mx.Sequence]
        Dictionary of named sequences
    """
    print("\n" + "=" * 60)
    print("PREPARING STIMULATION SEQUENCES")
    print("=" * 60)
    
    sequences = {
        "ball_position": prepare_ball_position_sequence(),
        "hit_feedback": prepare_hit_feedback_sequence(),
        "miss_feedback": prepare_miss_feedback_sequence(),
    }
    
    print("\n✓ Ball position feedback sequence ready")
    print(f"  - Single pulse: {STIM_PARAMS['ball_position']['amplitude_mV']}mV")
    print(f"  - Phase width: {STIM_PARAMS['ball_position']['phase_us']}µs")
    
    print("\n✓ Hit feedback sequence ready")
    params = STIM_PARAMS["hit_feedback"]
    num_pulses = int((params["burst_duration_ms"] / 1000.0) * params["burst_frequency_Hz"])
    print(f"  - Burst: {num_pulses} pulses at {params['burst_frequency_Hz']}Hz")
    print(f"  - Amplitude: {params['amplitude_mV']}mV")
    print(f"  - Duration: {params['burst_duration_ms']}ms")
    
    print("\n✓ Miss feedback sequence ready")
    params = STIM_PARAMS["miss_feedback"]
    num_pulses = int((params["burst_duration_ms"] / 1000.0) * params["burst_frequency_Hz"])
    print(f"  - Unpredictable burst: {num_pulses} pulses")
    print(f"  - Amplitude: {params['amplitude_mV']}mV (with jitter)")
    print(f"  - Duration: {params['burst_duration_ms']}ms")
    
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
    print("\n" + "=" * 60)
    print("STARTING DATA RECORDING")
    print("=" * 60)
    
    s = mx.Saving()
    s.open_directory(RECORDING_DIR)
    s.start_file(recording_name)
    
    # Define recording group with all 1024 channels
    s.group_define(0, "all_channels", list(range(1024)))
    
    # Start recording
    s.start_recording(wells)
    
    recording_path = Path(RECORDING_DIR) / f"{recording_name}.raw.h5"
    print(f"\n✓ Recording started")
    print(f"  - File: {recording_path}")
    print(f"  - Wells: {wells}")
    print(f"  - Channels: 1024")
    
    return s


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close file
    
    Parameters
    ----------
    saving : mx.Saving
        Saving object to stop
    """
    print("\n" + "=" * 60)
    print("STOPPING DATA RECORDING")
    print("=" * 60)
    
    saving.stop_recording()
    saving.stop_file()
    saving.group_delete_all()
    
    # Wait for file to be properly closed
    time.sleep(mx.Timing.waitAfterRecording)
    
    print("✓ Recording stopped and file closed")


# ============================================================================
# CONFIGURATION EXPORT FOR C++ MODULE
# ============================================================================

def export_cpp_config(
    array: mx.Array,
    stim_units: List[int],
    config_path: str
) -> Dict:
    """Export configuration for C++ closed-loop module
    
    The C++ module will read this JSON file to know:
    - Which electrodes to monitor for spikes
    - Which stimulation units to control
    - Stimulation parameters
    
    Parameters
    ----------
    array : mx.Array
        Configured array
    stim_units : List[int]
        List of stimulation unit indices
    config_path : str
        Path to save JSON config
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("EXPORTING CONFIGURATION FOR C++ MODULE")
    print("=" * 60)
    
    # Get channel mappings from array config
    config = array.get_config()
    
    # Map electrodes to channels
    sensory_channels = config.get_channels_for_electrodes(SENSORY_RECORDING_ELECTRODES)
    motor_1_channels = config.get_channels_for_electrodes(MOTOR_1_RECORDING_ELECTRODES)
    motor_2_channels = config.get_channels_for_electrodes(MOTOR_2_RECORDING_ELECTRODES)
    stim_channels = config.get_channels_for_electrodes(SENSORY_STIM_ELECTRODES)
    
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
        
        "game_parameters": {
            "ball_speed": 0.5,
            "paddle_speed": 1.0,
            "paddle_height": 0.2,
            "update_rate_ms": 10
        },
        
        "spike_detection": {
            "threshold_std": 5.0,
            "refractory_period_ms": 2.0
        }
    }
    
    # Save to JSON
    config_file = Path(config_path)
    with open(config_file, 'w') as f:
        json.dump(cpp_config, f, indent=2)
    
    print(f"\n✓ Configuration exported to: {config_file}")
    print("\nChannel mappings:")
    print(f"  - Sensory (ball position): {len(sensory_channels)} channels")
    print(f"  - Motor 1 (paddle up): {len(motor_1_channels)} channels")
    print(f"  - Motor 2 (paddle down): {len(motor_2_channels)} channels")
    print(f"  - Stimulation: {len(stim_channels)} channels")
    print(f"\nStimulation units: {stim_units}")
    
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
    print("\n" + "=" * 60)
    print("ANALYZING RECORDED DATA")
    print("=" * 60)
    
    with File(recording_path, 'r') as f:
        # Get spike data
        well_path = "wells/well000/rec0000"
        traces = f[f"{well_path}/groups/all_channels/raw"]
        events = f[f"{well_path}/events"]
        
        print(f"\n✓ Recording file opened: {recording_path}")
        print(f"  - Trace shape: {traces.shape}")
        print(f"  - Number of events: {len(events)}")
        print(f"  - Recording duration: {traces.shape[1] / 20000:.2f} seconds")
        
        # Count events by type
        if len(events) > 0:
            print("\nEvent summary:")
            event_types = {}
            for event in events:
                event_id = event[2]
                event_desc = event[3].decode('utf-8') if isinstance(event[3], bytes) else str(event[3])
                
                # Extract event type from description
                event_type = event_desc.split('_')[0] if '_' in event_desc else event_desc
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            for event_type, count in sorted(event_types.items()):
                print(f"  - {event_type}: {count} events")


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
    print("\n" + "=" * 60)
    print("PONG CLOSED-LOOP EXPERIMENT")
    print("=" * 60)
    print(f"\nCondition: {condition}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Wells: {wells}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"pong_{condition}_{timestamp}"
    config_path = Path(RECORDING_DIR) / f"{session_name}_config.json"
    
    # Step 1: Initialize system
    initialize_system()
    time.sleep(mx.Timing.waitInit)
    
    # Step 2: Configure array
    all_recording = list(set(
        SENSORY_RECORDING_ELECTRODES +
        MOTOR_1_RECORDING_ELECTRODES +
        MOTOR_2_RECORDING_ELECTRODES
    ))
    
    array = configure_array(all_recording, SENSORY_STIM_ELECTRODES)
    
    # Step 3: Activate wells and connect stimulation
    mx.activate(wells)
    stim_units = connect_stim_units_to_stim_electrodes(SENSORY_STIM_ELECTRODES, array)
    
    # Step 4: Download configuration to hardware
    print("\nDownloading configuration to hardware...")
    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)
    print("✓ Configuration downloaded")
    
    # Step 5: Calibration
    print("\nPerforming offset compensation...")
    mx.offset()
    time.sleep(15)
    print("✓ Offset compensation complete")
    
    # Step 6: Power up stimulation units
    configure_and_powerup_stim_units(stim_units)
    
    # Step 7: Clear event buffer
    mx.clear_events()
    
    # Step 8: Prepare stimulation sequences
    sequences = prepare_all_sequences()
    
    # Step 9: Export configuration for C++ module
    cpp_config = export_cpp_config(array, stim_units, str(config_path))
    
    # Step 10: Start recording
    saving = start_recording(session_name, wells)
    
    # Step 11: Wait for baseline recording
    print("\nRecording 30s baseline before starting C++ module...")
    time.sleep(30)
    
    # Step 12: C++ module would be loaded here
    print("\n" + "=" * 60)
    print("C++ MODULE INTEGRATION POINT")
    print("=" * 60)
    print("\nAt this point, you would load the C++ closed-loop module:")
    print(f"  mx.load_module('libpong_closed_loop.so', '{config_path}')")
    print("\nThe C++ module will:")
    print("  - Monitor spike events from sensory and motor channels")
    print("  - Update game physics at 10ms intervals")
    print("  - Trigger pre-configured sequences based on game state")
    print("  - Log game events to CSV file")
    
    # For now, simulate experiment duration
    print(f"\nRunning experiment for {duration_minutes} minutes...")
    print("(In production, C++ module runs here)")
    
    # Simple progress indicator
    total_seconds = duration_minutes * 60
    for i in range(total_seconds):
        if i % 60 == 0:
            minutes_remaining = (total_seconds - i) // 60
            print(f"  {minutes_remaining} minutes remaining...")
        time.sleep(1)
    
    # Step 13: Stop recording
    print("\nExperiment duration complete")
    stop_recording(saving)
    
    # Step 14: Analyze results
    recording_path = Path(RECORDING_DIR) / f"{session_name}.raw.h5"
    if recording_path.exists():
        analyze_recording(str(recording_path))
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Recording: {recording_path}")
    print(f"  - Configuration: {config_path}")
    print(f"\nNext steps:")
    print("  1. Review recorded data")
    print("  2. Analyze game performance from C++ logs")
    print("  3. Compare spike activity across conditions")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaxLab Pong Closed-Loop Experiment Setup"
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
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())