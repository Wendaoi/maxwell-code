#!/usr/bin/python

"""
STIMULATION EXAMPLES

This script can be used to run stimulation on a MaxTwo or
MaxOne system via the Python API. Several combinations of pulse
trains can be run and the script can be called from the command
line directly. 

Warning: No recordings are made in this example, but you can find
an example about how to record in another script, called `recordings.py`.
"""

import time

import maxlab as mx

from typing import List, Optional
from pathlib import Path

# fmt: off
electrodes = [4885, 4666, 4886, 4022, 5327, 5328, 5106, 5326, 3138, 3140, 2919,
              5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
              3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3142, 3797, 3361,
              4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
              3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
              3800, 3579, 3581,]
# fmt: on
event_counter = 1  # variable to keep track of the event_id


def initialize_system() -> None:
    """Initialize system into a defined state

    The function initializes the system into a defined state before
    starting any script. This way, one can be sure that the system
    is always in the same state while running the script, regardless
    of what has been done before with it. The function also powers on
    the stimulation units, which are turned off by default.

    Raises
    ------
    RuntimeError
        If system does not initialize correctly.

    """
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")


def configure_array(electrodes: List[int], stim_electrodes: List[int]) -> mx.Array:
    """Configure array

    This function configures the array, given a list of recording
    and stimulation electrodes. An important step in this function
    is the routing, which happens once the electrodes are selected.

    Parameters
    ----------
    electrodes : List[int]
        List of the index of the recording electrodes
    stim_electrodes : List[int]
        List of the index of the stimulation electrodes

    Returns
    -------
    mx.Array
        The configured array.

    Notes
    -----
    To finalize the configuration, the array needs to be downloaded to
    the MaxOne/Two. However, before being able to do that, we first need
    to connect the stimulation units to the stimulation electrodes with the
    function `connect_stim_units_to_stim_electrodes`. This is the reason
    why `array.download` is not part of this function.

    """
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    array.route()
    return array


def load_config(config_file: str) -> mx.Array:
    """Load a previously created configuration

    Load and configure an array from a previously generated
    configuration, either through the Python API or through
    the Scope GUI.

    Parameters
    ----------
    config_file : str
        Name of the previously generated config file.

    Returns
    -------
    mx.Array
        The configured array.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    Exception
        If the config_file cannot be loaded properly.

    """
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    array = mx.Array("stimulation")
    try:
        array.load_config(config_file)
    except Exception as e:
        raise Exception(f"Error loading config file '{config_file}': {str(e)}")
    return array


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Connect the stimulation units to the stimulation electrodes

    Once an array configuration has been obtained, either through routing
    or through loading a previous configuration, the stimulation units
    can be connected to the desired electrodes.

    Notes
    -----
    With this step, one needs to be careful, in rare cases it can happen
    that an electrode cannot be stimulated. For example, the electrode
    could not be routed (due to routing constraints), and the error message
    "No stimulation channel can connect to electrode: ..." will be printed.
    If this situation occurs, it is recommended to then select the electrode
    next to it.

    Parameters
    ----------
    stim_electrodes : List[int]
        List of the index of the stimulation electrodes
    array : mx.Array
        The configured array

    Returns
    -------
    List[str]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Raises
    ------
    RuntimeError
        If an electrode cannot be connected to a stimulation unit.
        If two electrodes are connected to the same stimulation unit.
    """
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(
                f"No stimulation channel can connect to electrode: {str(stim_el)}"
            )
        stim_unit_int = int(stim)
        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit.\
                               This is not allowed. Please Select a neighboring electrode of {stim_el}!"
            )
        else:
            stim_units.append(stim_unit_int)
    return stim_units


def powerup_stim_unit(stim_unit: int) -> mx.StimulationUnit:
    """Power up and connect a stimulation unit

    This function powers up and connect a specific stimulation
    unit in the MaxOne/Two. Without it, we would not be able to
    send the prepared sequence to the device and it thus needs
    to be run before running `mx.send(stim)` (as shown in the
    function `configure_and_powerup_stim_units).

    Parameters
    ----------
    stim_unit : str
        The index of the stimulation unit to power up

    Returns
    -------
    mx.StimulationUnit
        The powered up stimulation unit

    """
    return (
        mx.StimulationUnit(stim_unit)
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(0)
    )


def configure_and_powerup_stim_units(stim_units: List[int]) -> List[mx.StimulationUnit]:
    """Configure and powerup the stimulation units

    Once the electrodes are connected to the stimulation units,
    the stimulation units need to be configured and powererd up.

    Parameters
    ----------
    stim_units : List[str]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Returns
    -------
    List[mx.StimulationUnit]
        List of stimulation unit objects corresponding to the connected
        stimulation electrodes.

    """
    stim_unit_commands: List[mx.StimulationUnit] = []
    for stim_unit in stim_units:
        stim = powerup_stim_unit(stim_unit)
        stim_unit_commands.append(stim)
        mx.send(stim)
    return stim_unit_commands


def poweroff_all_stim_units() -> None:
    """Poweroff all stimulation units

    This function is used to make sure that every stimulation units is
    powered-off before starting sequentially to send the sequences to the
    different stimulation units individually.

    Returns
    -------
    None

    """
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)


def create_stim_pulse(
    seq: mx.Sequence, amplitude: int, delay_samples: int
) -> mx.Sequence:
    """Create stimulation pulse

    The stimulation units can be controlled through three independent
    sources, what we call DAC channels (for digital analog converter).
    By programming a DAC channel with digital values, we can control
    the output the stimulation units. DAC inputs are in the range between
    0 to 1023 bits, whereas 512 corresponds to zero volt and one bit
    corresponds to 2.9 mV. When the stimulation buffers are set to voltage
    mode, they act like an inverting amplifier, hence 512 + a number of bits corresponds to a negative voltage, or 512 - a number of bits corresponds to a positive voltage.
    corresponds to a negative voltage. Thus, to give a pulse of 100mV,
    the DAC channel temporarily would need to be set to 512 - 34 (100mV/2.9)
    and back again to 512.

    Notes
    -----
    In this example, all 32 units are controlled through the same DAC
    channel ( dac_source(0) ), thus by programming a biphasic pulse
    on DAC channel 0, all the stimulation units exhibit the biphasic
    pulse.

    Parameters
    ----------
    seq : mx.Sequence
        Sequence object holding a sequence of commands, as generated
        by `mx.Sequence()`.
    amplitude : int
        Amplitude of the pulse, with units [100mV/2.9], as explained above.
    delay_samples : int
        How many samples should sand between different sequence amplitude.

    Returns
    -------
    mx.Sequence
        Sequence object filled by the pulse.

    """
    global event_counter
    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, f"amplitude {amplitude} event_id {event_counter}"))
    seq.append(mx.DAC(0, 512 - amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512))
    return seq


def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    phase: int,
    amplitude: int,
    changing_amplitude: Optional[bool] = False,
    max_amplitude: Optional[int] = None,
    amplitude_interval: Optional[int] = None,
) -> mx.Sequence:
    """Prepare a stimulation sequence.

    This is just and example and it only illustrates how sequences
    can be constructed.

    Parameters
    ----------
    number_pulses_per_train : int
        Number of repetitions of one pulse.
    inter_pulse_interval : int
        Number of samples to delay between two consecutive pulses.
    phase : int
        Number of samples before the switch from high-low (and vice versa)
        voltage when creating stimulation pulse.
    amplitude : int
        Amplitude of the pulse (minimal if changing_amplitude is True).
        Unit is millivolt.
    changing_amplitude : Optional[bool]
        Whether the pulse amplitude changes, by default False.
    max_amplitude : Optional[int]
        Maximal amplitude of the pulse if changing_amplitude is True.
        Unit is millivolt.
    amplitude_interval : Optional[int]
        Increment amplitude interval if changing_amplitude is True.
        Unit is millivolt.

    Returns
    -------
    mx.Sequence
        Sequence object filled with the stimulation sequence.

    """
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    if changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError(
                "Both max_amplitude and amplitude_interval are required for changing_amplitude."
            )
        for cur_amplitude in range(amplitude, max_amplitude, amplitude_interval):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(seq, int(cur_amplitude/dac_lsb_mV), phase)
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(seq, int(amplitude/dac_lsb_mV), phase)
            seq.append(mx.DelaySamples(inter_pulse_interval))
    return seq


def send_stim_pulses_all_units(seq: mx.Sequence, number_pulse_trains: int) -> None:
    """Send stimulation pulses to all units simultaneously

    This function sends the sequence of pulses built by the
    function `prepare_stim_sequence` (the pulse train) for the
    case where the stimulation pulses are sent simultaneously to all units.

    Parameters
    ----------
    number_pulse_trains : int
        The number of repetitions of the pulse sequence.
    seq : mx.Sequence
        The sequence of pulses

    Returns
    -------
    None

    """
    for _ in range(number_pulse_trains):
        print("Send pulse")
        seq.send()
        time.sleep(10)


def send_stim_pulses_units_sequentially(
    seq: mx.Sequence, stim_units: List[int]
) -> None:
    """Send stimulation pulses to units sequentially

    This function sends the sequence of pulses built by the
    function `prepare_stim_sequence` for the case where there
    stimulation pulses are sent sequentially to one unit at a time.

    Parameters
    ----------
    seq : mx.Sequence
        The sequence of pulses
    stim_units : List[int]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Returns
    -------
    None

    """
    for stim_unit in stim_units:
        print(f"Power up stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        mx.send(stim)
        print("Send pulse")
        seq.send()
        print(f"Power down stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit).power_up(False)
        mx.send(stim)
        time.sleep(2)


if __name__ == "__main__":
    # The fist thing we do is initalize the system.
    initialize_system()

    # We have selected some recording electrodes (see at the top,
    # directly after the imports) and here we write down some
    # stimulation electrodes.
    # Note that these choices follow certain rules, i.e.
    #   * You cannot choose more than 32 stimulation electrodes.
    #   * You cannot choose more than 1024 recording electrodes.
    #   * You cannot choose two stimulation electrodes that get connected
    # to the same stimulation unit.
    # The above selected recording electrodes are just an example and
    # should showcase how one can select specific electrodes on the array.
    stim_electrodes = [3580, 4887]
    array = configure_array(electrodes, stim_electrodes)
    # If instead we use a pre-defined config, we can load it like this:
    # array = load_config("path/to/config/file.cfg")

    # Select the subset of wells we want to stimulate.
    wells = list(range(1))
    mx.activate(wells)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
    array.download(wells)
    # Wait a few seconds to make sure the configuration is downloaded
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    # Wait a few more seconds to make sure the offset compensation is done
    time.sleep(15)

    mx.clear_events()  # Empty event-buffer before adding anything to it

    simultaneous_and_same_seq = True
    if simultaneous_and_same_seq:
        stim_unit_commands = configure_and_powerup_stim_units(stim_units)

        # At this stage, we can start the recordings. Note that, if we wish to have events written
        # to the recorded file, it is important to construct our sequence after starting the recording.
        seq = prepare_stim_sequence(
            number_pulses_per_train=10,
            inter_pulse_interval=2000,  # samples
            phase=4,  # 4 samples = 200us (4*50us)
            amplitude=200,  # mV
        )

        # Note that since the loop to send the pulse trains is done in python,
        # the timing between the pulse trains won't be particularly precise.
        # Instead consider createing one large sequence (although there are
        # some limitations for MaxOne here as well, see documentation).
        send_stim_pulses_all_units(seq, number_pulse_trains=5)
    else:
        poweroff_all_stim_units()

        # Start potential recordings.
        seq = prepare_stim_sequence(
            number_pulses_per_train=10,
            inter_pulse_interval=2000,  # samples
            phase=4,  # 4 samples = 200us (4*50us)
            amplitude=200,  # mV
            changing_amplitude=True,
            max_amplitude=400,  # mV
            amplitude_interval=50,  # mV
        )
        send_stim_pulses_units_sequentially(seq, stim_units)

    # If one wishes to send 2 sequences (pulse trains) randomly, run for example:
    #
    # seq_1 = prepare_stim_sequence(number_pulses_per_train, inter_pulse_interval, phase, amplitude=200)
    # seq_2 = prepare_stim_sequence(number_pulses_per_train, inter_pulse_interval, phase, amplitude=150)

    # The pulse trains can be delivered whenever seems reasonable, or following
    # a specific stimulation schedudle, such as:

    # time.sleep(1)
    # send_stim_pulses_all_units(number_pulse_trains, seq_1)
    # time.sleep(5)
    # send_stim_pulses_all_units(number_pulse_trains, seq_2)
