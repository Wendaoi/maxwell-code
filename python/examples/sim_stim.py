"""
sim_stim.py
-------------

This script delivers two stimulation pulses to two electrodes
simultaneously by using two independent stimulation units. The
stimulation units are controlled simultaneously by two different
DACs. This is achieved by using a special DAC mode, as demonstrated
below.


        200us
    <----------->
    .-----------.                     ^
    |           |                     |
____|           |           .____     -   30 mV    stimulation unit A
                |           |
                |           |
                `-----------'



    .-----------.                     ^
    |           |                     |
    |           |                     |
____|           |           .____     -   100 mV   stimulation unit B
                |           |
                |           |
                `-----------'



----|-----------|-----------|---
    t1          t2          t3

"""

import maxlab as mx
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from time import sleep
from h5py import File
from pathlib import Path

# Recordings will be saved in the home direcotry.
recording_dir = str(Path.home())

# File name of the recording
file_name = "sim_stim"

# Stimulation amplitude in mV
amplitude_mV_A = 30  # mV
amplitude_mV_B = 100  # mV

# Width of one stimulation phase, in 50us samples
phase = 4  # 4 * 50us = 200 us

# Number of stimulation pulses
pulse_count = 3

# Time between two consecutive pulses, in 50us samples
inter_pulse_interval = 10000  # 10000 * 50us = 0.5 s

# A total of 58 electrodes will be selected for recording
# fmt: off
electrodes = [4885, 4666, 4886, 4022, 5327, 5328, 5106, 5326, 3138, 3140, 2919,
              5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
              3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3143, 3797, 3361,
              4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
              3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
              3800, 3579, 3581,]
# fmt: on

# This number can be obtained by calling `mx.query_DAC_lsb_mV()`.
LSB_mV = 2.981901


def amplitude_mV_to_DAC_bits(amplitude_mV: float) -> int:
    """Amplitude parameters for the stimulation

    The amplitude is specified in bits and the range is from 0 to 1023.
    A bit value of 512 corresponds to 0 mV. To convert from a desired
    stimulation amplitude given in millivolts to bits, the following
    formula is used:

    amplitude_bits = stimulation_voltage / float(mx.query_DAC_lsb_mV())

    The stimulation units (when used in voltage mode) employ an
    inverting amplifier to deliver the pulses, therefore one needs to
    subtract the amplitude in bits from 512 to get a positive stimulation
    pulse. For example, a 100mV pulse would be:

    amplitude = 512 - (100 / float(mx.query_DAC_lsb_mV()))
              = 512 - (100 / 2.981901) = 478

    Conversely, a -100mV pulse would be:
    amplitude = 512 + (100 / 2.981901) = 546

    Parameters
    ----------
    amplitude_mV : float
        The desired amplitude in mV for the voltage stimulation unit

    Returns
    -------
    int
        The amplitude in bits to program the DAC with

    Raises
    ------
    ValueError
        If the desired amplitude exceeds 1.0V
    """
    if abs(amplitude_mV) > 1000.0:
        raise ValueError(f"The desired amplitude is too large as it exceeds 1.0V")
    return int(512 - (amplitude_mV / LSB_mV))


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Connect the stimulation units to the stimulation electrodes

    Once an array configuration has been obtained, either through routing
    or through loading a previous configuration, the stimulation units
    can be connected to the desired electrodes.

    Notes
    -----
    In rare cases it can happen that an electrode cannot be stimulated,
    for example due to routing constraints. In such situations an error
    message "No stimulation channel can connect to electrode: ..." will
    be printed.  If this happens, it is recommended to select another
    electrode, for example the one next to it.

    Parameters
    ----------
    stim_electrodes : List[int]
        List of the index of the stimulation electrodes
    array : mx.Array
        The configured array

    Returns
    -------
    List[str]
        List of stimulation unit indices which were connected to
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
                f"Two electrodes connected to the same stim unit. This is not allowed. Please Select a neighboring electrode of {stim_el}!"
            )
        else:
            stim_units.append(stim_unit_int)
    return stim_units


if __name__ == "__main__":
    # Initialize system and enable stimulation
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")
    sleep(mx.Timing.waitInit)

    # The next step is to select two stimulation electrodes.
    # They must be contained in the `electrodes` list.
    # The following steps then configure the array, i.e.
    #  - connect recording electrodes
    #  - connect stimulation electrodes
    #  - 'Route' the configuration.
    stimulation_electrodes = [electrodes[20], electrodes[30]]
    array = mx.Array()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stimulation_electrodes)

    array.route()

    # Select for which wells the above will apply. If you have an MX1, the well needs to be [0].
    wells = [0]
    mx.activate(wells)

    stimulation_units = connect_stim_units_to_stim_electrodes(
        stimulation_electrodes, array
    )

    # Turn on the two stimulation units and connect them to the array.
    # The important part here is the choice of DAC. Since we want to
    # stimulate simultaneously with two different amplitudes, two DACs
    # are needed: dac_source(0) and dac_source(1).
    mx.send(
        mx.StimulationUnit(stimulation_units[0])
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(0)
    )
    mx.send(
        mx.StimulationUnit(stimulation_units[1])
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(1)
    )

    # All previous settings of the array were not yet applied to the
    # actual hardware. So far, they are just stored in memory.
    # Next step is to download the configuration to the hardware.
    array.download()

    # Wait a few seconds to make sure the configuration is downloaded
    sleep(mx.Timing.waitAfterDownload)

    # Perform offset compensation
    mx.offset()

    # Wait a few more seconds to make sure the offset compensation is done
    sleep(15)

    # Clear out any previously stored `mx.Events` events in the `mxwserver`
    mx.clear_events()

    # Prepare the stimulation sequence
    sequence = mx.Sequence()
    for stimulation_pulse in range(pulse_count):
        # Three stimulation pulses are created as part of the sequence.
        # When controlling the DACs, there are four DAC modes available,
        # which is set by the first argument of the DAC command.
        # This is further described in the documentation.
        # Type help(mx.DAC) to learn more.
        #
        # The four modes are:
        # 0: controls DAC 0
        # 1: controls DAC 1
        # 2: controls DAC 2
        # 3: controls DAC 0 and DAC 1 (at the very same time)
        #
        # The parameters of the mx.DAC() command are the following:
        # 1: DAC mode
        # 2: amplitude in bits for DAC 0 or 1 or 2
        # 3: amplitude in bits for DAC 1
        # Only when the DAC mode is equal to 3, the third argument is considered
        #
        # Insert an event into the data stream, to record the exact
        # timing of the stimulation pulse.
        sequence.append(
            mx.Event(
                0,
                1,
                stimulation_pulse + 1,
                f"stimulation pulses: {amplitude_mV_A}mV, {amplitude_mV_B}mV",
            )
        )

        # Apply the positive phase stimulation pulse (t1)
        sequence.append(
            mx.DAC(
                3,
                amplitude_mV_to_DAC_bits(amplitude_mV_A),
                amplitude_mV_to_DAC_bits(amplitude_mV_B),
            )
        )
        # Wait for 'phase' samples
        sequence.append(mx.DelaySamples(phase))

        # Apply the negative phase stimulation pulse (t2)
        sequence.append(
            mx.DAC(
                3,
                amplitude_mV_to_DAC_bits(-1 * amplitude_mV_A),
                amplitude_mV_to_DAC_bits(-1 * amplitude_mV_B),
            )
        )
        # Wait for 'phase' samples
        sequence.append(mx.DelaySamples(phase))

        # Reset the stimulation units output back to 0V (t3)
        sequence.append(
            mx.DAC(3, amplitude_mV_to_DAC_bits(0), amplitude_mV_to_DAC_bits(0))
        )
        # Wait between two consecutive pulses
        sequence.append(mx.DelaySamples(inter_pulse_interval))

    # START EXPERIMENT

    # Create a file to save the recording data
    s = mx.Saving()

    # Set the recording directory
    s.open_directory(recording_dir)

    # Start a recording file
    s.start_file(file_name)

    # Each recording file needs to have at least one group defined. The
    # third parameter lists the channels which should be contained in the
    # group
    s.group_define(0, "all_channels", list(range(1024)))

    # Start the actual recording
    s.start_recording(wells)

    print("Start recording")

    # Wait some time before applying the stimulation pulses
    sleep(2)

    # Deliver the stimulation pulses
    sequence.send()

    # Wait some time after applying the stimulation pulses
    sleep(2)

    print("Stop recording")

    # Stop the recording and the file
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()

    # Wait few seconds for the file to be closed
    sleep(mx.Timing.waitAfterRecording)

    # Optional: remove the sequence from the `mxwserver` and python
    del sequence

    # Here we create one plot for each pulse to illustrate the results
    config = array.get_config()
    stimulation_channels = config.get_channels_for_electrodes(stimulation_electrodes)
    stimulation_channels.sort()

    # Extract the recorded data from the file
    file = File(f"{recording_dir}/{file_name}.raw.h5")
    traces = file["wells/well000/rec0000/groups/all_channels/raw"]
    raw_trace_A = traces[stimulation_channels[0], :].T
    raw_trace_B = traces[stimulation_channels[1], :].T
    events = file["wells/well000/rec0000/events"]

    # Plot the stimulation pulses
    fig, axs = plt.subplots(
        1,
        pulse_count,
        figsize=(14, 4),
        dpi=100,
        sharey=True,
        facecolor="w",
        edgecolor="k",
    )

    first_event = events[0][0]
    first_fno = file["/wells/well000/rec0000/groups/all_channels/frame_nos"][0]

    for i, event in enumerate(events[:]):
        # Compute frame number of first event
        event_time = event[0] - first_fno

        # Plot the raw traces
        axs[i].plot(
            np.linspace(0, len(raw_trace_A) / 20, len(raw_trace_B)), raw_trace_A
        )
        axs[i].plot(
            np.linspace(0, len(raw_trace_B) / 20, len(raw_trace_B)), raw_trace_B
        )

        # Adjust the x-axis limits
        axs[i].set_xlim((event_time / 20 - 5, event_time / 20 + 5))

        # Plot the time of the stimulation event
        axs[i].vlines([event_time], 1050, 1100)

        # Add axis labels
        axs[i].set_ylabel("bits")
        axs[i].set_xlabel("ms")

    plt.show()
