#!/usr/bin/python

import maxlab as mx

import time

import os

## Setup script to prepare the configurations for the close-loop module
#
# 0. Initialize system
#
# 1. Load a previously created configuration
#
# 2. Connect two electrodes to stimulation units and power up stimulation units
#
#    In rare cases it can happen that the selected electrode cannot be
#    stimulated. So, always check whether the electrode got properly
#    connected. As it is done in this example.  If the electrode is not
#    properly connected, the stimulation has no effect.
#
# 3. Prepare two different sequences of pulse trains
#
#    An almost infinite amount (only limited by the computer memory) of
#    independent stimulation pulse trains can be prepared without actually
#    deliver them yet.
#
# 4. Deliver the two pulse trains randomly
#
#    The previously prepared pulse trains can be delivered whenever
#    seems reasonable, or following a specific stimulation schedule.
#


name_of_configuration = "closeLoop.cfg"
trigger_electrode = 13248
closed_loop_electrode = 13378
trigger_stimulation_amplitude = 10 #in bits
closed_loop_stimulation_amplitude = 10 #in bits
data_directory = "."

######################################################################
# 0. Initialize system
######################################################################

# The next four lines remove the two sequences, in case they were
# already defined in the server. We need to clear them here,
# otherwise the close_loop stimulation would also trigger, when
# we reconfigure the chip, due to the artifacts.
s = mx.Sequence("trigger", persistent=False)
del s
s = mx.Sequence("closed_loop", persistent=False)
del s

# Normal initialization of the chip
mx.initialize()
time.sleep(mx.Timing.waitInit)
mx.send(mx.Core().enable_stimulation_power(True))
mx.send(mx.Amplifier().set_gain(512))


# For the purpose of this example, we set the spike detection
# threshold a bit higher, to avoid/reduce false positives.
# Set detection threshold to 8.5x std of noise
mx.set_event_threshold(8.5)


######################################################################
# 1. Load a previously created configuration
######################################################################

array = mx.Array("stimulation")
array.load_config(name_of_configuration)

######################################################################
# 2. Connect two electrodes to stimulation units
#    and power up the stimulation units
######################################################################

array.connect_electrode_to_stimulation(trigger_electrode)
array.connect_electrode_to_stimulation(closed_loop_electrode)

stimulation1 = array.query_stimulation_at_electrode(trigger_electrode)
stimulation2 = array.query_stimulation_at_electrode(closed_loop_electrode)

if not stimulation1:
    print("Error: electrode: " + str(trigger_electrode) + " cannot be stimulated")

if not stimulation2:
    print("Error: electrode: " + str(closed_loop_electrode) + " cannot be stimulated")

# Download the prepared array configuration to the chip
array.download()
time.sleep(mx.Timing.waitAfterDownload)
mx.offset()
time.sleep(mx.Timing.waitInMX1Offset + mx.Timing.waitAfterOffset)  # Needs to be adjusted for MaxTwo


# Prepare commands to power up and power down the two stimulation units
# The two stim units are controlled by different DACs
cmd_power_stim1 = mx.StimulationUnit(stimulation1).power_up(True).connect(True).set_voltage_mode().dac_source(0)
cmd_power_stim2 = mx.StimulationUnit(stimulation2).power_up(True).connect(True).set_voltage_mode().dac_source(1)

# Power up stim units
mx.send(cmd_power_stim1)
mx.send(cmd_power_stim2)

# Use the electrode next to the stimulation electrode for trigger detection
amp = array.query_amplifier_at_electrode(trigger_electrode + 1)
print("This amplifier channel is connected to the electrode next to the first stimulation electrode: " + amp)
print("Use this channel to detect (simulated) spikes in the C++ closed loop application")


######################################################################
# 3. Prepare two different sequences of pulse trains
######################################################################


def create_stim_pulse(seq: mx.Sequence, amplitude: int, phase: int, dac: int) -> mx.Sequence:
    seq.append(mx.DAC(dac, 512 - amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac, 512 + amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac, 512))
    return seq


######################################################################
# Deliver 10 stimulation pulses to the trigger electrode
######################################################################

print("Start delivering stimulation pulses to the trigger electrode")

s = mx.Saving()
s.open_directory(data_directory)

s.group_delete_all()
s.group_define(0, "routed")

time.sleep(2)
print("Start saving to file")
s.start_file("close_loop_test")
s.start_recording([0])


# Prepare one pulse called 'trigger', we use this to simulate a spike
# on one of the channels by applying an electrical stimulation pulse 
# using DAC 0
sequence_1 = mx.Sequence('trigger', persistent=True)
sequence_1.append(mx.Event(0, 1, 1, "stim trigger"))
sequence_1 = create_stim_pulse(sequence_1, trigger_stimulation_amplitude, 4, 0)

# Create another pulse called 'closed_loop' to stimulate the second electrode
# through DAC 1. This sequence needs to be prepared here in python, but it 
# will be triggered through the 'closed_loop' token in the C++ application
sequence_2 = mx.Sequence('closed_loop', persistent=True)
sequence_2.append(mx.Event(0, 1, 2, "stim closed_loop"))
sequence_2 = create_stim_pulse(sequence_2, closed_loop_stimulation_amplitude, 4, 1)

time.sleep(2)
for rep in range(10):
    sequence_1.send()
    time.sleep(1.5)

print("Stop saving to file")
s.stop_recording()
time.sleep(mx.Timing.waitAfterRecording)
s.stop_file()