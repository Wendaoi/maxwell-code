#!/bin/python3
import time
import random
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

maxlab.util.activate([0,1,2,3,4,5])
maxlab.util.initialize()
maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
time.sleep(2)

#electrodes = random.sample(range(0, 26400), 1024)
#stimulation_electrodes = random.sample(electrodes, 2)

electrodes = range(1000, 1100)
stimulation_electrodes = [1050]

array = maxlab.chip.Array('stimulation')
array.reset()
array.clear_selected_electrodes( )
array.select_electrodes( electrodes )
array.select_stimulation_electrodes( stimulation_electrodes )
array.route()
stimulation_units = []

for stim_el in stimulation_electrodes:
    array.connect_electrode_to_stimulation( stim_el )
    stim = array.query_stimulation_at_electrode( stim_el )
    print(array.query_amplifier_at_electrode(stim_el))
    if stim:
        stimulation_units.append( stim )
    else:
        print("No stimulation channel can connect to electrode: " + str(stim_el))

maxlab.send(maxlab.chip.Amplifier().set_gain(512))
array.download()
time.sleep(5)

maxlab.send_raw('stream_reset_command_counter')

#maxlab.util.offset()
#time.sleep(10)

# First, poweroff all the stimulation units
for stimulation_unit in range(0,32):
    # Stimulation Unit
    stim = maxlab.chip.StimulationUnit(stimulation_unit)
    stim.power_up(False)
    stim.connect(False)
    maxlab.send(stim)

for stimulation_unit in stimulation_units:
    print("Power up stimulation unit " + str(stimulation_unit))
    stim = maxlab.chip.StimulationUnit(stimulation_unit)
    stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
    maxlab.send(stim)
    time.sleep(2)
    print("Send pulse")

    pulse_counter = 0
    for amp in [1]:
        maxlab.send_raw('system_loop_prepare')
        for _ in range(6):
            #maxlab.send_raw('system_loop_set_name %s' % str(pulse_counter))
            pulse_counter += 1
            maxlab.send_raw('system_loop_append_dac 0 %d' % (512-amp))
            maxlab.send_raw('system_loop_append_delay 20')
            maxlab.send_raw('system_loop_append_dac 0 %d' % (512+amp))
            maxlab.send_raw('system_loop_append_delay 20')
            maxlab.send_raw('system_loop_append_dac 0 512')
            maxlab.send_raw('system_loop_append_delay 500')

            #time.sleep(2)
            #maxlab.send_raw("maxtwo_command SEQ_PAUSE 0.05")
        maxlab.send_raw('system_loop_finish')
        maxlab.send_raw('system_loop_run_once')

    time.sleep(10)
    print("Power down stimulation unit " + str(stimulation_unit))
    stim = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False)
    maxlab.send(stim)
    time.sleep(2)
