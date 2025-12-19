#!/usr/bin/python

"""
RECORDINGS EXAMPLE

This script illustrates how basic recordings can be run
via the python API. The following steps are needed for a full
recording:

1. Initialize the recording settings.
2. Prepare the stimulation array (see simulate.py for more
explanation).
3. Define an electrode group for data storage.
4. Begins the recording.
5. Stop the recording and save the file.

For further explanation about specific functions, please have
a look at the API documentation. 

"""

import maxlab as mx
import time

# fmt: off
electrodes = [4885, 4666, 4886, 4022, 5327, 5328, 5106, 5326, 3138, 3140, 2919,
              5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
              3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3142, 3797, 3361,
              4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
              3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
              3800, 3579, 3581,]
# fmt: on

if __name__ == "__main__":
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")
    time.sleep(mx.Timing.waitInit)


    # If we use MaxTwo, recordings from multiple wells can be generated. 
    # In order to do so, we need to specify from which well(s) we wish to 
    # record with the following variable. E.g., if wells 2 and 4, then
    # recording_wells = [2, 4]. For this example, we use well 0 only:
    wells = [0]
    mx.activate(wells)
    dir_name = "/tmp"
    file_name = "test_recording"

    # Prepare the stimulation array
    array = mx.Array("test")
    array.select_electrodes(electrodes)
    array.route()
    array.download()
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)
    mx.clear_events()  # Empty event-buffer before adding anything to it

    s = mx.Saving()
    s.open_directory(dir_name)
    s.start_file(file_name)

    # With the new data format, we must declare which electrodes
    # we want to store data from. This can be set through the group_define
    # function which has the following form:
    # s.group_define(well_nr, "any_name", list_of_channels_to_record_from)
    s.group_define(0, "all_channels", list(range(1024)))

    s.start_recording(wells)
    
    print("Start recording")
    time.sleep(10)
    print("Stop recording")
    
    s.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    s.stop_file()
    s.group_delete_all()
