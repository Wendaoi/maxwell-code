#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <chrono>
#include "maxlab/maxlab.h"

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Call with: %s [detection_channel]", argv[0]);
        exit(1);
    }
    const int detection_channel = atoi(argv[1]);

    maxlab::checkVersions();
    maxlab::verifyStatus(maxlab::DataStreamerRaw_open());
    std::this_thread::sleep_for(std::chrono::seconds(2));//Allow data stream to open

    uint64_t blanking = 0;
    while (true)
    {
        // fprintf(stderr, "Get into this loop");
        maxlab::RawFrameData frameData;
        maxlab::Status status = maxlab::DataStreamerRaw_receiveNextFrame(&frameData);
        if (status == maxlab::Status::MAXLAB_NO_FRAME || frameData.frameInfo.corrupted)
            continue;
        if (blanking > 0)
        {
            blanking--;
            if(blanking != 0)
                continue;
        }

        if (frameData.amplitudes[detection_channel] > 40.f) //Amplitudes can be variable. Adjust this as necessary.
        {
            maxlab::verifyStatus(maxlab::sendSequence("closed_loop"));
            blanking = 8000;
        }
    }
    maxlab::verifyStatus(maxlab::DataStreamerRaw_close());
}
