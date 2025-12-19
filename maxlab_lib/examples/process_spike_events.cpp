#include <iostream>
#include <thread>
#include <chrono>

#include "maxlab/maxlab.h"
#include <unistd.h>

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Call with:\t" << argv[0] << "\t[detection_channel]" << std::endl;
        exit(1);
    }
    const int detection_channel = atoi(argv[1]);

    uint8_t targetWell{0};

    uint64_t blanking{0};
    maxlab::checkVersions();
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));
    std::this_thread::sleep_for(std::chrono::seconds(2));;//Allow data stream to open

    maxlab::FilteredFrameData frameData;
    while (true)
    {
        maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frameData);
        if (status == maxlab::Status::MAXLAB_NO_FRAME)
            continue;

        if (frameData.frameInfo.well_id != targetWell)
            continue;

        if (blanking > 0)
        {
            blanking--;
            continue;
        }

        for (uint64_t i = 0; i < frameData.spikeCount; ++i)
        {
            const maxlab::SpikeEvent & spike = frameData.spikeEvents[i];
            if (spike.channel == detection_channel)
            {
                maxlab::verifyStatus(maxlab::sendSequence("closed_loop"));
                blanking = 8000;
            }
        }
    }
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
}
