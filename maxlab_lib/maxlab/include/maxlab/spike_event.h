/**
 * @file spike_event.hpp
 */

#pragma once

#include <cstdint>

namespace maxlab
{
/**
* @brief Structure used for encoding of a spike occurrence.
* This structure stores the frame number at which a spike is detected
* (typically at the maximal value of the spike), the amplitude of that spike
* and the channel at which this spike was detected.
*/
struct SpikeEvent
{
    unsigned long frameNo = 0; ///< 8 bytes - Frame number (time point) at which spike is detected.
    float amp = 0;             ///< 4 bytes - Amplitude of the detected spike.
    uint16_t channel = 0;      ///< 2 bytes - Channel number at which spike is detected.
    unsigned char wellId = 0;  ///< 1 byte  - Well id where this spike originates.
};
}