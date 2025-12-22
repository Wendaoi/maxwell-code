#ifndef SPIKE_DETECTION_H
#define SPIKE_DETECTION_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "filter.h"

struct SpikeDetectorConfig {
    double sample_rate_hz = 20000.0;
    double highpass_hz = 100.0;
    double lowpass_hz = 1.0;
    double highpass_q = 0.707;
    float threshold_multiplier = 5.0f;
    float min_threshold = -20.0f;
    int refractory_samples = 1000;
};

class SpikeDetector {
public:
    SpikeDetector(std::size_t channel_count, const SpikeDetectorConfig& config);

    void resetFilters();
    void resetCounts();

    void processFrame(const float* samples, std::size_t channel_count);
    void addSpikeEvent(std::uint16_t channel);

    void getCounts(std::vector<std::uint32_t>* out) const;
    std::size_t channelCount() const;

private:
    struct ChannelState {
        SecondOrderHighpassFilter highpass;
        FirstOrderLowpassFilter lowpass;
        float smoothed_abs = 0.0f;
        float threshold = 0.0f;
        float prev_value = 0.0f;
        int samples_since_last_spike = 0;
        std::uint32_t spike_count = 0;

        ChannelState(const SpikeDetectorConfig& config);
    };

    void processSample(ChannelState& state, float value);

    SpikeDetectorConfig config_;
    std::vector<ChannelState> channels_;
};

#endif // SPIKE_DETECTION_H
