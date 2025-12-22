#include "spike_detection.h"

#include <algorithm>
#include <cmath>

SpikeDetector::ChannelState::ChannelState(const SpikeDetectorConfig& config)
    : highpass(config.highpass_hz, config.highpass_q, config.sample_rate_hz),
      lowpass(config.lowpass_hz, config.sample_rate_hz),
      smoothed_abs(0.0f),
      threshold(config.min_threshold),
      prev_value(0.0f),
      samples_since_last_spike(0),
      spike_count(0) {}

SpikeDetector::SpikeDetector(std::size_t channel_count, const SpikeDetectorConfig& config)
    : config_(config) {
    channels_.reserve(channel_count);
    for (std::size_t i = 0; i < channel_count; ++i) {
        channels_.emplace_back(config_);
    }
}

void SpikeDetector::resetFilters() {
    for (auto& channel : channels_) {
        channel.highpass.reset();
        channel.lowpass.reset();
        channel.smoothed_abs = 0.0f;
        channel.threshold = config_.min_threshold;
        channel.prev_value = 0.0f;
        channel.samples_since_last_spike = 0;
    }
}

void SpikeDetector::resetCounts() {
    for (auto& channel : channels_) {
        channel.spike_count = 0;
    }
}

void SpikeDetector::processFrame(const float* samples, std::size_t channel_count) {
    const std::size_t count = (std::min)(channel_count, channels_.size());
    for (std::size_t i = 0; i < count; ++i) {
        processSample(channels_[i], samples[i]);
    }
}

void SpikeDetector::addSpikeEvent(std::uint16_t channel) {
    if (channel >= channels_.size()) return;
    channels_[channel].spike_count++;
}

void SpikeDetector::getCounts(std::vector<std::uint32_t>* out) const {
    if (!out) return;
    out->assign(channels_.size(), 0);
    for (std::size_t i = 0; i < channels_.size(); ++i) {
        (*out)[i] = channels_[i].spike_count;
    }
}

std::size_t SpikeDetector::channelCount() const {
    return channels_.size();
}

void SpikeDetector::processSample(ChannelState& state, float value) {
    const float filtered = state.highpass.filterOne(value);
    const float abs_value = std::fabs(filtered);
    state.smoothed_abs = state.lowpass.filterOne(abs_value);

    state.threshold = (std::min)(config_.min_threshold,
                                 -state.smoothed_abs * config_.threshold_multiplier);

    bool spike = false;
    if (state.samples_since_last_spike >= config_.refractory_samples) {
        if (state.prev_value > state.threshold && filtered <= state.threshold) {
            spike = true;
        }
    }

    if (spike) {
        state.spike_count++;
        state.samples_since_last_spike = 0;
    } else {
        state.samples_since_last_spike++;
    }

    state.prev_value = filtered;
}
