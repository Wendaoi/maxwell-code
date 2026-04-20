#include "spike_detection.h"

#include <algorithm>
#include <cmath>

SpikeDetector::ChannelState::ChannelState(const SpikeDetectorConfig& config)
    : median_lowpass(config.median_lowpass_hz, config.sample_rate_hz),
      highpass(config.highpass_hz, config.highpass_q, config.sample_rate_hz),
      mad_lowpass(config.mad_lowpass_hz, config.sample_rate_hz),
      median_estimate(0.0f),
      activity(0.0f),
      absolute_deviation(0.0f),
      mad_estimate(0.0f),
      threshold(0.0f),
      prev_activity(0.0f),
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
        channel.median_lowpass.reset();
        channel.highpass.reset();
        channel.mad_lowpass.reset();
        channel.median_estimate = 0.0f;
        channel.activity = 0.0f;
        channel.absolute_deviation = 0.0f;
        channel.mad_estimate = 0.0f;
        channel.threshold = 0.0f;
        channel.prev_activity = 0.0f;
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

SpikeDetector::ChannelDebugState SpikeDetector::debugState(std::size_t channel) const {
    if (channel >= channels_.size()) {
        return {};
    }

    const ChannelState& state = channels_[channel];
    ChannelDebugState debug;
    debug.median_estimate = state.median_estimate;
    debug.activity = state.activity;
    debug.mad_estimate = state.mad_estimate;
    debug.threshold = state.threshold;
    return debug;
}

std::size_t SpikeDetector::channelCount() const {
    return channels_.size();
}

void SpikeDetector::processSample(ChannelState& state, float value) {
    state.activity = state.highpass.filterOne(value);
    state.absolute_deviation = std::fabs(state.activity);
    state.mad_estimate = state.mad_lowpass.filterOne(state.absolute_deviation);
    state.threshold = -state.mad_estimate * config_.threshold_mad_scale;
    state.median_estimate = state.median_lowpass.filterOne(value);

    bool spike = false;
    if (state.samples_since_last_spike >= config_.refractory_samples) {
        if (state.prev_activity > state.threshold && state.activity <= state.threshold) {
            spike = true;
        }
    }

    if (spike) {
        state.spike_count++;
        state.samples_since_last_spike = 0;
    } else {
        state.samples_since_last_spike++;
    }

    state.prev_activity = state.activity;
}
