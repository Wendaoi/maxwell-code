#include "motor_decoder.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr double kMinCorrectionFactor = 0.25;
constexpr double kMaxCorrectionFactor = 4.0;

}  // namespace

MotorDecoder::MotorDecoder(MotorDecoderConfig config) : config_(std::move(config)) {
    up_.channels = config_.up_channels;
    down_.channels = config_.down_channels;
}

void MotorDecoder::observeBaselineWindow(const std::vector<std::uint32_t>& spike_counts) {
    if (gains_frozen_) {
        return;
    }

    std::uint64_t up_valid_channels = 0;
    std::uint64_t down_valid_channels = 0;

    up_.baseline_spikes += sumValidCounts(spike_counts, up_.channels, &up_valid_channels);
    down_.baseline_spikes += sumValidCounts(spike_counts, down_.channels, &down_valid_channels);
    up_.saw_valid_configured_channel = up_.saw_valid_configured_channel || up_valid_channels > 0;
    down_.saw_valid_configured_channel =
        down_.saw_valid_configured_channel || down_valid_channels > 0;
    ++up_.baseline_windows;
    ++down_.baseline_windows;
}

void MotorDecoder::freezeGains() {
    if (gains_frozen_) {
        return;
    }

    up_.gain = computeGain(up_, config_);
    down_.gain = computeGain(down_, config_);
    gains_frozen_ = true;
}

MotorActivity MotorDecoder::decodeWindow(const std::vector<std::uint32_t>& spike_counts) const {
    return decodeWindowInternal(spike_counts);
}

std::uint32_t MotorDecoder::sumValidCounts(const std::vector<std::uint32_t>& spike_counts,
                                           const std::vector<std::size_t>& channels,
                                           std::uint64_t* ignored_valid_channel_count) {
    std::uint32_t total = 0;
    std::uint64_t valid_count = 0;

    for (std::size_t channel : channels) {
        if (channel >= spike_counts.size()) {
            continue;
        }
        total += spike_counts[channel];
        ++valid_count;
    }

    if (ignored_valid_channel_count) {
        *ignored_valid_channel_count = valid_count;
    }

    return total;
}

double MotorDecoder::computeGain(const GroupState& group, const MotorDecoderConfig& config) {
    if (group.channels.empty() || !group.saw_valid_configured_channel || group.baseline_windows == 0 ||
        config.window_ms <= 0 || config.target_rate_hz <= 0.0) {
        return 1.0;
    }

    const double window_seconds = static_cast<double>(config.window_ms) / 1000.0;
    const double configured_channel_count = static_cast<double>(group.channels.size());
    if (configured_channel_count <= 0.0 || window_seconds <= 0.0) {
        return 1.0;
    }

    const double baseline_rate_hz = static_cast<double>(group.baseline_spikes) /
                                    static_cast<double>(group.baseline_windows) /
                                    configured_channel_count / window_seconds;
    const double correction_factor = baseline_rate_hz / config.target_rate_hz;
    return std::clamp(correction_factor, kMinCorrectionFactor, kMaxCorrectionFactor);
}

MotorActivity MotorDecoder::decodeWindowInternal(
    const std::vector<std::uint32_t>& spike_counts) const {
    MotorActivity activity;
    activity.raw_up = sumValidCounts(spike_counts, up_.channels, nullptr);
    activity.raw_down = sumValidCounts(spike_counts, down_.channels, nullptr);
    activity.up_gain = gains_frozen_ ? up_.gain : 1.0;
    activity.down_gain = gains_frozen_ ? down_.gain : 1.0;
    activity.corrected_up = activity.up_gain <= 0.0 ? 0.0 : activity.raw_up / activity.up_gain;
    activity.corrected_down =
        activity.down_gain <= 0.0 ? 0.0 : activity.raw_down / activity.down_gain;
    return activity;
}
