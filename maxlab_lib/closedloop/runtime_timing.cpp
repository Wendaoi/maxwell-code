#include "runtime_timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

std::uint64_t rounded_sample_count(double samples, const char* context) {
    if (!std::isfinite(samples) || samples < 0.0) {
        throw std::invalid_argument(context);
    }
    if (samples == 0.0) {
        return 0;
    }
    const double rounded = std::round(samples);
    if (rounded > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(context);
    }
    return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(rounded));
}

}  // namespace

std::uint64_t samples_for_duration_ms(double sample_rate_hz, int duration_ms) {
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0 || duration_ms < 0) {
        throw std::invalid_argument("invalid sample duration");
    }
    return rounded_sample_count(sample_rate_hz * static_cast<double>(duration_ms) / 1000.0,
                                "invalid sample duration");
}

std::uint64_t samples_per_window(double sample_rate_hz, int window_ms) {
    if (window_ms <= 0) {
        throw std::invalid_argument("window_ms must be positive");
    }
    return std::max<std::uint64_t>(1, samples_for_duration_ms(sample_rate_hz, window_ms));
}

std::uint64_t samples_per_sensory_interval(double sample_rate_hz, int frequency_hz) {
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0 || frequency_hz <= 0) {
        throw std::invalid_argument("invalid sensory frequency");
    }
    return rounded_sample_count(sample_rate_hz / static_cast<double>(frequency_hz),
                                "invalid sensory frequency");
}

std::uint64_t extend_blinding_until(std::uint64_t current_frame,
                                    std::uint64_t duration_samples,
                                    std::uint64_t active_until_sample) {
    if (duration_samples == 0) {
        return active_until_sample;
    }
    return std::max(active_until_sample, current_frame + duration_samples);
}

int elapsed_ms_from_phase_start(std::uint64_t phase_start_frame,
                                std::uint64_t current_frame,
                                double sample_rate_hz) {
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0 ||
        current_frame < phase_start_frame) {
        throw std::invalid_argument("invalid phase elapsed time");
    }

    const double elapsed_ms = static_cast<double>(current_frame - phase_start_frame) * 1000.0 /
                              sample_rate_hz;
    const double rounded = std::round(elapsed_ms);
    if (rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("phase elapsed time exceeds int range");
    }
    return static_cast<int>(rounded);
}

int combined_miss_pause_ms(int feedback_duration_ms, int rest_pause_ms) {
    if (feedback_duration_ms < 0 || rest_pause_ms < 0) {
        throw std::invalid_argument("miss timings must be non-negative");
    }
    const long long total = static_cast<long long>(feedback_duration_ms) +
                            static_cast<long long>(rest_pause_ms);
    if (total > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("combined miss pause exceeds int range");
    }
    return static_cast<int>(total);
}
