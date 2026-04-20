#include "spike_detection.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_constant_baseline_does_not_spike() {
    SpikeDetectorConfig config;
    config.sample_rate_hz = 20000.0;
    config.threshold_mad_scale = 5.0f;
    config.refractory_samples = 40;
    SpikeDetector detector(1, config);

    for (int i = 0; i < 4000; ++i) {
        const float sample = -12.0f;
        detector.processFrame(&sample, 1);
    }

    std::vector<std::uint32_t> counts;
    detector.getCounts(&counts);
    expect(counts.at(0) == 0, "constant baseline should not spike");
}

void test_negative_pulse_crosses_mad_threshold_once() {
    SpikeDetectorConfig config;
    config.sample_rate_hz = 20000.0;
    config.threshold_mad_scale = 5.0f;
    config.refractory_samples = 40;
    SpikeDetector detector(1, config);

    for (int i = 0; i < 4000; ++i) {
        const float sample = 0.0f;
        detector.processFrame(&sample, 1);
    }

    for (int i = 0; i < 8; ++i) {
        const float pulse = -120.0f;
        detector.processFrame(&pulse, 1);
    }

    std::vector<std::uint32_t> counts;
    detector.getCounts(&counts);
    expect(counts.at(0) == 1, "single pulse should spike exactly once");
}

void test_refractory_blocks_immediate_redetection() {
    SpikeDetectorConfig config;
    config.sample_rate_hz = 20000.0;
    config.threshold_mad_scale = 5.0f;
    config.refractory_samples = 40;
    SpikeDetector detector(1, config);

    for (int i = 0; i < 4000; ++i) {
        const float sample = 0.0f;
        detector.processFrame(&sample, 1);
    }

    for (int i = 0; i < 8; ++i) {
        const float pulse = -120.0f;
        detector.processFrame(&pulse, 1);
    }
    for (int i = 0; i < 10; ++i) {
        const float sample = 0.0f;
        detector.processFrame(&sample, 1);
    }
    for (int i = 0; i < 8; ++i) {
        const float pulse = -120.0f;
        detector.processFrame(&pulse, 1);
    }

    std::vector<std::uint32_t> counts;
    detector.getCounts(&counts);
    expect(counts.at(0) == 1, "refractory should suppress immediate re-detection");
}

void test_threshold_has_no_hard_floor() {
    SpikeDetectorConfig config;
    config.sample_rate_hz = 20000.0;
    config.threshold_mad_scale = 5.0f;
    config.refractory_samples = 40;
    SpikeDetector detector(1, config);

    for (int i = 0; i < 4000; ++i) {
        const float sample = 0.0f;
        detector.processFrame(&sample, 1);
    }

    const SpikeDetector::ChannelDebugState state = detector.debugState(0);
    expect(state.threshold > -0.001f, "threshold should be derived from MAD only, not a hard floor");
}

void test_median_estimate_side_path_tracks_baseline_independently() {
    SpikeDetectorConfig config;
    config.sample_rate_hz = 20000.0;
    config.threshold_mad_scale = 5.0f;
    config.refractory_samples = 40;
    SpikeDetector detector(1, config);

    for (int i = 0; i < 6000; ++i) {
        const float sample = 25.0f;
        detector.processFrame(&sample, 1);
    }

    const SpikeDetector::ChannelDebugState state = detector.debugState(0);
    expect(state.median_estimate > 10.0f, "median side path should track baseline");
    expect(std::fabs(state.activity) < 0.1f, "high-pass activity should decay near zero for constant baseline");
}

}  // namespace

int main() {
    try {
        test_constant_baseline_does_not_spike();
        test_negative_pulse_crosses_mad_threshold_once();
        test_refractory_blocks_immediate_redetection();
        test_threshold_has_no_hard_floor();
        test_median_estimate_side_path_tracks_baseline_independently();
    } catch (const std::exception& e) {
        throw;
    }
    return 0;
}
