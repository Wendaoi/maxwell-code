#include "runtime_timing.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_samples_per_window_uses_sample_rate() {
    expect(samples_per_window(20000.0, 10) == 200, "10ms at 20kHz should be 200 samples");
}

void test_short_blinding_duration_uses_sample_rate() {
    expect(samples_for_duration_ms(20000.0, 5) == 100, "5ms at 20kHz should be 100 samples");
}

void test_sensory_frequency_interval_uses_sample_rate() {
    expect(samples_per_sensory_interval(20000.0, 40) == 500, "40Hz at 20kHz should be 500 samples");
    expect(samples_per_sensory_interval(20000.0, 4) == 5000, "4Hz at 20kHz should be 5000 samples");
    expect(samples_per_sensory_interval(20000.0, 40) > samples_per_window(20000.0, 10),
           "40Hz sensory should not fire every 200-sample motor window");
    expect(samples_for_duration_ms(20000.0, 5) < samples_per_sensory_interval(20000.0, 40),
           "5ms sensory blinding must be shorter than 40Hz sensory interval");
}

void test_extend_blinding_until_uses_later_timestamp() {
    expect(extend_blinding_until(1000, 100, 0) == 1100, "first blinding window");
    expect(extend_blinding_until(1050, 2100, 1200) == 3150,
           "later command should extend to the later timestamp");
}

void test_combined_miss_pause_includes_feedback_and_rest() {
    expect(combined_miss_pause_ms(4000, 4000) == 8000,
           "miss pause should include feedback duration plus rest pause");
}

void test_elapsed_ms_from_phase_start_uses_stream_frames() {
    expect(elapsed_ms_from_phase_start(1000, 1200, 20000.0) == 10,
           "200 samples after phase start should be 10ms at 20kHz");
    expect(elapsed_ms_from_phase_start(1000, 1400, 20000.0) == 20,
           "400 samples after phase start should be 20ms at 20kHz");
}

}  // namespace

int main() {
    try {
        test_samples_per_window_uses_sample_rate();
        test_short_blinding_duration_uses_sample_rate();
        test_sensory_frequency_interval_uses_sample_rate();
        test_extend_blinding_until_uses_later_timestamp();
        test_combined_miss_pause_includes_feedback_and_rest();
        test_elapsed_ms_from_phase_start_uses_stream_frames();
    } catch (const std::exception& e) {
        std::cerr << "runtime_timing_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "runtime_timing_test passed\n";
    return 0;
}
