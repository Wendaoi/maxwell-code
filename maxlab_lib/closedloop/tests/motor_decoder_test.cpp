#include "motor_decoder.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_close(double actual, double expected, double tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

void test_decode_uses_identity_gains_before_freeze() {
    MotorDecoderConfig config;
    config.up_channels = {0, 1, 99};
    config.down_channels = {2, 3, 100};

    MotorDecoder decoder(config);
    const std::vector<std::uint32_t> spike_counts = {3, 0, 4, 0};

    const MotorActivity activity = decoder.decodeWindow(spike_counts);

    expect(activity.raw_up == 3, "raw_up before freeze");
    expect(activity.raw_down == 4, "raw_down before freeze");
    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain before freeze");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain before freeze");
    expect_close(activity.corrected_up, 3.0, 1e-9, "corrected_up before freeze");
    expect_close(activity.corrected_down, 4.0, 1e-9, "corrected_down before freeze");
}

void test_freeze_computes_gain_from_baseline_windows() {
    MotorDecoderConfig config;
    config.up_channels = {0, 1, 99};
    config.down_channels = {2};

    MotorDecoder decoder(config);

    for (int i = 0; i < 10; ++i) {
        decoder.observeBaselineWindow({1, 0, 1, 0});
    }

    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({3, 0, 4, 0});

    expect_close(activity.up_gain, 1.666666666667, 1e-9, "up_gain after freeze");
    expect_close(activity.down_gain, 4.0, 1e-9, "down_gain after freeze");
    expect_close(activity.corrected_up, 1.8, 1e-9, "corrected_up after freeze");
    expect_close(activity.corrected_down, 1.0, 1e-9, "corrected_down after freeze");
}

void test_no_baseline_windows_keeps_identity_gains() {
    MotorDecoderConfig config;
    config.up_channels = {0};
    config.down_channels = {1};

    MotorDecoder decoder(config);
    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({5, 7});

    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain with no baseline");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain with no baseline");
}

void test_empty_channel_groups_keep_identity_gains() {
    MotorDecoderConfig config;

    MotorDecoder decoder(config);
    decoder.observeBaselineWindow({5, 7});
    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({5, 7});

    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain with empty group");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain with empty group");
}

void test_fully_out_of_range_group_keeps_identity_gains() {
    MotorDecoderConfig config;
    config.up_channels = {99};
    config.down_channels = {100};

    MotorDecoder decoder(config);
    for (int i = 0; i < 4; ++i) {
        decoder.observeBaselineWindow({1});
    }
    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({1});

    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain with out-of-range group");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain with out-of-range group");
}

void test_zero_window_ms_keeps_identity_gains() {
    MotorDecoderConfig config;
    config.up_channels = {0};
    config.down_channels = {1};
    config.window_ms = 0;

    MotorDecoder decoder(config);
    decoder.observeBaselineWindow({1, 1});
    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({5, 7});

    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain with zero window_ms");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain with zero window_ms");
}

void test_zero_target_rate_keeps_identity_gains() {
    MotorDecoderConfig config;
    config.up_channels = {0};
    config.down_channels = {1};
    config.target_rate_hz = 0.0;

    MotorDecoder decoder(config);
    decoder.observeBaselineWindow({1, 1});
    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({5, 7});

    expect_close(activity.up_gain, 1.0, 1e-9, "up_gain with zero target_rate_hz");
    expect_close(activity.down_gain, 1.0, 1e-9, "down_gain with zero target_rate_hz");
}

void test_gain_clamps_to_expected_bounds() {
    MotorDecoderConfig config;
    config.up_channels = {0};
    config.down_channels = {1};

    MotorDecoder decoder(config);

    for (int i = 0; i < 10; ++i) {
        decoder.observeBaselineWindow({10, 0});
    }

    decoder.freezeGains();

    const MotorActivity activity = decoder.decodeWindow({10, 10});

    expect_close(activity.up_gain, 4.0, 1e-9, "up_gain upper clamp");
    expect_close(activity.down_gain, 0.25, 1e-9, "down_gain lower clamp");
    expect_close(activity.corrected_up, 2.5, 1e-9, "corrected_up upper clamp");
    expect_close(activity.corrected_down, 40.0, 1e-9, "corrected_down lower clamp");
}

}  // namespace

int main() {
    try {
        test_decode_uses_identity_gains_before_freeze();
        test_freeze_computes_gain_from_baseline_windows();
        test_no_baseline_windows_keeps_identity_gains();
        test_empty_channel_groups_keep_identity_gains();
        test_fully_out_of_range_group_keeps_identity_gains();
        test_zero_window_ms_keeps_identity_gains();
        test_zero_target_rate_keeps_identity_gains();
        test_gain_clamps_to_expected_bounds();
    } catch (const std::exception& e) {
        std::cerr << "motor_decoder_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "motor_decoder_test passed\n";
    return 0;
}
