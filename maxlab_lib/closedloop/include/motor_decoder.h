#ifndef MOTOR_DECODER_H
#define MOTOR_DECODER_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct MotorDecoderConfig {
    std::vector<std::size_t> up_channels;
    std::vector<std::size_t> down_channels;
    double sample_rate_hz = 20000.0;
    int window_ms = 10;
    double target_rate_hz = 20.0;
};

struct MotorActivity {
    std::uint32_t raw_up = 0;
    std::uint32_t raw_down = 0;
    double up_gain = 1.0;
    double down_gain = 1.0;
    double corrected_up = 0.0;
    double corrected_down = 0.0;
};

class MotorDecoder {
public:
    explicit MotorDecoder(MotorDecoderConfig config);

    void observeBaselineWindow(const std::vector<std::uint32_t>& spike_counts);
    void freezeGains();
    MotorActivity decodeWindow(const std::vector<std::uint32_t>& spike_counts) const;

private:
    struct GroupState {
        std::vector<std::size_t> channels;
        std::uint64_t baseline_spikes = 0;
        std::size_t baseline_windows = 0;
        bool saw_valid_configured_channel = false;
        double gain = 1.0;
    };

    static std::uint32_t sumValidCounts(const std::vector<std::uint32_t>& spike_counts,
                                        const std::vector<std::size_t>& channels,
                                        std::uint64_t* ignored_valid_channel_count);
    static double computeGain(const GroupState& group, const MotorDecoderConfig& config);
    MotorActivity decodeWindowInternal(const std::vector<std::uint32_t>& spike_counts) const;

    MotorDecoderConfig config_;
    GroupState up_;
    GroupState down_;
    bool gains_frozen_ = false;
};

#endif  // MOTOR_DECODER_H
