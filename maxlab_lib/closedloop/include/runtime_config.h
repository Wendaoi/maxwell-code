#ifndef RUNTIME_CONFIG_H
#define RUNTIME_CONFIG_H

#include <cstddef>
#include <map>
#include <string>
#include <vector>

enum class RuntimeCondition {
    Stimulus,
    Silent,
    NoFeedback,
    Rest,
};

enum class StreamMode {
    Raw,
    Filtered,
};

struct RuntimeConfig {
    RuntimeCondition condition;
    StreamMode stream_mode;
    double sample_rate_hz;
    int window_ms;
    int pre_rest_seconds;
    int game_seconds;
    int exclude_initial_game_seconds;
    int miss_feedback_duration_ms;
    int miss_pause_ms;
    int hit_sensory_suppression_ms;
    int sensory_blinding_ms;
    int hit_feedback_blinding_ms;
    int miss_feedback_blinding_ms;
    double motor_gain_target_hz;
    double spike_threshold_mad_scale = 5.0;
    double spike_refractory_period_ms = 2.0;
    std::vector<int> motor_up_channels;
    std::vector<int> motor_down_channels;
    std::vector<int> stim_channels;
    std::vector<std::string> positions;
    std::vector<int> frequencies;
    std::map<std::string, std::map<int, std::string>> sequence_lookup;
    std::string hit_feedback_sequence;
    std::vector<std::string> miss_feedback_sequences;
    std::string runtime_events_path;
    std::string window_samples_path;
    std::string quality_summary_path;

    const std::string& sequence_for(std::size_t position_index, std::size_t frequency_index) const;
};

RuntimeCondition parse_condition(const std::string& value);
StreamMode parse_stream_mode(const std::string& value);
RuntimeConfig load_runtime_config(const std::string& path);

#endif  // RUNTIME_CONFIG_H
