#include "runtime_config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path);
    expect(static_cast<bool>(out), "failed to open test file");
    out << content;
}

void test_alias_parsers() {
    expect(parse_condition("STIM") == RuntimeCondition::Stimulus, "STIM");
    expect(parse_condition("stimulus") == RuntimeCondition::Stimulus, "STIMULUS");
    expect(parse_condition("silent") == RuntimeCondition::Silent, "SILENT");
    expect(parse_condition("NO_FEEDBACK") == RuntimeCondition::NoFeedback, "NO_FEEDBACK");
    expect(parse_condition("NO-FEEDBACK") == RuntimeCondition::NoFeedback, "NO-FEEDBACK");
    expect(parse_condition("NOFEEDBACK") == RuntimeCondition::NoFeedback, "NOFEEDBACK");
    expect(parse_condition("REST") == RuntimeCondition::Rest, "REST");

    expect(parse_stream_mode("raw") == StreamMode::Raw, "raw");
    expect(parse_stream_mode("FILTERED") == StreamMode::Filtered, "filtered");
}

void test_loader_and_lookup() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "runtime_config_test.json";
    write_file(path, R"JSON({
        "condition": "NO-FEEDBACK",
        "runtime": {
            "stream_mode": "filtered",
            "sample_rate_hz": 20000.0,
            "window_ms": 120,
            "pre_rest_seconds": 600,
            "game_seconds": 1200,
            "exclude_initial_game_seconds": 10,
            "miss_feedback_duration_ms": 4000,
            "miss_pause_ms": 300,
            "hit_sensory_suppression_ms": 150,
            "sensory_blinding_ms": 5,
            "hit_feedback_blinding_ms": 105,
            "miss_feedback_blinding_ms": 4005,
            "motor_gain_target_hz": 20.0,
            "positions": ["p0", "p1"],
            "frequencies": [4, 8],
            "sequence_lookup": {
                "p0": {"4": "p0_4hz", "8": "p0_8hz"},
                "p1": {"4": "p1_4hz", "8": "p1_8hz"}
            },
            "hit_feedback_sequence": "hit_feedback",
            "miss_feedback_sequences": ["miss_feedback_0", "miss_feedback_1"]
        },
        "channels": {
            "motor_up_channels": [1, 2],
            "motor_down_channels": [3, 4],
            "stim_channels": [5, 6, 7]
        },
        "sequences": {
            "ball_position": {
                "positions": ["p0", "p1"],
                "frequencies": [4, 8],
                "sequence_lookup": {
                    "p0": {"4": "p0_4hz", "8": "p0_8hz"},
                    "p1": {"4": "p1_4hz", "8": "p1_8hz"}
                }
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback"
            },
            "miss_feedback": {
                "sequence_names": ["miss_feedback_0", "miss_feedback_1"]
            }
        },
        "experiment": {
            "ignored": true
        },
        "recording": {
            "session_name": "session_x",
            "raw_h5": "/tmp/session_x.raw.h5",
            "runtime_events": "/tmp/runtime_events.jsonl",
            "window_samples": "/tmp/window_samples.csv",
            "quality_summary": "/tmp/quality_summary.json"
        },
        "spike_detection": {
            "threshold_mad_scale": 5.0,
            "refractory_period_ms": 2.0
        }
    })JSON");

    const RuntimeConfig config = load_runtime_config(path.string());

    expect(config.condition == RuntimeCondition::NoFeedback, "condition");
    expect(config.stream_mode == StreamMode::Filtered, "stream_mode");
    expect(config.sample_rate_hz == 20000.0, "sample_rate_hz");
    expect(config.window_ms == 120, "window_ms");
    expect(config.pre_rest_seconds == 600, "pre_rest_seconds");
    expect(config.game_seconds == 1200, "game_seconds");
    expect(config.exclude_initial_game_seconds == 10, "exclude_initial_game_seconds");
    expect(config.miss_feedback_duration_ms == 4000, "miss_feedback_duration_ms");
    expect(config.miss_pause_ms == 300, "miss_pause_ms");
    expect(config.hit_sensory_suppression_ms == 150, "hit_sensory_suppression_ms");
    expect(config.sensory_blinding_ms == 5, "sensory_blinding_ms");
    expect(config.hit_feedback_blinding_ms == 105, "hit_feedback_blinding_ms");
    expect(config.miss_feedback_blinding_ms == 4005, "miss_feedback_blinding_ms");
    expect(config.motor_gain_target_hz == 20.0, "motor_gain_target_hz");
    expect(config.spike_threshold_mad_scale == 5.0, "spike_threshold_mad_scale");
    expect(config.spike_refractory_period_ms == 2.0, "spike_refractory_period_ms");
    expect(config.motor_up_channels.size() == 2, "motor_up_channels");
    expect(config.motor_down_channels.size() == 2, "motor_down_channels");
    expect(config.stim_channels.size() == 3, "stim_channels");
    expect(config.positions.size() == 2, "positions");
    expect(config.frequencies.size() == 2, "frequencies");
    expect(config.sequence_lookup.at("p1").at(8) == "p1_8hz", "sequence_lookup");
    expect(config.hit_feedback_sequence == "hit_feedback", "hit_feedback_sequence");
    expect(config.miss_feedback_sequences.size() == 2, "miss_feedback_sequences");
    expect(config.runtime_events_path == "/tmp/runtime_events.jsonl", "runtime_events_path");
    expect(config.window_samples_path == "/tmp/window_samples.csv", "window_samples_path");
    expect(config.quality_summary_path == "/tmp/quality_summary.json", "quality_summary_path");
    expect(config.sequence_for(1, 0) == "p1_4hz", "sequence_for");

    bool caught = false;
    try {
        (void)config.sequence_for(2, 0);
    } catch (const std::out_of_range&) {
        caught = true;
        }
    expect(caught, "sequence_for out_of_range");
}

void test_legacy_threshold_std_alias() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "runtime_config_legacy_threshold.json";
    write_file(path, R"JSON({
        "condition": "STIM",
        "runtime": {
            "stream_mode": "raw",
            "sample_rate_hz": 20000.0,
            "window_ms": 10,
            "pre_rest_seconds": 600,
            "game_seconds": 1200,
            "exclude_initial_game_seconds": 10,
            "miss_feedback_duration_ms": 4000,
            "miss_pause_ms": 4000,
            "hit_sensory_suppression_ms": 100,
            "sensory_blinding_ms": 5,
            "hit_feedback_blinding_ms": 105,
            "miss_feedback_blinding_ms": 4005,
            "motor_gain_target_hz": 20.0
        },
        "channels": {
            "motor_up_channels": [1],
            "motor_down_channels": [2],
            "stim_channels": [3]
        },
        "sequences": {
            "ball_position": {
                "positions": ["p0"],
                "frequencies": [4],
                "sequence_lookup": {
                    "p0": {"4": "p0_4hz"}
                }
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback"
            },
            "miss_feedback": {
                "sequence_names": ["miss_feedback_0"]
            }
        },
        "recording": {
            "session_name": "session_legacy",
            "raw_h5": "/tmp/session_legacy.raw.h5",
            "runtime_events": "/tmp/runtime_events_legacy.jsonl",
            "window_samples": "/tmp/window_samples_legacy.csv",
            "quality_summary": "/tmp/quality_summary_legacy.json"
        },
        "spike_detection": {
            "threshold_std": 7.5,
            "refractory_period_ms": 2.0
        }
    })JSON");

    const RuntimeConfig config = load_runtime_config(path.string());
    expect(config.spike_threshold_mad_scale == 7.5,
           "threshold_std legacy alias should map to spike_threshold_mad_scale");
}

void test_missing_required_field() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "runtime_config_missing.json";
    write_file(path, R"JSON({
        "condition": "STIM",
        "runtime": {
            "stream_mode": "raw"
        },
        "channels": {},
        "sequences": {}
    })JSON");

    bool caught = false;
    try {
        (void)load_runtime_config(path.string());
    } catch (const std::runtime_error&) {
        caught = true;
    }
    expect(caught, "missing required field should throw");
}

void test_invalid_runtime_contract_throws() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "runtime_config_invalid.json";
    write_file(path, R"JSON({
        "condition": "STIM",
        "runtime": {
            "stream_mode": "raw",
            "sample_rate_hz": 20000,
            "window_ms": 10,
            "miss_feedback_duration_ms": 0,
            "miss_pause_ms": 0,
            "hit_sensory_suppression_ms": 0,
            "sensory_blinding_ms": 5,
            "hit_feedback_blinding_ms": 105,
            "miss_feedback_blinding_ms": 4005,
            "motor_gain_target_hz": 20
        },
        "channels": {
            "motor_up_channels": [],
            "motor_down_channels": [2],
            "stim_channels": [3]
        },
        "sequences": {
            "ball_position": {
                "positions": ["p0"],
                "frequencies": [4],
                "sequence_lookup": {
                    "p0": {"4": "p0_4hz"}
                }
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback"
            },
            "miss_feedback": {
                "sequence_names": ["miss_feedback_0"]
            }
        },
        "spike_detection": {
            "threshold_std": 5.0,
            "refractory_period_ms": 2.0
        }
    })JSON");

    bool caught = false;
    try {
        (void)load_runtime_config(path.string());
    } catch (const std::runtime_error&) {
        caught = true;
    }
    expect(caught, "invalid runtime contract should throw");
}

void test_unicode_escape_sequences_in_json_strings() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "runtime_config_unicode.json";
    write_file(path, R"JSON({
        "condition": "STIM",
        "runtime": {
            "stream_mode": "raw",
            "sample_rate_hz": 20000,
            "window_ms": 10,
            "pre_rest_seconds": 600,
            "game_seconds": 1200,
            "exclude_initial_game_seconds": 10,
            "miss_feedback_duration_ms": 4000,
            "miss_pause_ms": 4000,
            "hit_sensory_suppression_ms": 100,
            "sensory_blinding_ms": 5,
            "hit_feedback_blinding_ms": 105,
            "miss_feedback_blinding_ms": 4005,
            "motor_gain_target_hz": 20
        },
        "channels": {
            "motor_up_channels": [1],
            "motor_down_channels": [2],
            "stim_channels": [3]
        },
        "sequences": {
            "ball_position": {
                "positions": ["p0"],
                "frequencies": [4],
                "sequence_lookup": {
                    "p0": {"4": "p0_4hz"}
                }
            },
            "hit_feedback": {
                "sequence_name": "hit_feedback"
            },
            "miss_feedback": {
                "sequence_names": ["miss_feedback_0"]
            }
        },
        "recording": {
            "session_name": "session_\u00d7",
            "raw_h5": "/tmp/session_x.raw.h5",
            "runtime_events": "/tmp/runtime_events_\u00d7.jsonl",
            "window_samples": "/tmp/window_samples.csv",
            "quality_summary": "/tmp/quality_summary.json"
        },
        "experiment": {
            "description": "Motor 1/2 \u00d7 UP/DOWN"
        },
        "spike_detection": {
            "threshold_mad_scale": 5.0,
            "refractory_period_ms": 2.0
        }
    })JSON");

    const RuntimeConfig config = load_runtime_config(path.string());
    expect(config.runtime_events_path == "/tmp/runtime_events_\xC3\x97.jsonl",
           "unicode escape should decode to UTF-8");
}

}  // namespace

int main() {
    try {
        test_alias_parsers();
        test_loader_and_lookup();
        test_legacy_threshold_std_alias();
        test_missing_required_field();
        test_invalid_runtime_contract_throws();
        test_unicode_escape_sequences_in_json_strings();
    } catch (const std::exception& e) {
        std::cerr << "runtime_config_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "runtime_config_test passed\n";
    return 0;
}
