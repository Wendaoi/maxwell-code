#include "runtime_config.h"

#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    Storage storage;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(storage); }
    bool is_bool() const { return std::holds_alternative<bool>(storage); }
    bool is_number() const { return std::holds_alternative<double>(storage); }
    bool is_string() const { return std::holds_alternative<std::string>(storage); }
    bool is_array() const { return std::holds_alternative<Array>(storage); }
    bool is_object() const { return std::holds_alternative<Object>(storage); }

    const bool& as_bool() const { return std::get<bool>(storage); }
    double as_number() const { return std::get<double>(storage); }
    const std::string& as_string() const { return std::get<std::string>(storage); }
    const Array& as_array() const { return std::get<Array>(storage); }
    const Object& as_object() const { return std::get<Object>(storage); }
};

int hex_digit_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    throw std::runtime_error("Malformed JSON unicode escape");
}

void append_utf8(std::string& result, std::uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        result.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        throw std::runtime_error("Invalid JSON unicode code point");
    }
}

class JsonParser {
public:
    explicit JsonParser(std::string input) : input_(std::move(input)) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value();
        skip_ws();
        if (!eof()) {
            throw std::runtime_error("Unexpected trailing JSON content");
        }
        return value;
    }

private:
    std::string input_;
    std::size_t pos_ = 0;

    bool eof() const { return pos_ >= input_.size(); }

    char peek() const {
        if (eof()) {
            throw std::runtime_error("Unexpected end of JSON");
        }
        return input_[pos_];
    }

    char get() {
        char c = peek();
        ++pos_;
        return c;
    }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    void expect(char expected) {
        if (get() != expected) {
            throw std::runtime_error("Malformed JSON");
        }
    }

    JsonValue parse_value() {
        skip_ws();
        if (eof()) {
            throw std::runtime_error("Unexpected end of JSON");
        }

        switch (peek()) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return JsonValue{.storage = parse_string()};
            case 't':
                parse_literal("true");
                return JsonValue{.storage = true};
            case 'f':
                parse_literal("false");
                return JsonValue{.storage = false};
            case 'n':
                parse_literal("null");
                return JsonValue{.storage = nullptr};
            default:
                return JsonValue{.storage = parse_number()};
        }
    }

    void parse_literal(std::string_view literal) {
        for (char expected : literal) {
            if (eof() || get() != expected) {
                throw std::runtime_error("Malformed JSON literal");
            }
        }
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue::Object object;
        skip_ws();
        if (!eof() && peek() == '}') {
            ++pos_;
            return JsonValue{.storage = std::move(object)};
        }

        while (true) {
            skip_ws();
            if (eof() || peek() != '"') {
                throw std::runtime_error("Expected JSON object key");
            }
            std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue value = parse_value();
            object.emplace(std::move(key), std::move(value));
            skip_ws();
            if (!eof() && peek() == '}') {
                ++pos_;
                break;
            }
            expect(',');
        }

        return JsonValue{.storage = std::move(object)};
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue::Array array;
        skip_ws();
        if (!eof() && peek() == ']') {
            ++pos_;
            return JsonValue{.storage = std::move(array)};
        }

        while (true) {
            array.push_back(parse_value());
            skip_ws();
            if (!eof() && peek() == ']') {
                ++pos_;
                break;
            }
            expect(',');
        }

        return JsonValue{.storage = std::move(array)};
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        auto parse_unicode_code_unit = [this]() -> std::uint32_t {
            std::uint32_t code_unit = 0;
            for (int i = 0; i < 4; ++i) {
                if (eof()) {
                    throw std::runtime_error("Unterminated unicode escape");
                }
                code_unit = (code_unit << 4) |
                            static_cast<std::uint32_t>(hex_digit_value(get()));
            }
            return code_unit;
        };
        while (true) {
            if (eof()) {
                throw std::runtime_error("Unterminated JSON string");
            }
            char c = get();
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                if (eof()) {
                    throw std::runtime_error("Unterminated escape sequence");
                }
                char escaped = get();
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(escaped);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u': {
                        std::uint32_t codepoint = parse_unicode_code_unit();
                        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                            if (eof() || get() != '\\' || eof() || get() != 'u') {
                                throw std::runtime_error("Expected low surrogate after high surrogate");
                            }
                            const std::uint32_t low_surrogate = parse_unicode_code_unit();
                            if (low_surrogate < 0xDC00 || low_surrogate > 0xDFFF) {
                                throw std::runtime_error("Invalid JSON surrogate pair");
                            }
                            codepoint = 0x10000 + (((codepoint - 0xD800) << 10) |
                                                   (low_surrogate - 0xDC00));
                        } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                            throw std::runtime_error("Unexpected low surrogate");
                        }
                        append_utf8(result, codepoint);
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported JSON escape");
                }
            } else {
                result.push_back(c);
            }
        }
        return result;
    }

    double parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        bool saw_digit = false;
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
            ++pos_;
            saw_digit = true;
        }
        if (!eof() && peek() == '.') {
            ++pos_;
            bool saw_fraction_digit = false;
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
                saw_fraction_digit = true;
            }
            if (!saw_fraction_digit) {
                throw std::runtime_error("Malformed JSON number");
            }
        }
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) {
                ++pos_;
            }
            bool saw_exponent_digit = false;
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
                saw_exponent_digit = true;
            }
            if (!saw_exponent_digit) {
                throw std::runtime_error("Malformed JSON number");
            }
        }
        std::string token = input_.substr(start, pos_ - start);
        if (!saw_digit || token.empty() || token == "-") {
            throw std::runtime_error("Malformed JSON number");
        }
        return std::stod(token);
    }
};

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    std::size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string normalize_token(const std::string& value) {
    std::string result = trim_copy(value);
    for (char& c : result) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return result;
}

const JsonValue& require_member(const JsonValue::Object& object, const std::string& key) {
    auto it = object.find(key);
    if (it == object.end()) {
        throw std::runtime_error("Missing required field: " + key);
    }
    return it->second;
}

const JsonValue* optional_member(const JsonValue::Object& object, const std::string& key) {
    auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return &it->second;
}

const JsonValue::Object& require_object(const JsonValue& value, const std::string& context) {
    if (!value.is_object()) {
        throw std::runtime_error("Expected object for " + context);
    }
    return value.as_object();
}

const JsonValue::Array& require_array(const JsonValue& value, const std::string& context) {
    if (!value.is_array()) {
        throw std::runtime_error("Expected array for " + context);
    }
    return value.as_array();
}

std::string require_string(const JsonValue& value, const std::string& context) {
    if (!value.is_string()) {
        throw std::runtime_error("Expected string for " + context);
    }
    return value.as_string();
}

long long require_int(const JsonValue& value, const std::string& context) {
    if (!value.is_number()) {
        throw std::runtime_error("Expected integer for " + context);
    }
    const double number = value.as_number();
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<long long>::min()) ||
        number > static_cast<double>(std::numeric_limits<long long>::max())) {
        throw std::runtime_error("Expected integer for " + context);
    }
    return static_cast<long long>(number);
}

double require_number(const JsonValue& value, const std::string& context) {
    if (!value.is_number()) {
        throw std::runtime_error("Expected number for " + context);
    }
    const double number = value.as_number();
    if (!std::isfinite(number)) {
        throw std::runtime_error("Expected finite number for " + context);
    }
    return number;
}

std::vector<int> read_int_array(const JsonValue& value, const std::string& context) {
    const auto& array = require_array(value, context);
    std::vector<int> result;
    result.reserve(array.size());
    for (std::size_t i = 0; i < array.size(); ++i) {
        result.push_back(static_cast<int>(require_int(array[i], context + "[" + std::to_string(i) + "]")));
    }
    return result;
}

std::vector<std::string> read_string_array(const JsonValue& value, const std::string& context) {
    const auto& array = require_array(value, context);
    std::vector<std::string> result;
    result.reserve(array.size());
    for (std::size_t i = 0; i < array.size(); ++i) {
        result.push_back(require_string(array[i], context + "[" + std::to_string(i) + "]"));
    }
    return result;
}

std::map<int, std::string> read_frequency_lookup(const JsonValue& value, const std::string& context) {
    const auto& object = require_object(value, context);
    std::map<int, std::string> result;
    for (const auto& [key, entry] : object) {
        int frequency = std::stoi(key);
        result.emplace(frequency, require_string(entry, context + "." + key));
    }
    return result;
}

std::map<std::string, std::map<int, std::string>> read_sequence_lookup(const JsonValue& value) {
    const auto& object = require_object(value, "sequences.ball_position.sequence_lookup");
    std::map<std::string, std::map<int, std::string>> result;
    for (const auto& [position, entry] : object) {
        result.emplace(position, read_frequency_lookup(entry, "sequences.ball_position.sequence_lookup." + position));
    }
    return result;
}

void validate_channels(const std::vector<int>& channels, const std::string& context) {
    if (channels.empty()) {
        throw std::runtime_error(context + " must not be empty");
    }
    for (int channel : channels) {
        if (channel < 0 || channel >= 1024) {
            throw std::runtime_error(context + " contains channel outside 0..1023");
        }
    }
}

void validate_runtime_contract(const RuntimeConfig& config) {
    if (config.sample_rate_hz <= 0.0) {
        throw std::runtime_error("runtime.sample_rate_hz must be positive");
    }
    if (config.window_ms <= 0) {
        throw std::runtime_error("runtime.window_ms must be positive");
    }
    if (config.pre_rest_seconds < 0 || config.game_seconds <= 0 ||
        config.exclude_initial_game_seconds < 0) {
        throw std::runtime_error("runtime phase durations are invalid");
    }
    if (config.motor_gain_target_hz <= 0.0) {
        throw std::runtime_error("runtime.motor_gain_target_hz must be positive");
    }
    if (config.miss_feedback_duration_ms < 0 ||
        config.miss_pause_ms < 0 ||
        config.hit_sensory_suppression_ms < 0 ||
        config.sensory_blinding_ms < 0 ||
        config.hit_feedback_blinding_ms < 0 ||
        config.miss_feedback_blinding_ms < 0) {
        throw std::runtime_error("runtime timings must be non-negative");
    }
    if (config.spike_threshold_mad_scale <= 0.0) {
        throw std::runtime_error("spike_detection.threshold_mad_scale must be positive");
    }
    if (config.spike_refractory_period_ms < 0.0) {
        throw std::runtime_error("spike_detection.refractory_period_ms must be non-negative");
    }

    validate_channels(config.motor_up_channels, "channels.motor_up_channels");
    validate_channels(config.motor_down_channels, "channels.motor_down_channels");
    validate_channels(config.stim_channels, "channels.stim_channels");

    if (config.positions.empty()) {
        throw std::runtime_error("sequences.ball_position.positions must not be empty");
    }
    if (config.frequencies.empty()) {
        throw std::runtime_error("sequences.ball_position.frequencies must not be empty");
    }
    for (int frequency : config.frequencies) {
        if (frequency <= 0) {
            throw std::runtime_error("sequences.ball_position.frequencies must be positive");
        }
    }

    for (const std::string& position : config.positions) {
        const auto position_it = config.sequence_lookup.find(position);
        if (position_it == config.sequence_lookup.end()) {
            throw std::runtime_error("missing sequence lookup for position: " + position);
        }
        for (int frequency : config.frequencies) {
            const auto sequence_it = position_it->second.find(frequency);
            if (sequence_it == position_it->second.end() || sequence_it->second.empty()) {
                throw std::runtime_error("missing sequence lookup entry for position/frequency");
            }
        }
    }
}

}  // namespace

RuntimeCondition parse_condition(const std::string& value) {
    const std::string normalized = normalize_token(value);
    if (normalized == "STIM" || normalized == "STIMULUS") {
        return RuntimeCondition::Stimulus;
    }
    if (normalized == "SILENT") {
        return RuntimeCondition::Silent;
    }
    if (normalized == "NO_FEEDBACK" || normalized == "NO-FEEDBACK" || normalized == "NOFEEDBACK") {
        return RuntimeCondition::NoFeedback;
    }
    if (normalized == "REST") {
        return RuntimeCondition::Rest;
    }
    throw std::runtime_error("Unsupported condition: " + value);
}

StreamMode parse_stream_mode(const std::string& value) {
    const std::string normalized = normalize_token(value);
    if (normalized == "RAW") {
        return StreamMode::Raw;
    }
    if (normalized == "FILTERED") {
        return StreamMode::Filtered;
    }
    throw std::runtime_error("Unsupported stream mode: " + value);
}

const std::string& RuntimeConfig::sequence_for(std::size_t position_index, std::size_t frequency_index) const {
    if (position_index >= positions.size()) {
        throw std::out_of_range("position index out of range");
    }
    if (frequency_index >= frequencies.size()) {
        throw std::out_of_range("frequency index out of range");
    }

    const std::string& position = positions.at(position_index);
    const int frequency = frequencies.at(frequency_index);
    const auto position_it = sequence_lookup.find(position);
    if (position_it == sequence_lookup.end()) {
        throw std::out_of_range("missing sequence lookup for position");
    }
    return position_it->second.at(frequency);
}

RuntimeConfig load_runtime_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open runtime config: " + path);
    }

    std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    JsonParser parser(std::move(json));
    JsonValue root = parser.parse();

    const auto& root_object = require_object(root, "root");
    const auto& runtime_object = require_object(require_member(root_object, "runtime"), "runtime");
    const auto& channels_object = require_object(require_member(root_object, "channels"), "channels");
    const auto& sequences_object = require_object(require_member(root_object, "sequences"), "sequences");
    const auto& ball_position_object = require_object(require_member(sequences_object, "ball_position"), "sequences.ball_position");
    const auto& recording_object = require_object(require_member(root_object, "recording"), "recording");

    RuntimeConfig config;
    config.condition = parse_condition(require_string(require_member(root_object, "condition"), "condition"));
    config.stream_mode = parse_stream_mode(require_string(require_member(runtime_object, "stream_mode"), "runtime.stream_mode"));
    config.sample_rate_hz = require_number(require_member(runtime_object, "sample_rate_hz"), "runtime.sample_rate_hz");
    config.window_ms = static_cast<int>(require_int(require_member(runtime_object, "window_ms"), "runtime.window_ms"));
    config.pre_rest_seconds = static_cast<int>(require_int(require_member(runtime_object, "pre_rest_seconds"), "runtime.pre_rest_seconds"));
    config.game_seconds = static_cast<int>(require_int(require_member(runtime_object, "game_seconds"), "runtime.game_seconds"));
    config.exclude_initial_game_seconds = static_cast<int>(require_int(require_member(runtime_object, "exclude_initial_game_seconds"), "runtime.exclude_initial_game_seconds"));
    config.miss_feedback_duration_ms = static_cast<int>(require_int(require_member(runtime_object, "miss_feedback_duration_ms"), "runtime.miss_feedback_duration_ms"));
    config.miss_pause_ms = static_cast<int>(require_int(require_member(runtime_object, "miss_pause_ms"), "runtime.miss_pause_ms"));
    config.hit_sensory_suppression_ms = static_cast<int>(require_int(require_member(runtime_object, "hit_sensory_suppression_ms"), "runtime.hit_sensory_suppression_ms"));
    config.sensory_blinding_ms = static_cast<int>(require_int(require_member(runtime_object, "sensory_blinding_ms"), "runtime.sensory_blinding_ms"));
    config.hit_feedback_blinding_ms = static_cast<int>(require_int(require_member(runtime_object, "hit_feedback_blinding_ms"), "runtime.hit_feedback_blinding_ms"));
    config.miss_feedback_blinding_ms = static_cast<int>(require_int(require_member(runtime_object, "miss_feedback_blinding_ms"), "runtime.miss_feedback_blinding_ms"));
    config.motor_gain_target_hz = require_number(require_member(runtime_object, "motor_gain_target_hz"), "runtime.motor_gain_target_hz");

    if (const JsonValue* spike_detection = optional_member(root_object, "spike_detection")) {
        const auto& spike_object = require_object(*spike_detection, "spike_detection");
        if (const JsonValue* mad_scale = optional_member(spike_object, "threshold_mad_scale")) {
            config.spike_threshold_mad_scale =
                require_number(*mad_scale, "spike_detection.threshold_mad_scale");
        } else {
            config.spike_threshold_mad_scale =
                require_number(require_member(spike_object, "threshold_std"),
                               "spike_detection.threshold_std");
        }
        config.spike_refractory_period_ms = require_number(require_member(spike_object, "refractory_period_ms"), "spike_detection.refractory_period_ms");
    }

    config.motor_up_channels = read_int_array(require_member(channels_object, "motor_up_channels"), "channels.motor_up_channels");
    config.motor_down_channels = read_int_array(require_member(channels_object, "motor_down_channels"), "channels.motor_down_channels");
    config.stim_channels = read_int_array(require_member(channels_object, "stim_channels"), "channels.stim_channels");

    config.positions = read_string_array(require_member(ball_position_object, "positions"), "sequences.ball_position.positions");
    config.frequencies = read_int_array(require_member(ball_position_object, "frequencies"), "sequences.ball_position.frequencies");
    config.sequence_lookup = read_sequence_lookup(require_member(ball_position_object, "sequence_lookup"));
    config.hit_feedback_sequence = require_string(require_member(require_object(require_member(sequences_object, "hit_feedback"), "sequences.hit_feedback"), "sequence_name"), "sequences.hit_feedback.sequence_name");
    config.miss_feedback_sequences = read_string_array(require_member(require_object(require_member(sequences_object, "miss_feedback"), "sequences.miss_feedback"), "sequence_names"), "sequences.miss_feedback.sequence_names");
    config.runtime_events_path = require_string(require_member(recording_object, "runtime_events"), "recording.runtime_events");
    config.window_samples_path = require_string(require_member(recording_object, "window_samples"), "recording.window_samples");
    config.quality_summary_path = require_string(require_member(recording_object, "quality_summary"), "recording.quality_summary");

    validate_runtime_contract(config);

    return config;
}
