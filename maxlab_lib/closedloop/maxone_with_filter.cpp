#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

#include "gamewindow.h"
#include "maxlab/maxlab.h"
#include "motor_decoder.h"
#include "ponggame.h"
#include "runtime_config.h"
#include "runtime_logging.h"
#include "runtime_timing.h"
#include "spike_detection.h"

#ifdef USE_QT
#include <QApplication>
#include <QMetaObject>
#endif

using SteadyClock = std::chrono::steady_clock;

static std::atomic<bool> g_running{true};

struct RunConfig {
    std::string config_path;
    RuntimeConfig runtime;
    uint8_t target_well = 0;
    bool show_gui = true;
    bool wait_for_sync = true;
    std::size_t channel_count = 1024;
    SpikeDetectorConfig detector;
};

static const char* condition_to_string(RuntimeCondition condition) {
    switch (condition) {
        case RuntimeCondition::Stimulus:
            return "STIM";
        case RuntimeCondition::Silent:
            return "SILENT";
        case RuntimeCondition::NoFeedback:
            return "NO_FEEDBACK";
        case RuntimeCondition::Rest:
            return "REST";
    }
    return "UNKNOWN";
}

static const char* stream_mode_to_string(StreamMode mode) {
    switch (mode) {
        case StreamMode::Raw:
            return "raw";
        case StreamMode::Filtered:
            return "filtered";
    }
    return "unknown";
}

static ExperimentCondition to_experiment_condition(RuntimeCondition condition) {
    switch (condition) {
        case RuntimeCondition::Stimulus:
            return ExperimentCondition::Stimulus;
        case RuntimeCondition::Silent:
            return ExperimentCondition::Silent;
        case RuntimeCondition::NoFeedback:
            return ExperimentCondition::NoFeedback;
        case RuntimeCondition::Rest:
            return ExperimentCondition::Rest;
    }
    return ExperimentCondition::Stimulus;
}

static std::vector<std::size_t> to_size_t_channels(const std::vector<int>& channels) {
    std::vector<std::size_t> result;
    result.reserve(channels.size());
    for (int channel : channels) {
        if (channel >= 0) {
            result.push_back(static_cast<std::size_t>(channel));
        }
    }
    return result;
}

static std::size_t infer_channel_count(const RuntimeConfig& runtime) {
    std::size_t channel_count = maxlab::RawFrameData::amplitudeCount;
    auto include_channels = [&channel_count](const std::vector<int>& channels) {
        for (int channel : channels) {
            if (channel >= 0) {
                channel_count = std::max(channel_count, static_cast<std::size_t>(channel) + 1);
            }
        }
    };
    include_channels(runtime.motor_up_channels);
    include_channels(runtime.motor_down_channels);
    include_channels(runtime.stim_channels);
    return channel_count;
}

static MotorDecoderConfig make_decoder_config(const RuntimeConfig& runtime) {
    MotorDecoderConfig config;
    config.up_channels = to_size_t_channels(runtime.motor_up_channels);
    config.down_channels = to_size_t_channels(runtime.motor_down_channels);
    config.sample_rate_hz = runtime.sample_rate_hz;
    config.window_ms = runtime.window_ms;
    config.target_rate_hz = runtime.motor_gain_target_hz;
    return config;
}

struct AppState {
    AppState(const RunConfig& config, GameWindow* ui_window)
        : runtime(config.runtime),
          samples_per_game_window(samples_per_window(runtime.sample_rate_hz, runtime.window_ms)),
          pong_game(),
          detector(config.channel_count, config.detector),
          motor_decoder(make_decoder_config(runtime)),
          logger(runtime.runtime_events_path,
                 runtime.window_samples_path,
                 runtime.quality_summary_path),
          window(ui_window),
          rng(std::random_device{}()) {
        spike_counts.resize(config.channel_count, 0);
        pong_game.setCondition(to_experiment_condition(runtime.condition));
    }

    RuntimeConfig runtime;
    uint64_t samples_per_game_window;
    uint64_t readout_blinding_until_sample = 0;
    uint64_t pause_until_sample = 0;
    uint64_t suppress_sensory_until_sample = 0;
    uint64_t next_sensory_sample = 0;
    uint64_t phase_start_frame = 0;
    uint64_t window_start_frame = 0;
    uint64_t total_frames_seen = 0;
    uint64_t rally_id = 0;
    int last_sensory_pos = -1;
    int last_sensory_hz = 0;
    std::string phase = "pre_rest";
    bool sensory_schedule_initialized = false;
    PongGame pong_game;
    SpikeDetector detector;
    MotorDecoder motor_decoder;
    RuntimeLogger logger;
    GameWindow* window;
    std::vector<std::uint32_t> spike_counts;
    bool window_has_blinded_readout = false;
    uint64_t accepted_frames_in_window = 0;
    uint64_t accepted_stream_frames = 0;
    std::mt19937 rng;
};

static void on_sigint(int) {
    g_running = false;
}

static void reset_window(AppState& state) {
    state.detector.resetCounts();
    state.window_has_blinded_readout = false;
    state.accepted_frames_in_window = 0;
    state.window_start_frame = state.accepted_stream_frames;
}

static void update_window(AppState& state) {
    if (state.window != nullptr) {
        state.window->setState(state.pong_game.getPaddle1Y(),
                               state.pong_game.getBallX(),
                               state.pong_game.getBallY(),
                               state.pong_game.getPaddleHeight());
    }
}

static bool readout_blinded(const AppState& state) {
    return state.accepted_stream_frames <= state.readout_blinding_until_sample;
}

static void apply_readout_blinding(AppState& state, int duration_ms) {
    if (duration_ms <= 0) {
        return;
    }
    const uint64_t duration_samples =
        samples_for_duration_ms(state.runtime.sample_rate_hz, duration_ms);
    state.readout_blinding_until_sample =
        extend_blinding_until(state.accepted_stream_frames,
                              duration_samples,
                              state.readout_blinding_until_sample);
}

static bool poll_start_signal(bool enable_sync) {
    if (!enable_sync) {
        return true;
    }

    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(STDIN_FILENO, &read_fds);
    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    const int ready = select(STDIN_FILENO + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &read_fds)) {
        return false;
    }

    std::string signal;
    if (!std::getline(std::cin, signal)) {
        std::cerr << "[SYNC] stdin closed; starting game loop" << std::endl;
        return true;
    }
    if (signal.find("start") != std::string::npos || signal.empty()) {
        std::cout << "[SYNC] Start signal received" << std::endl;
        return true;
    }

    std::cerr << "[SYNC] Unexpected signal '" << signal << "'; starting anyway" << std::endl;
    return true;
}

static std::size_t frequency_index_for_ball_x(int ball_x,
                                              int game_width,
                                              std::size_t frequency_count) {
    if (frequency_count == 0 || game_width <= 0) {
        return 0;
    }

    const double norm_x =
        std::clamp(static_cast<double>(ball_x) / static_cast<double>(game_width), 0.0, 1.0);
    const double leftward = 1.0 - norm_x;
    const std::size_t index = static_cast<std::size_t>(leftward * frequency_count);
    return std::min(index, frequency_count - 1);
}

static bool can_send_sensory(const AppState& state) {
    return state.runtime.condition != RuntimeCondition::Rest &&
           state.accepted_stream_frames >= state.pause_until_sample &&
           state.accepted_stream_frames >= state.suppress_sensory_until_sample;
}

static void send_sequence_with_blinding(AppState& state,
                                        const std::string& sequence_name,
                                        int blinding_ms) {
    if (sequence_name.empty()) {
        return;
    }

    const maxlab::Status status = maxlab::sendSequence(sequence_name.c_str());
    if (status != maxlab::Status::MAXLAB_OK) {
        throw std::runtime_error("sendSequence failed for '" + sequence_name +
                                 "' with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    apply_readout_blinding(state, blinding_ms);
    std::cout << "[STIM] sendSequence(" << sequence_name << ")" << std::endl;
    state.logger.logEvent("sequence_sent",
                          state.accepted_stream_frames,
                          state.phase,
                          sequence_name);
}

static void maybe_send_sensory(AppState& state) {
    if (state.runtime.positions.empty() ||
        state.runtime.frequencies.empty()) {
        return;
    }

    const int pos_idx = state.pong_game.getSensoryStimZone();
    if (pos_idx < 0) {
        return;
    }

    const std::size_t position_index =
        std::min(static_cast<std::size_t>(pos_idx), state.runtime.positions.size() - 1);
    const std::size_t frequency_index =
        frequency_index_for_ball_x(state.pong_game.getBallX(),
                                   state.pong_game.getGameWidth(),
                                   state.runtime.frequencies.size());
    const int frequency_hz = state.runtime.frequencies[frequency_index];
    state.last_sensory_pos = static_cast<int>(position_index);
    state.last_sensory_hz = frequency_hz;
    const uint64_t interval_samples =
        samples_per_sensory_interval(state.runtime.sample_rate_hz, frequency_hz);

    if (!state.sensory_schedule_initialized) {
        state.next_sensory_sample = state.accepted_stream_frames + interval_samples;
        state.sensory_schedule_initialized = true;
        return;
    }

    if (!can_send_sensory(state)) {
        if (state.accepted_stream_frames >= state.next_sensory_sample) {
            state.next_sensory_sample = state.accepted_stream_frames + interval_samples;
        }
        return;
    }

    if (state.accepted_stream_frames < state.next_sensory_sample) {
        return;
    }

    const std::string& sequence_name = state.runtime.sequence_for(position_index, frequency_index);
    send_sequence_with_blinding(state, sequence_name, state.runtime.sensory_blinding_ms);
    state.logger.logEvent("sensory_stimulus",
                          state.accepted_stream_frames,
                          state.phase,
                          sequence_name);
    state.next_sensory_sample = state.accepted_stream_frames + interval_samples;
}

static void handle_hit(AppState& state) {
    std::cout << "[EVENT] BallHitPlayerPaddle bounces=" << state.pong_game.getBounces()
              << std::endl;
    state.logger.logEvent("hit",
                          state.accepted_stream_frames,
                          state.phase,
                          "bounces=" + std::to_string(state.pong_game.getBounces()));

    if (state.runtime.condition == RuntimeCondition::Stimulus) {
        send_sequence_with_blinding(state,
                                    state.runtime.hit_feedback_sequence,
                                    state.runtime.hit_feedback_blinding_ms);
        state.suppress_sensory_until_sample =
            state.accepted_stream_frames +
            samples_for_duration_ms(state.runtime.sample_rate_hz,
                                    state.runtime.hit_sensory_suppression_ms);
    }
}

static void handle_miss(AppState& state) {
    std::cout << "[EVENT] PlayerMissed bounces=" << state.pong_game.getBounces() << std::endl;
    state.logger.logEvent("miss",
                          state.accepted_stream_frames,
                          state.phase,
                          "bounces=" + std::to_string(state.pong_game.getBounces()) +
                              ",rally_id=" + std::to_string(state.rally_id));
    ++state.rally_id;

    if (state.runtime.condition == RuntimeCondition::Stimulus &&
        !state.runtime.miss_feedback_sequences.empty()) {
        std::uniform_int_distribution<std::size_t> dist(
            0, state.runtime.miss_feedback_sequences.size() - 1);
        send_sequence_with_blinding(state,
                                    state.runtime.miss_feedback_sequences[dist(state.rng)],
                                    state.runtime.miss_feedback_blinding_ms);
    }

    if (state.runtime.condition == RuntimeCondition::Stimulus ||
        state.runtime.condition == RuntimeCondition::Silent) {
        const int pause_ms = combined_miss_pause_ms(state.runtime.miss_feedback_duration_ms,
                                                    state.runtime.miss_pause_ms);
        state.pause_until_sample =
            state.accepted_stream_frames +
            samples_for_duration_ms(state.runtime.sample_rate_hz, pause_ms);
        state.logger.logEvent("pause",
                              state.accepted_stream_frames,
                              state.phase,
                              "duration_ms=" + std::to_string(pause_ms));
    }
}

static void handle_window(AppState& state) {
    state.detector.getCounts(&state.spike_counts);
    const MotorActivity motor = state.motor_decoder.decodeWindow(state.spike_counts);
    const uint64_t frame_start = state.window_start_frame;
    const uint64_t frame_end = state.accepted_stream_frames;

    if (state.accepted_stream_frames < state.pause_until_sample) {
        WindowSample sample;
        sample.phase = state.phase;
        sample.frame_start = frame_start;
        sample.frame_end = frame_end;
        sample.elapsed_ms = elapsed_ms_from_phase_start(
            state.phase_start_frame, frame_end, state.runtime.sample_rate_hz);
        sample.ball_x = state.pong_game.getBallX();
        sample.ball_y = state.pong_game.getBallY();
        sample.ball_vx = state.pong_game.getBallSpeedX();
        sample.ball_vy = state.pong_game.getBallSpeedY();
        sample.paddle_y = state.pong_game.getPaddle1Y();
        sample.raw_up = motor.raw_up;
        sample.raw_down = motor.raw_down;
        sample.corrected_up = motor.corrected_up;
        sample.corrected_down = motor.corrected_down;
        sample.up_gain = motor.up_gain;
        sample.down_gain = motor.down_gain;
        sample.sensory_pos = state.last_sensory_pos;
        sample.sensory_hz = state.last_sensory_hz;
        sample.rally_id = state.rally_id;
        state.logger.logWindow(sample);
        update_window(state);
        reset_window(state);
        return;
    }

    GameEvent event = GameEvent::None;
    if (!state.window_has_blinded_readout) {
        event = state.pong_game.update(static_cast<int>(motor.corrected_up),
                                       static_cast<int>(motor.corrected_down));
    }

    switch (event) {
        case GameEvent::None:
            break;
        case GameEvent::BallHitPlayerPaddle:
            handle_hit(state);
            break;
        case GameEvent::PlayerMissed:
            handle_miss(state);
            break;
    }

    update_window(state);
    WindowSample sample;
    sample.phase = state.phase;
    sample.frame_start = frame_start;
    sample.frame_end = frame_end;
    sample.elapsed_ms = elapsed_ms_from_phase_start(
        state.phase_start_frame, frame_end, state.runtime.sample_rate_hz);
    sample.ball_x = state.pong_game.getBallX();
    sample.ball_y = state.pong_game.getBallY();
    sample.ball_vx = state.pong_game.getBallSpeedX();
    sample.ball_vy = state.pong_game.getBallSpeedY();
    sample.paddle_y = state.pong_game.getPaddle1Y();
    sample.raw_up = motor.raw_up;
    sample.raw_down = motor.raw_down;
    sample.corrected_up = motor.corrected_up;
    sample.corrected_down = motor.corrected_down;
    sample.up_gain = motor.up_gain;
    sample.down_gain = motor.down_gain;
    sample.sensory_pos = state.last_sensory_pos;
    sample.sensory_hz = state.last_sensory_hz;
    sample.rally_id = state.rally_id;
    state.logger.logWindow(sample);
    reset_window(state);
}

static void handle_baseline_window(AppState& state) {
    const uint64_t frame_start = state.window_start_frame;
    const uint64_t frame_end = state.accepted_stream_frames;
    if (state.accepted_frames_in_window > 0) {
        state.detector.getCounts(&state.spike_counts);
        state.motor_decoder.observeBaselineWindow(state.spike_counts);
        const MotorActivity motor = state.motor_decoder.decodeWindow(state.spike_counts);
        WindowSample sample;
        sample.phase = state.phase;
        sample.frame_start = frame_start;
        sample.frame_end = frame_end;
        sample.elapsed_ms = elapsed_ms_from_phase_start(
            state.phase_start_frame, frame_end, state.runtime.sample_rate_hz);
        sample.ball_x = state.pong_game.getBallX();
        sample.ball_y = state.pong_game.getBallY();
        sample.ball_vx = state.pong_game.getBallSpeedX();
        sample.ball_vy = state.pong_game.getBallSpeedY();
        sample.paddle_y = state.pong_game.getPaddle1Y();
        sample.raw_up = motor.raw_up;
        sample.raw_down = motor.raw_down;
        sample.corrected_up = motor.corrected_up;
        sample.corrected_down = motor.corrected_down;
        sample.up_gain = motor.up_gain;
        sample.down_gain = motor.down_gain;
        sample.sensory_pos = -1;
        sample.sensory_hz = 0;
        sample.rally_id = state.rally_id;
        state.logger.logWindow(sample);
    }
    reset_window(state);
}

static void open_stream(StreamMode mode) {
    if (mode == StreamMode::Raw) {
        maxlab::verifyStatus(maxlab::DataStreamerRaw_open());
    } else {
        maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));
    }
}

static maxlab::Status close_stream(StreamMode mode) {
    return mode == StreamMode::Raw ? maxlab::DataStreamerRaw_close()
                                   : maxlab::DataStreamerFiltered_close();
}

static void log_close_failure(maxlab::Status status) {
    if (status != maxlab::Status::MAXLAB_OK) {
        std::cerr << "[STREAM] close failed with status "
                  << static_cast<int>(status) << std::endl;
    }
}

static void throw_on_receive_error(maxlab::Status status, StreamMode mode) {
    if (status == maxlab::Status::MAXLAB_OK || status == maxlab::Status::MAXLAB_NO_FRAME) {
        return;
    }

    throw std::runtime_error(std::string("DataStreamer ") +
                             stream_mode_to_string(mode) +
                             " receive failed with status " +
                             std::to_string(static_cast<int>(status)));
}

static bool receive_and_process_frame(AppState& state, const RunConfig& config) {
    if (config.runtime.stream_mode == StreamMode::Raw) {
        maxlab::RawFrameData frame_data;
        const maxlab::Status status = maxlab::DataStreamerRaw_receiveNextFrame(&frame_data);
        throw_on_receive_error(status, config.runtime.stream_mode);
        if (status == maxlab::Status::MAXLAB_NO_FRAME) {
            state.logger.noteDroppedFrame();
            return false;
        }
        ++state.total_frames_seen;
        if (frame_data.frameInfo.corrupted ||
            frame_data.frameInfo.well_id != config.target_well) {
            if (frame_data.frameInfo.corrupted) {
                state.logger.noteCorruptedFrame();
            }
            return false;
        }
        ++state.accepted_frames_in_window;
        ++state.accepted_stream_frames;
        if (readout_blinded(state)) {
            state.window_has_blinded_readout = true;
            return true;
        }
        state.detector.processFrame(frame_data.amplitudes, maxlab::RawFrameData::amplitudeCount);
        return true;
    }

    maxlab::FilteredFrameData frame_data;
    const maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frame_data);
    throw_on_receive_error(status, config.runtime.stream_mode);
    if (status == maxlab::Status::MAXLAB_NO_FRAME) {
        state.logger.noteDroppedFrame();
        return false;
    }
    ++state.total_frames_seen;
    if (frame_data.frameInfo.corrupted ||
        frame_data.frameInfo.well_id != config.target_well) {
        if (frame_data.frameInfo.corrupted) {
            state.logger.noteCorruptedFrame();
        }
        return false;
    }
    ++state.accepted_frames_in_window;
    ++state.accepted_stream_frames;
    if (readout_blinded(state)) {
        state.window_has_blinded_readout = true;
        return true;
    }
    for (uint64_t i = 0; i < frame_data.spikeCount; ++i) {
        const maxlab::SpikeEvent& spike = frame_data.spikeEvents[i];
        state.detector.addSpikeEvent(spike.channel);
    }
    return true;
}

static void log_runtime_config(const RunConfig& config) {
    const RuntimeConfig& runtime = config.runtime;
    std::cout << "[INFO] maxone_with_filter start"
              << " config_path=" << config.config_path
              << " target_well=" << static_cast<int>(config.target_well)
              << " show_gui=" << (config.show_gui ? 1 : 0)
              << " wait_for_sync=" << (config.wait_for_sync ? 1 : 0)
              << " condition=" << condition_to_string(runtime.condition)
              << " stream_mode=" << stream_mode_to_string(runtime.stream_mode)
              << " window_ms=" << runtime.window_ms
              << " pre_rest_seconds=" << runtime.pre_rest_seconds
              << " game_seconds=" << runtime.game_seconds
              << " sample_rate_hz=" << runtime.sample_rate_hz
              << " sensory_blinding_ms=" << runtime.sensory_blinding_ms
              << " hit_feedback_blinding_ms=" << runtime.hit_feedback_blinding_ms
              << " miss_feedback_blinding_ms=" << runtime.miss_feedback_blinding_ms
              << " miss_feedback_duration_ms=" << runtime.miss_feedback_duration_ms
              << " miss_pause_ms=" << runtime.miss_pause_ms
              << " hit_sensory_suppression_ms=" << runtime.hit_sensory_suppression_ms
              << " motor_gain_target_hz=" << runtime.motor_gain_target_hz
              << " channel_count=" << config.channel_count << std::endl;
    std::cout << "[INFO] channels"
              << " motor_up=" << runtime.motor_up_channels.size()
              << " motor_down=" << runtime.motor_down_channels.size()
              << " stim=" << runtime.stim_channels.size()
              << " positions=" << runtime.positions.size()
              << " frequencies=" << runtime.frequencies.size()
              << " miss_feedback=" << runtime.miss_feedback_sequences.size()
              << std::endl;
}

static int run_game_loop(const RunConfig& config, GameWindow* window) {
    bool stream_open = false;
    try {
        maxlab::checkVersions();
        open_stream(config.runtime.stream_mode);
        stream_open = true;
        std::cout << "[STREAM] Opened " << stream_mode_to_string(config.runtime.stream_mode)
                  << " stream; waiting 2 seconds for stabilization" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        AppState state(config, window);
        bool started = !config.wait_for_sync;
        state.logger.logEvent("phase_start", state.accepted_stream_frames, state.phase, "pre_rest");

        if (started) {
            std::cout << "[SYNC] Sync disabled, starting immediately" << std::endl;
        } else {
            std::cout << "[SYNC] Waiting for start signal while collecting baseline" << std::endl;
        }

        while (g_running.load() && !started) {
            started = poll_start_signal(config.wait_for_sync);
            if (receive_and_process_frame(state, config) &&
                state.accepted_frames_in_window >= state.samples_per_game_window) {
                handle_baseline_window(state);
            }
        }

        state.logger.logEvent("phase_end", state.accepted_stream_frames, state.phase, "pre_rest");
        state.phase = "game";
        state.phase_start_frame = state.accepted_stream_frames;
        state.logger.logEvent("phase_start", state.accepted_stream_frames, state.phase, "game");
        state.motor_decoder.freezeGains();
        reset_window(state);
        std::cout << "[SYNC] Motor gains frozen; entering game loop" << std::endl;

        while (g_running.load()) {
            if (receive_and_process_frame(state, config)) {
                if (state.accepted_frames_in_window >= state.samples_per_game_window) {
                    handle_window(state);
                }
                maybe_send_sensory(state);
            }
        }

        state.logger.logEvent("phase_end", state.accepted_stream_frames, state.phase, "game");
        state.logger.writeSummary(state.total_frames_seen, state.accepted_stream_frames);
        const maxlab::Status close_status = close_stream(config.runtime.stream_mode);
        log_close_failure(close_status);
        return 0;
    } catch (const std::exception& e) {
        if (stream_open) {
            log_close_failure(close_stream(config.runtime.stream_mode));
        }
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

static RunConfig parse_args(int argc, char* argv[]) {
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: maxone_with_filter <config_path> [target_well=0] [show_gui=1] "
            "[wait_for_sync=1]");
    }

    RunConfig config;
    config.config_path = argv[1];
    config.runtime = load_runtime_config(config.config_path);
    if (argc >= 3) {
        config.target_well = static_cast<uint8_t>(std::atoi(argv[2]));
    }
    if (argc >= 4) {
        config.show_gui = (std::atoi(argv[3]) != 0);
    }
    if (argc >= 5) {
        config.wait_for_sync = (std::atoi(argv[4]) != 0);
    }

    config.channel_count = infer_channel_count(config.runtime);
    config.detector.sample_rate_hz = config.runtime.sample_rate_hz;
    config.detector.threshold_multiplier = static_cast<float>(config.runtime.spike_threshold_std);
    config.detector.refractory_samples = std::max(
        1,
        static_cast<int>(std::lround(config.runtime.spike_refractory_period_ms *
                                     config.runtime.sample_rate_hz / 1000.0)));
    return config;
}

int main(int argc, char* argv[]) {
    try {
        const RunConfig config = parse_args(argc, argv);
        std::signal(SIGINT, on_sigint);
        log_runtime_config(config);

#ifdef USE_QT
        if (config.show_gui) {
            QApplication app(argc, argv);
            GameWindow window;
            window.show();
            std::atomic<int> worker_rc{0};

            std::thread worker([&]() {
                worker_rc = run_game_loop(config, &window);
                QMetaObject::invokeMethod(&app, "quit", Qt::QueuedConnection);
            });

            const int rc = app.exec();
            g_running = false;
            if (worker.joinable()) worker.join();
            if (worker_rc.load() != 0) {
                return worker_rc.load();
            }
            return rc;
        }
#else
        if (config.show_gui) {
            std::cerr << "Qt UI is disabled (build without USE_QT); running headless."
                      << std::endl;
        }
#endif

        return run_game_loop(config, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "Startup failed: " << e.what() << std::endl;
        return 1;
    }
}
