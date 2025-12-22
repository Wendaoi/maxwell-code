#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "gamewindow.h"
#include "maxlab/maxlab.h"
#include "spike_detection.h"
#include "threads/ponggame.h"

#ifdef USE_QT
#include <QApplication>
#include <QMetaObject>
#endif

using clock = std::chrono::steady_clock;

static std::atomic<bool> g_running{true};

static constexpr float kGameWidth = 640.0f;
static constexpr float kGameHeight = 480.0f;

static const std::array<const char*, 8> kStimPositions = {
    "pos0",
    "pos1",
    "pos2",
    "pos3",
    "pos4",
    "pos5",
    "pos6",
    "pos7",
};
static const std::array<int, 10> kStimFrequencies = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40};
static const char* kSequenceOnHit = "hit_feedback";
static const char* kSequenceOnMiss = "miss_feedback";

struct RunConfig {
    uint8_t target_well = 0;
    int window_ms = 5;
    uint64_t blanking_frames_after_trigger = 8000;
    bool show_gui = false;
    std::size_t channel_count = 1024;
    SpikeDetectorConfig detector;
};

struct AppState {
    AppState(const RunConfig& config, GameWindow* ui_window)
        : window_start(clock::now()),
          window_len(std::chrono::milliseconds(config.window_ms)),
          blanking(0),
          blanking_frames_after_trigger(config.blanking_frames_after_trigger),
          pong_game(),
          detector(config.channel_count, config.detector),
          window(ui_window) {
        spike_counts.resize(config.channel_count, 0);
    }

    clock::time_point window_start;
    std::chrono::milliseconds window_len;
    uint64_t blanking;
    uint64_t blanking_frames_after_trigger;
    PongGame pong_game;
    SpikeDetector detector;
    GameWindow* window;
    std::vector<std::uint32_t> spike_counts;
};

static void on_sigint(int) {
    g_running = false;
}

static void reset_window(AppState& state, clock::time_point now) {
    state.window_start = now;
    state.detector.resetCounts();
}

static int channel_to_quarter(std::size_t channel, std::size_t channel_count) {
    if (channel_count < 4) return -1;
    const std::size_t quarter_size = channel_count / 4;
    if (quarter_size == 0) return -1;
    int q = static_cast<int>(channel / quarter_size);
    if (q > 3) q = 3;
    return q;
}

static void handle_window(AppState& state, clock::time_point now) {
    state.detector.getCounts(&state.spike_counts);

    std::array<uint64_t, 4> quarter_counts{0, 0, 0, 0};
    for (std::size_t ch = 0; ch < state.spike_counts.size(); ++ch) {
        const int q = channel_to_quarter(ch, state.spike_counts.size());
        if (q >= 0 && q < 4) {
            quarter_counts[static_cast<std::size_t>(q)] += state.spike_counts[ch];
        }
    }

    const uint64_t sum_up = quarter_counts[0] + quarter_counts[1];   // Q0+Q1
    const uint64_t sum_down = quarter_counts[2] + quarter_counts[3]; // Q2+Q3

    GameEvent event = state.pong_game.update(static_cast<int>(sum_up), static_cast<int>(sum_down));
    if (event == GameEvent::None) {
        if (state.pong_game.getCondition() != ExperimentCondition::Rest) {
            const float ball_x = static_cast<float>(state.pong_game.getBallX());
            const float ball_y = static_cast<float>(state.pong_game.getBallY());
            const float norm_x = std::clamp(ball_x / kGameWidth, 0.0f, 1.0f);
            const float norm_y = std::clamp(ball_y / kGameHeight, 0.0f, 1.0f);

            int pos_idx = static_cast<int>(norm_y * 7.99f);
            int freq_idx = static_cast<int>((1.0f - norm_x) * 9.99f);
            pos_idx = std::clamp(pos_idx, 0, static_cast<int>(kStimPositions.size() - 1));
            freq_idx = std::clamp(freq_idx, 0, static_cast<int>(kStimFrequencies.size() - 1));

            const std::size_t pos_index = static_cast<std::size_t>(pos_idx);
            const std::size_t freq_index = static_cast<std::size_t>(freq_idx);
            std::string seq_name = std::string(kStimPositions[pos_index]) + "_" +
                                   std::to_string(kStimFrequencies[freq_index]) + "hz";
            maxlab::verifyStatus(maxlab::sendSequence(seq_name.c_str()));
        }
    } else if (event == GameEvent::BallHitPlayerPaddle) {
        if (state.blanking == 0) {
            maxlab::verifyStatus(maxlab::sendSequence(kSequenceOnHit));
            state.blanking = state.blanking_frames_after_trigger;
        }
    } else if (event == GameEvent::PlayerMissed) {
        if (state.blanking == 0) {
            maxlab::verifyStatus(maxlab::sendSequence(kSequenceOnMiss));
            state.blanking = state.blanking_frames_after_trigger;
        }
    }

    if (state.window != nullptr) {
        state.window->setState(
            state.pong_game.getPaddle1Y(),
            state.pong_game.getBallX(),
            state.pong_game.getBallY(),
            state.pong_game.getPaddleHeight());
    }

    reset_window(state, now);
}

struct FrameSamplesView {
    const float* samples = nullptr;
    std::size_t channel_count = 0;
};

static bool extract_frame_samples(const maxlab::FilteredFrameData& frame_data,
                                  FrameSamplesView* out) {
#ifdef MAXONE_USE_RAW_SAMPLES
    // TODO: Wire the raw/filtered sample pointer and channel count from maxlab headers.
    out->samples = frame_data.analogSamples;
    out->channel_count = frame_data.frameInfo.channelCount;
    return (out->samples != nullptr && out->channel_count > 0);
#else
    (void)frame_data;
    (void)out;
    return false;
#endif
}

static int run_game_loop(const RunConfig& config, GameWindow* window) {
    try {
        maxlab::checkVersions();
        maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));

        // 允许数据流稳定
        std::this_thread::sleep_for(std::chrono::seconds(2));

        AppState state(config, window);

        while (g_running.load()) {
            maxlab::FilteredFrameData frame_data;
            const maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frame_data);

            if (status == maxlab::Status::MAXLAB_NO_FRAME) {
                auto now = clock::now();
                if (now - state.window_start >= state.window_len) {
                    handle_window(state, now);
                }
                continue;
            }

            if (status != maxlab::Status::MAXLAB_OK) {
                continue;
            }

            if (frame_data.frameInfo.corrupted) continue;
            if (frame_data.frameInfo.well_id != config.target_well) continue;

            if (state.blanking > 0) --state.blanking;

            FrameSamplesView samples;
            if (extract_frame_samples(frame_data, &samples)) {
                state.detector.processFrame(samples.samples, samples.channel_count);
            } else {
                for (uint64_t i = 0; i < frame_data.spikeCount; ++i) {
                    const maxlab::SpikeEvent& sp = frame_data.spikeEvents[i];
                    state.detector.addSpikeEvent(sp.channel);
                }
            }

            auto now = clock::now();
            if (now - state.window_start >= state.window_len) {
                handle_window(state, now);
            }
        }

        maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
        return 0;
    } catch (const std::exception& e) {
        (void)maxlab::DataStreamerFiltered_close();
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

static RunConfig parse_args(int argc, char* argv[]) {
    RunConfig config;
    if (argc >= 2) config.target_well = static_cast<uint8_t>(std::atoi(argv[1]));
    if (argc >= 3) config.window_ms = (std::max)(1, std::atoi(argv[2]));
    if (argc >= 4) {
        config.blanking_frames_after_trigger =
            static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10));
    }
    if (argc >= 5) config.show_gui = (std::atoi(argv[4]) != 0);
    if (argc >= 6) config.detector.sample_rate_hz = std::atof(argv[5]);
    if (argc >= 7) config.detector.threshold_multiplier = static_cast<float>(std::atof(argv[6]));
    if (argc >= 8) config.detector.min_threshold = static_cast<float>(std::atof(argv[7]));
    if (argc >= 9) config.detector.refractory_samples = std::atoi(argv[8]);
    if (argc >= 10) config.channel_count = static_cast<std::size_t>(std::atoi(argv[9]));
    return config;
}

int main(int argc, char* argv[]) {
    // argv[1] = targetWell (默认 0)
    // argv[2] = window_ms (默认 5)
    // argv[3] = blanking_frames (默认 8000)
    // argv[4] = show_gui (默认 0)
    // argv[5] = sample_rate_hz (默认 20000)
    // argv[6] = threshold_multiplier (默认 5.0)
    // argv[7] = min_threshold (默认 -20)
    // argv[8] = refractory_samples (默认 1000)
    // argv[9] = channel_count (默认 1024)
    const RunConfig config = parse_args(argc, argv);

    std::signal(SIGINT, on_sigint);

#ifdef USE_QT
    if (config.show_gui) {
        QApplication app(argc, argv);
        GameWindow window;
        window.show();

        std::thread worker([&]() {
            run_game_loop(config, &window);
            QMetaObject::invokeMethod(&app, "quit", Qt::QueuedConnection);
        });

        const int rc = app.exec();
        g_running = false;
        if (worker.joinable()) worker.join();
        return rc;
    }
#else
    if (config.show_gui) {
        std::cerr << "Qt UI is disabled (build without USE_QT); running headless." << std::endl;
    }
#endif

    return run_game_loop(config, nullptr);
}
