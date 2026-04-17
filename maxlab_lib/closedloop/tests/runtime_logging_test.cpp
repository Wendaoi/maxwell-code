#include "runtime_logging.h"

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

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    expect(static_cast<bool>(input), "failed to open output file");
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

void test_runtime_logger_writes_events_windows_and_summary() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "runtime_logging_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path events = dir / "events.jsonl";
    const std::filesystem::path windows = dir / "windows.csv";
    const std::filesystem::path summary = dir / "summary.json";
    std::filesystem::remove(events);
    std::filesystem::remove(windows);
    std::filesystem::remove(summary);

    RuntimeLogger logger(events.string(), windows.string(), summary.string());
    logger.logEvent("phase_start", 100, "pre_rest", "condition=REST");

    WindowSample sample;
    sample.phase = "game";
    sample.frame_start = 200;
    sample.frame_end = 399;
    sample.elapsed_ms = 10;
    sample.ball_x = 240;
    sample.ball_y = 220;
    sample.ball_vx = -1.5;
    sample.ball_vy = 0.5;
    sample.paddle_y = 180;
    sample.raw_up = 3;
    sample.raw_down = 2;
    sample.corrected_up = 1.5;
    sample.corrected_down = 1.0;
    sample.up_gain = 2.0;
    sample.down_gain = 2.0;
    sample.sensory_pos = 4;
    sample.sensory_hz = 40;
    sample.rally_id = 7;
    logger.logWindow(sample);
    logger.noteCorruptedFrame();
    logger.noteDroppedFrame();
    logger.writeSummary(400, 360);

    const std::string event_text = read_file(events);
    const std::string window_text = read_file(windows);
    const std::string summary_text = read_file(summary);

    expect(event_text.find("\"event\":\"phase_start\"") != std::string::npos, "event type missing");
    expect(event_text.find("\"phase\":\"pre_rest\"") != std::string::npos, "event phase missing");
    expect(window_text.find("phase,frame_start,frame_end,elapsed_ms") == 0, "csv header missing");
    expect(window_text.find("game,200,399,10,240,220") != std::string::npos, "csv row missing");
    expect(summary_text.find("\"accepted_frames\":360") != std::string::npos, "summary accepted frames");
    expect(summary_text.find("\"corrupted_frames\":1") != std::string::npos, "summary corrupted frames");
    expect(summary_text.find("\"dropped_frames\":1") != std::string::npos, "summary dropped frames");
}

}  // namespace

int main() {
    try {
        test_runtime_logger_writes_events_windows_and_summary();
    } catch (const std::exception& e) {
        std::cerr << "runtime_logging_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "runtime_logging_test passed\n";
    return 0;
}
