#ifndef RUNTIME_LOGGING_H
#define RUNTIME_LOGGING_H

#include <cstdint>
#include <fstream>
#include <string>

struct WindowSample {
    std::string phase;
    std::uint64_t frame_start = 0;
    std::uint64_t frame_end = 0;
    int elapsed_ms = 0;
    int ball_x = 0;
    int ball_y = 0;
    double ball_vx = 0.0;
    double ball_vy = 0.0;
    int paddle_y = 0;
    std::uint32_t raw_up = 0;
    std::uint32_t raw_down = 0;
    double corrected_up = 0.0;
    double corrected_down = 0.0;
    double up_gain = 1.0;
    double down_gain = 1.0;
    int sensory_pos = -1;
    int sensory_hz = 0;
    std::uint64_t rally_id = 0;
};

class RuntimeLogger {
public:
    RuntimeLogger(std::string event_path, std::string window_path, std::string summary_path);

    void logEvent(const std::string& event,
                  std::uint64_t frame,
                  const std::string& phase,
                  const std::string& detail);
    void logWindow(const WindowSample& sample);
    void noteCorruptedFrame();
    void noteDroppedFrame();
    void writeSummary(std::uint64_t total_frames, std::uint64_t accepted_frames);

private:
    static std::string escapeJson(const std::string& value);
    void ensureWindowHeader();

    std::string summary_path_;
    std::ofstream events_;
    std::ofstream windows_;
    bool wrote_window_header_ = false;
    std::uint64_t corrupted_frames_ = 0;
    std::uint64_t dropped_frames_ = 0;
};

#endif  // RUNTIME_LOGGING_H
