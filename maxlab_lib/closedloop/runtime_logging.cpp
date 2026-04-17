#include "runtime_logging.h"

#include <stdexcept>
#include <utility>

RuntimeLogger::RuntimeLogger(std::string event_path,
                             std::string window_path,
                             std::string summary_path)
    : summary_path_(std::move(summary_path)),
      events_(event_path),
      windows_(window_path) {
    if (!events_) {
        throw std::runtime_error("failed to open runtime event log");
    }
    if (!windows_) {
        throw std::runtime_error("failed to open runtime window log");
    }
}

std::string RuntimeLogger::escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(c);
                break;
        }
    }
    return escaped;
}

void RuntimeLogger::logEvent(const std::string& event,
                             std::uint64_t frame,
                             const std::string& phase,
                             const std::string& detail) {
    events_ << "{\"event\":\"" << escapeJson(event)
            << "\",\"frame\":" << frame
            << ",\"phase\":\"" << escapeJson(phase)
            << "\",\"detail\":\"" << escapeJson(detail)
            << "\"}\n";
    events_.flush();
}

void RuntimeLogger::ensureWindowHeader() {
    if (wrote_window_header_) {
        return;
    }
    windows_ << "phase,frame_start,frame_end,elapsed_ms,ball_x,ball_y,ball_vx,ball_vy,"
             << "paddle_y,raw_up,raw_down,corrected_up,corrected_down,up_gain,down_gain,"
             << "sensory_pos,sensory_hz,rally_id\n";
    wrote_window_header_ = true;
}

void RuntimeLogger::logWindow(const WindowSample& sample) {
    ensureWindowHeader();
    windows_ << sample.phase << ','
             << sample.frame_start << ','
             << sample.frame_end << ','
             << sample.elapsed_ms << ','
             << sample.ball_x << ','
             << sample.ball_y << ','
             << sample.ball_vx << ','
             << sample.ball_vy << ','
             << sample.paddle_y << ','
             << sample.raw_up << ','
             << sample.raw_down << ','
             << sample.corrected_up << ','
             << sample.corrected_down << ','
             << sample.up_gain << ','
             << sample.down_gain << ','
             << sample.sensory_pos << ','
             << sample.sensory_hz << ','
             << sample.rally_id << '\n';
    windows_.flush();
}

void RuntimeLogger::noteCorruptedFrame() {
    ++corrupted_frames_;
}

void RuntimeLogger::noteDroppedFrame() {
    ++dropped_frames_;
}

void RuntimeLogger::writeSummary(std::uint64_t total_frames, std::uint64_t accepted_frames) {
    std::ofstream summary(summary_path_);
    if (!summary) {
        throw std::runtime_error("failed to open runtime quality summary");
    }
    summary << "{\n"
            << "  \"total_frames\":" << total_frames << ",\n"
            << "  \"accepted_frames\":" << accepted_frames << ",\n"
            << "  \"corrupted_frames\":" << corrupted_frames_ << ",\n"
            << "  \"dropped_frames\":" << dropped_frames_ << "\n"
            << "}\n";
}
