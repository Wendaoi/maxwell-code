#ifndef RUNTIME_TIMING_H
#define RUNTIME_TIMING_H

#include <cstdint>

std::uint64_t samples_for_duration_ms(double sample_rate_hz, int duration_ms);
std::uint64_t samples_per_window(double sample_rate_hz, int window_ms);
std::uint64_t samples_per_sensory_interval(double sample_rate_hz, int frequency_hz);
std::uint64_t extend_blinding_until(std::uint64_t current_frame,
                                    std::uint64_t duration_samples,
                                    std::uint64_t active_until_sample);
int elapsed_ms_from_phase_start(std::uint64_t phase_start_frame,
                                std::uint64_t current_frame,
                                double sample_rate_hz);
int combined_miss_pause_ms(int feedback_duration_ms, int rest_pause_ms);

#endif  // RUNTIME_TIMING_H
