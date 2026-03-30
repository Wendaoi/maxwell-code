#ifndef TRAINING_CONTROLLER_H
#define TRAINING_CONTROLLER_H

#include <cstdint>
#include <cstddef>
#include <deque>
#include <random>
#include <string>
#include <vector>

enum class ClosedLoopMode {
    CycledAdaptive,
    ContinuousAdaptive,
};

struct TrainingDecision {
    bool delivered = false;
    int pattern_index = -1;
    std::string sequence_name;
    double mean_5 = 0.0;
    double mean_20 = 0.0;
};

class TrainingController {
public:
    TrainingController(std::vector<std::string> pattern_names,
                       double value_alpha = 0.3,
                       double eligibility_gamma = 0.3,
                       double min_reward = 10.0,
                       std::uint32_t random_seed = 12345);

    TrainingDecision onEpisodeEnd(double reward_seconds);
    std::size_t patternCount() const { return pattern_names_.size(); }
    const std::vector<double>& values() const { return values_; }

private:
    double movingAverage(std::size_t count) const;
    void updateValues(double reward_seconds);
    void decayEligibility();
    int samplePatternIndex();

    std::vector<std::string> pattern_names_;
    std::vector<double> values_;
    std::vector<double> eligibility_;
    double value_alpha_;
    double eligibility_gamma_;
    double min_reward_;
    std::deque<double> recent_rewards_;
    std::mt19937 rng_;
};

#endif // TRAINING_CONTROLLER_H
