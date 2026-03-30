#ifndef CARTPOLE_TASK_H
#define CARTPOLE_TASK_H

#include <random>

class CartpoleTask {
public:
    explicit CartpoleTask(double timestep_seconds = 0.2);

    void reset();
    bool step(double force_newtons);
    bool isTerminal() const;

    double getCartPosition() const { return x_; }
    double getCartVelocity() const { return x_dot_; }
    double getPoleAngleRad() const { return theta_; }
    double getPoleAngularVelocity() const { return theta_dot_; }
    double getTimeBalanced() const { return time_balanced_seconds_; }
    double getLastForce() const { return last_force_newtons_; }
    double getTimestepSeconds() const { return timestep_seconds_; }

private:
    double timestep_seconds_;
    double gravity_;
    double masscart_;
    double masspole_;
    double total_mass_;
    double length_;
    double polemass_length_;
    double theta_threshold_radians_;

    double x_;
    double x_dot_;
    double theta_;
    double theta_dot_;
    double time_balanced_seconds_;
    double last_force_newtons_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> theta_init_dist_;
    std::uniform_real_distribution<double> theta_dot_init_dist_;
};

#endif // CARTPOLE_TASK_H
