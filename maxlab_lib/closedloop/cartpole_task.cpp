#include "cartpole_task.h"

#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

CartpoleTask::CartpoleTask(double timestep_seconds)
    : timestep_seconds_(timestep_seconds),
      gravity_(9.8),
      masscart_(1.0),
      masspole_(0.1),
      total_mass_(masscart_ + masspole_),
      length_(0.5),
      polemass_length_(masspole_ * length_),
      theta_threshold_radians_(16.0 * kPi / 180.0),
      x_(0.0),
      x_dot_(0.0),
      theta_(0.0),
      theta_dot_(0.0),
      time_balanced_seconds_(0.0),
      last_force_newtons_(0.0),
      rng_(std::random_device{}()),
      theta_init_dist_(-0.05, 0.05),
      theta_dot_init_dist_(-0.05, 0.05) {
    reset();
}

void CartpoleTask::reset() {
    x_ = 0.0;
    x_dot_ = 0.0;
    theta_ = theta_init_dist_(rng_);
    theta_dot_ = theta_dot_init_dist_(rng_);
    time_balanced_seconds_ = 0.0;
    last_force_newtons_ = 0.0;
}

bool CartpoleTask::step(double force_newtons) {
    last_force_newtons_ = force_newtons;

    const double costheta = std::cos(theta_);
    const double sintheta = std::sin(theta_);
    const double temp = (force_newtons + polemass_length_ * theta_dot_ * theta_dot_ * sintheta) / total_mass_;
    const double thetaacc =
        (gravity_ * sintheta - costheta * temp) /
        (length_ * (4.0 / 3.0 - masspole_ * costheta * costheta / total_mass_));
    const double xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

    x_ += timestep_seconds_ * x_dot_;
    x_dot_ += timestep_seconds_ * xacc;
    theta_ += timestep_seconds_ * theta_dot_;
    theta_dot_ += timestep_seconds_ * thetaacc;
    time_balanced_seconds_ += timestep_seconds_;

    return isTerminal();
}

bool CartpoleTask::isTerminal() const {
    return std::fabs(theta_) > theta_threshold_radians_;
}
