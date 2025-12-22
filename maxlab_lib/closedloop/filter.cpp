#include "filter.h"

#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

FirstOrderLowpassFilter::FirstOrderLowpassFilter(double cutoff_hz, double sample_rate_hz)
    : b0_(0.0),
      b1_(0.0),
      a1_(0.0),
      x1_(0.0f),
      y1_(0.0f) {
    const double fs = sample_rate_hz;
    const double omega_c = 2.0 * fs * std::tan(kPi * cutoff_hz / fs);
    const double k = 2.0 * fs;

    // Analog: H(s) = omega_c / (s + omega_c) (1st order Bessel low-pass)
    const double a0 = omega_c;
    const double a1 = 1.0;
    const double b0 = omega_c;
    const double b1 = 0.0;

    const double A0 = a1 * k + a0;
    const double A1 = a0 - a1 * k;
    const double B0 = b1 * k + b0;
    const double B1 = b0 - b1 * k;

    b0_ = B0 / A0;
    b1_ = B1 / A0;
    a1_ = A1 / A0;
}

void FirstOrderLowpassFilter::reset() {
    x1_ = 0.0f;
    y1_ = 0.0f;
}

float FirstOrderLowpassFilter::filterOne(float input) {
    const double y = b0_ * input + b1_ * x1_ - a1_ * y1_;
    x1_ = input;
    y1_ = static_cast<float>(y);
    return y1_;
}

SecondOrderHighpassFilter::SecondOrderHighpassFilter(double cutoff_hz,
                                                     double q,
                                                     double sample_rate_hz)
    : b0_(0.0),
      b1_(0.0),
      b2_(0.0),
      a1_(0.0),
      a2_(0.0),
      x1_(0.0f),
      x2_(0.0f),
      y1_(0.0f),
      y2_(0.0f) {
    (void)q; // Bessel high-pass design uses fixed damping.

    const double fs = sample_rate_hz;
    const double omega_c = 2.0 * fs * std::tan(kPi * cutoff_hz / fs);
    const double k = 2.0 * fs;

    // Analog 2nd order Bessel low-pass: H(s) = 3 / (s^2 + 3s + 3)
    // Low-pass to high-pass transform: H(s) = 3 s^2 / (3 s^2 + 3 omega_c s + omega_c^2)
    const double a2 = 3.0;
    const double a1 = 3.0 * omega_c;
    const double a0 = omega_c * omega_c;
    const double b2 = 3.0;
    const double b1 = 0.0;
    const double b0 = 0.0;

    const double A0 = a2 * k * k + a1 * k + a0;
    const double A1 = 2.0 * (a0 - a2 * k * k);
    const double A2 = a2 * k * k - a1 * k + a0;
    const double B0 = b2 * k * k + b1 * k + b0;
    const double B1 = 2.0 * (b0 - b2 * k * k);
    const double B2 = b2 * k * k - b1 * k + b0;

    b0_ = B0 / A0;
    b1_ = B1 / A0;
    b2_ = B2 / A0;
    a1_ = A1 / A0;
    a2_ = A2 / A0;
}

void SecondOrderHighpassFilter::reset() {
    x1_ = 0.0f;
    x2_ = 0.0f;
    y1_ = 0.0f;
    y2_ = 0.0f;
}

float SecondOrderHighpassFilter::filterOne(float input) {
    const double y = b0_ * input + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_;
    x1_ = input;
    y2_ = y1_;
    y1_ = static_cast<float>(y);
    return y1_;
}
