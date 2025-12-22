#ifndef FILTER_H
#define FILTER_H

class FirstOrderLowpassFilter {
public:
    FirstOrderLowpassFilter(double cutoff_hz, double sample_rate_hz);

    void reset();
    float filterOne(float input);

private:
    double b0_;
    double b1_;
    double a1_;
    float x1_;
    float y1_;
};

class SecondOrderHighpassFilter {
public:
    SecondOrderHighpassFilter(double cutoff_hz, double q, double sample_rate_hz);

    void reset();
    float filterOne(float input);

private:
    double b0_;
    double b1_;
    double b2_;
    double a1_;
    double a2_;
    float x1_;
    float x2_;
    float y1_;
    float y2_;
};

#endif // FILTER_H
