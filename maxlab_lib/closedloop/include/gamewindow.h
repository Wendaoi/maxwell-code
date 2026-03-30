#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#ifdef USE_QT
#include <atomic>
#include <QWidget>

class GameWindow : public QWidget {
public:
    explicit GameWindow(QWidget* parent = nullptr);
    void setState(float cartX, float poleAngleRad, float timeBalancedSeconds, float forceNewtons);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::atomic<float> cart_x_;
    std::atomic<float> pole_angle_rad_;
    std::atomic<float> time_balanced_seconds_;
    std::atomic<float> force_newtons_;
};
#else
class GameWindow {
public:
    explicit GameWindow(void* parent = nullptr) { (void)parent; }
    void setState(float, float, float, float) {}
};
#endif

#endif // GAMEWINDOW_H
