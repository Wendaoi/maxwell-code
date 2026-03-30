#ifdef USE_QT
#include "gamewindow.h"

#include <QPainter>
#include <QTimer>

GameWindow::GameWindow(QWidget* parent)
    : QWidget(parent),
      cart_x_(0.0f),
      pole_angle_rad_(0.0f),
      time_balanced_seconds_(0.0f),
      force_newtons_(0.0f) {
    setWindowTitle("Cartpole Viewer");
    resize(640, 480);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, QOverload<>::of(&QWidget::update));
    timer->start(16); // ~60 FPS
}

void GameWindow::setState(float cartX, float poleAngleRad, float timeBalancedSeconds, float forceNewtons) {
    cart_x_.store(cartX, std::memory_order_relaxed);
    pole_angle_rad_.store(poleAngleRad, std::memory_order_relaxed);
    time_balanced_seconds_.store(timeBalancedSeconds, std::memory_order_relaxed);
    force_newtons_.store(forceNewtons, std::memory_order_relaxed);
}

void GameWindow::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), QColor(18, 20, 24));

    const int w = width();
    const int h = height();
    const int ground_y = h - 70;
    const float cart_x = cart_x_.load(std::memory_order_relaxed);
    const float pole_angle = pole_angle_rad_.load(std::memory_order_relaxed);
    const float time_balanced = time_balanced_seconds_.load(std::memory_order_relaxed);
    const float force_newtons = force_newtons_.load(std::memory_order_relaxed);

    painter.setPen(QPen(QColor(90, 95, 105), 2));
    painter.drawLine(0, ground_y, w, ground_y);

    const float cart_range = static_cast<float>(w) * 0.35f;
    const int cart_width = 90;
    const int cart_height = 28;
    const int cart_center_x = static_cast<int>(w / 2.0f + cart_x * cart_range);
    const QRect cart_rect(cart_center_x - cart_width / 2, ground_y - cart_height, cart_width, cart_height);

    painter.setBrush(QColor(84, 161, 255));
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(cart_rect, 6, 6);

    const QPoint pivot(cart_center_x, ground_y - cart_height);
    const int pole_length = 170;
    const QPoint pole_tip(
        static_cast<int>(pivot.x() + std::sin(pole_angle) * pole_length),
        static_cast<int>(pivot.y() - std::cos(pole_angle) * pole_length));

    painter.setPen(QPen(QColor(240, 200, 80), 8, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(pivot, pole_tip);

    painter.setBrush(QColor(255, 244, 214));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(pivot, 8, 8);

    painter.setPen(QColor(230, 230, 230));
    painter.drawText(16, 28, QString("time balanced: %1 s").arg(time_balanced, 0, 'f', 1));
    painter.drawText(16, 50, QString("force: %1 N").arg(force_newtons, 0, 'f', 2));
}
#endif
