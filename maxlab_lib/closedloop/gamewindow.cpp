#ifdef USE_QT
#include "gamewindow.h"

#include <QPainter>
#include <QTimer>

GameWindow::GameWindow(QWidget* parent)
    : QWidget(parent),
      paddle_y_(0),
      ball_x_(0),
      ball_y_(0),
      paddle_h_(0) {
    setWindowTitle("PongGame Viewer");
    resize(640, 480);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, QOverload<>::of(&QWidget::update));
    timer->start(16); // ~60 FPS
}

void GameWindow::setState(int paddleY, int ballX, int ballY, int paddleHeight) {
    paddle_y_.store(paddleY, std::memory_order_relaxed);
    ball_x_.store(ballX, std::memory_order_relaxed);
    ball_y_.store(ballY, std::memory_order_relaxed);
    paddle_h_.store(paddleHeight, std::memory_order_relaxed);
}

void GameWindow::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), QColor(18, 20, 24));

    const int w = width();
    const int h = height();
    const float sx = static_cast<float>(w) / 640.0f;
    const float sy = static_cast<float>(h) / 480.0f;

    const int paddleY = static_cast<int>(paddle_y_.load(std::memory_order_relaxed) * sy);
    const int paddleH = static_cast<int>(paddle_h_.load(std::memory_order_relaxed) * sy);
    const int ballX = static_cast<int>(ball_x_.load(std::memory_order_relaxed) * sx);
    const int ballY = static_cast<int>(ball_y_.load(std::memory_order_relaxed) * sy);

    painter.setPen(QPen(QColor(230, 230, 230), 4));
    painter.drawLine(8, paddleY, 8, paddleY + paddleH);

    painter.setBrush(QColor(240, 200, 80));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(QPoint(ballX, ballY), 6, 6);
}
#endif
