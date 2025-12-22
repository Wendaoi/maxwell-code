#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#ifdef USE_QT
#include <atomic>
#include <QWidget>

class GameWindow : public QWidget {
public:
    explicit GameWindow(QWidget* parent = nullptr);
    void setState(int paddleY, int ballX, int ballY, int paddleHeight);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::atomic<int> paddle_y_;
    std::atomic<int> ball_x_;
    std::atomic<int> ball_y_;
    std::atomic<int> paddle_h_;
};
#else
class GameWindow {
public:
    explicit GameWindow(void* parent = nullptr) { (void)parent; }
    void setState(int, int, int, int) {}
};
#endif

#endif // GAMEWINDOW_H
