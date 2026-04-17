#include "gamewindow.h"
#include "ponggame.h"

#include <QApplication>
#include <QColor>
#include <QImage>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool color_is_close(const QColor& actual,
                    const QColor& expected,
                    int tolerance = 8) {
    return std::abs(actual.red() - expected.red()) <= tolerance &&
           std::abs(actual.green() - expected.green()) <= tolerance &&
           std::abs(actual.blue() - expected.blue()) <= tolerance;
}

void test_default_window_size_matches_game_geometry() {
    PongGame game;
    GameWindow window;

    expect(window.width() == game.getGameWidth(),
           "default GameWindow width should match PongGame geometry");
    expect(window.height() == game.getGameHeight(),
           "default GameWindow height should match PongGame geometry");
}

void test_ball_renders_using_game_geometry_scale() {
    PongGame game;
    GameWindow window;
    window.resize(game.getGameWidth(), game.getGameHeight());
    window.setState(120, 120, 240, game.getPaddleHeight());
    window.show();
    QApplication::processEvents();

    QImage image(window.size(), QImage::Format_ARGB32);
    image.fill(Qt::transparent);
    window.render(&image);

    const QColor background(18, 20, 24);
    const QColor ball(240, 200, 80);

    expect(color_is_close(image.pixelColor(120, 240), ball),
           "ball should render at the logical x position when widget matches game size");
    expect(color_is_close(image.pixelColor(90, 240), background),
           "old 640px scaling should not shift the ball left");
}

}  // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    try {
        test_default_window_size_matches_game_geometry();
        test_ball_renders_using_game_geometry_scale();
    } catch (const std::exception& e) {
        std::cerr << "gamewindow_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "gamewindow_test passed\n";
    return 0;
}
