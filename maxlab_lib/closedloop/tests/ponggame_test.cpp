#define private public
#include "ponggame.h"
#undef private

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_close(float actual, float expected, float tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void test_centered_default_state_maps_to_center_zone() {
    PongGame game;

    expect(game.getSensoryStimZone() == 4, "centered default state should map to zone 4");
}

void test_sensory_zone_depends_on_absolute_ball_y_not_paddle_position() {
    PongGame upper_paddle;
    upper_paddle.ballY = 90.0f;
    upper_paddle.paddleY = 0;

    PongGame lower_paddle;
    lower_paddle.ballY = 90.0f;
    lower_paddle.paddleY = lower_paddle.getGameHeight() - lower_paddle.getPaddleHeight();

    expect(upper_paddle.getSensoryStimZone() == 1,
           "absolute ballY=90 should map to zone 1");
    expect(lower_paddle.getSensoryStimZone() == 1,
           "sensory zone should not change when only paddle position changes");
}

void test_rest_condition_suppresses_sensory_zone() {
    PongGame game;
    game.setCondition(ExperimentCondition::Rest);

    expect(game.getSensoryStimZone() == -1, "Rest condition should return no sensory zone");
}

void test_serve_speed_sets_five_second_round_trip() {
    std::srand(1);
    PongGame game;

    expect_close(game.getBallSpeedX(), 1.92f, 0.0001f,
                 "serve x speed should cross the 480px screen in 2.5s at 10ms/update");
}

void test_serve_y_speed_ratio_is_randomized_in_required_bands() {
    std::srand(1);

    int low_band_count = 0;
    int high_band_count = 0;
    for (int i = 0; i < 1000; ++i) {
        PongGame game;
        const float x_speed = std::fabs(game.getBallSpeedX());
        const float y_speed = std::fabs(game.getBallSpeedY());
        const float ratio = y_speed / x_speed;

        expect(ratio >= 0.5f && ratio <= 2.0f,
               "serve y speed should stay between 0.5x and 2.0x the x speed");
        if (ratio <= 1.0f) {
            ++low_band_count;
        } else {
            ++high_band_count;
        }
    }

    expect(low_band_count >= 400 && low_band_count <= 600,
           "about half of serves should use the 0.5x-1.0x y-speed band");
    expect(high_band_count >= 400 && high_band_count <= 600,
           "about half of serves should use the 1.0x-2.0x y-speed band");
}

void test_paddle_stays_put_when_relative_difference_is_within_three_percent() {
    PongGame game;
    const int initial_y = game.getPaddle1Y();

    (void)game.update(103, 100);
    expect(game.getPaddle1Y() == initial_y,
           "paddle should not move when relative spike difference is at most three percent");
}

void test_paddle_moves_when_relative_difference_exceeds_three_percent() {
    PongGame game;
    const int initial_y = game.getPaddle1Y();

    (void)game.update(104, 96);
    expect(game.getPaddle1Y() < initial_y,
           "paddle should move up when relative spike difference exceeds three percent");
}

void test_paddle_can_travel_from_top_to_bottom_within_one_second() {
    PongGame game;

    while (game.getPaddle1Y() > 0) {
        (void)game.update(100, 0);
    }

    const int bottom_y = game.getGameHeight() - game.getPaddleHeight();
    int steps = 0;
    while (game.getPaddle1Y() < bottom_y && steps < 100) {
        (void)game.update(0, 100);
        ++steps;
    }

    expect(game.getPaddle1Y() == bottom_y,
           "paddle should reach the bottom edge within one second of updates");
    expect(steps <= 100,
           "paddle should require at most 100 updates to travel from top to bottom");
}

void test_miss_preserves_completed_rally_bounces_until_next_update() {
    std::srand(1);
    PongGame game;

    bool saw_hit = false;
    for (int i = 0; i < 5000; ++i) {
        int spikes_up = 0;
        int spikes_down = 0;

        if (!saw_hit) {
            const int paddle_top = game.getPaddle1Y();
            const int paddle_bottom = paddle_top + game.getPaddleHeight();
            const int ball_y = game.getBallY();
            if (ball_y < paddle_top) {
                spikes_up = 1;
            } else if (ball_y > paddle_bottom) {
                spikes_down = 1;
            }
        } else {
            spikes_up = 1;
        }

        const GameEvent event = game.update(spikes_up, spikes_down);
        if (event == GameEvent::BallHitPlayerPaddle) {
            saw_hit = true;
        }

        if (saw_hit && event == GameEvent::PlayerMissed) {
            expect(game.getBounces() > 0,
                   "miss should preserve completed rally bounces for downstream logging");
            (void)game.update(0, 0);
            expect(game.getBounces() == 0,
                   "next rally should reset bounce count after miss is observed");
            return;
        }
    }

    throw std::runtime_error("expected a rally with one hit followed by a miss");
}

}  // namespace

int main() {
    try {
        test_centered_default_state_maps_to_center_zone();
        test_sensory_zone_depends_on_absolute_ball_y_not_paddle_position();
        test_rest_condition_suppresses_sensory_zone();
        test_serve_speed_sets_five_second_round_trip();
        test_serve_y_speed_ratio_is_randomized_in_required_bands();
        test_paddle_stays_put_when_relative_difference_is_within_three_percent();
        test_paddle_moves_when_relative_difference_exceeds_three_percent();
        test_paddle_can_travel_from_top_to_bottom_within_one_second();
        test_miss_preserves_completed_rally_bounces_until_next_update();
    } catch (const std::exception& e) {
        std::cerr << "ponggame_test failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "ponggame_test passed\n";
    return 0;
}
