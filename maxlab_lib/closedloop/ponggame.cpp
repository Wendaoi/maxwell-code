#include "ponggame.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

constexpr float kGameWindowMs = 10.0f;
constexpr float kHalfTripSeconds = 2.5f;
constexpr float kPaddleMoveRelativeThreshold = 0.03f;
constexpr int kPaddleSpeedPixelsPerUpdate = 5;
constexpr int kUpdatesPerSecond = static_cast<int>(1000.0f / kGameWindowMs);
constexpr int kPaddleTravelRange =
    ponggame_defaults::kGameHeight - ponggame_defaults::kPaddleHeight;

static_assert(kPaddleSpeedPixelsPerUpdate * kUpdatesPerSecond >= kPaddleTravelRange,
              "Paddle speed must cover the full travel range within one second");

float randomFloat(float minValue, float maxValue) {
    const float unit = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return minValue + unit * (maxValue - minValue);
}

float randomServeYSpeed(float absSpeedX) {
    const bool lowBand = (rand() % 2) == 0;
    const float multiplier = lowBand ? randomFloat(0.5f, 1.0f)
                                     : randomFloat(1.0f, 2.0f);
    const float sign = (rand() % 2) == 0 ? -1.0f : 1.0f;
    return sign * absSpeedX * multiplier;
}

}  // namespace

PongGame::PongGame() {
    gameWidth = ponggame_defaults::kGameWidth;
    gameHeight = ponggame_defaults::kGameHeight;
    paddleWidth = ponggame_defaults::kPaddleWidth;
    paddleHeight = ponggame_defaults::kPaddleHeight;
    ballSize = ponggame_defaults::kBallSize; // 球无体积，作为点处理
    currentCondition = ExperimentCondition::Stimulus;
    resetBall(true);
    bounces_in_rally = 0;
    reset_bounces_on_next_update = false;
    paddleY = gameHeight / 2 - paddleHeight / 2;
}

void PongGame::resetBall(bool randomVector) {
    ballX = static_cast<float>(gameWidth) / 2.0f;  // 球画面中心向右出发
    ballY = static_cast<float>(gameHeight) / 2.0f;
    const float updatesPerHalfTrip = kHalfTripSeconds * 1000.0f / kGameWindowMs;
    const float serveSpeedX = static_cast<float>(gameWidth) / updatesPerHalfTrip;
    if (randomVector) {
        ballSpeedX = serveSpeedX;  // 向右运动
    } else {
        ballSpeedX = -serveSpeedX;  // 向左运动
    }
    ballSpeedY = randomServeYSpeed(std::fabs(ballSpeedX));
}

void PongGame::setCondition(ExperimentCondition condition) {
    currentCondition = condition;
}

ExperimentCondition PongGame::getCondition() const {
    return currentCondition;
}

GameEvent PongGame::update(int spikesUp, int spikesDown) {
    if (reset_bounces_on_next_update) {
        bounces_in_rally = 0;
        reset_bounces_on_next_update = false;
    }

    // 1. 根据尖峰更新玩家球拍位置
    const float total_spikes = static_cast<float>(spikesUp + spikesDown);
    if (total_spikes > 0.0f) {
        const float relative_difference =
            std::fabs(static_cast<float>(spikesUp - spikesDown)) / total_spikes;
        if (relative_difference > kPaddleMoveRelativeThreshold) {
            if (spikesUp > spikesDown) {
                paddleY -= kPaddleSpeedPixelsPerUpdate;
            } else if (spikesDown > spikesUp) {
                paddleY += kPaddleSpeedPixelsPerUpdate;
            }
        }
    }
    paddleY = std::max(0, std::min(paddleY, gameHeight - paddleHeight));

    // 2. 使用增量的方式逐步更新球位置，处理碰撞
    GameEvent event = updateBallPosition();

    return event;
}

GameEvent PongGame::updateBallPosition() {
    // 球作为点处理，简化解算
    // 移动球点到新位置
    float newBallX = ballX + ballSpeedX;
    float newBallY = ballY + ballSpeedY;

    // 2.1 处理X轴碰撞检测（球拍和边界）
    if (ballSpeedX < 0.0f && checkPaddleCollision(newBallX, newBallY)) { // 仅当球向左移动检测球拍碰撞
        // 球拍碰撞，反弹（简化逻辑，与边界反弹一致）
        ballSpeedX = -ballSpeedX;

        // 更新位置（反弹后立即反向移动一步，以避免立即再次碰撞）
        ballX = - newBallX; 
        ballY = newBallY;

        return GameEvent::BallHitPlayerPaddle;
    } else {
        // 无球拍碰撞，检查左右边界
        if (newBallX <= 0.0f) {
            // 球点越过左侧边界（玩家未接住）
            ballX = 0.0f;
            ballSpeedX = -ballSpeedX; // 反弹，但实际是重置

            if (currentCondition != ExperimentCondition::NoFeedback) {
                resetBall(true);
                reset_bounces_on_next_update = true;
            }
            return GameEvent::PlayerMissed;
        } else if (newBallX >= static_cast<float>(gameWidth)) {
            // 球点碰到右侧边界，反弹
            ballX = static_cast<float>(gameWidth);
            ballSpeedX = -ballSpeedX;
            // 不更新Y位置，因为未移动到新位置
        } else {
            // 正常移动
            ballX = newBallX;
        }
    }

    // 2.2 处理Y轴边界碰撞
    if (newBallY <= 0.0f) {
        // 球点碰到上边界，反弹
        ballY = 0.0f;
        ballSpeedY = -ballSpeedY;
    } else if (newBallY >= static_cast<float>(gameHeight)) {
        // 球点碰到下边界，反弹
        ballY = static_cast<float>(gameHeight);
        ballSpeedY = -ballSpeedY;
    } else {
        // 正常移动
        ballY = newBallY;
    }

    return GameEvent::None;
}

bool PongGame::checkPaddleCollision(float newBallX, float newBallY) {
    // 球拍视为垂直线段 x=0，从 y=paddleY 到 y=paddleY + paddleHeight
    // 球视为点，从 (ballX, ballY) 到 (newBallX, newBallY)
    // 仅当球向左移动且轨迹穿过 x=0 时检测

    if (!(newBallX <= 0.0f)) { // 球未从右侧越过x=0
        return false;
    }

    // 计算球轨迹与 x=0 的交点Y坐标（简化解算）
    float y_intersect = ballY + (0.0f - ballX) * (newBallY - ballY) / (newBallX - ballX);

    // 检查交点是否在球拍线段范围内
    if (y_intersect >= static_cast<float>(paddleY) && y_intersect <= static_cast<float>(paddleY + paddleHeight)) {
        bounces_in_rally++;
        return true;
    }

    return false;
}

int PongGame::getSensoryStimZone() const {
    if (currentCondition == ExperimentCondition::Rest) {
        return -1; // Rest条件下无感觉输入
    }

    // 将球点相对于球拍线段的位置映射到8个感觉刺激区域。
    // 屏幕Y坐标向下递增；区域0/pos0在球拍上方，区域7/pos7在球拍下方。

    // 计算球点和球拍中心的Y坐标
    float ballCenterY = ballY; // 球为点，其位置即中心
    float paddleCenterY = static_cast<float>(paddleY) + static_cast<float>(paddleHeight) / 2.0f;

    // 计算相对位置 (球相对于球拍中心的Y偏移)
    float relativeY = ballCenterY - paddleCenterY;

    // 将相对位置映射到8个区域 (范围 [-gameHeight/2, gameHeight/2] 均匀分为8个区域)
    const float zoneRange = static_cast<float>(gameHeight) / 8.0f;
    int offsetY = static_cast<int>((relativeY + gameHeight / 2.0f) / zoneRange);

    // 边界处理：确保在0-7范围内
    if (offsetY < 0) offsetY = 0;
    if (offsetY > 7) offsetY = 7;

    return offsetY;
}
