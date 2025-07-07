#include "utec/agent/EnvGym.h"
#include <cstdlib>
#include <cmath>

namespace utec::nn {

    State EnvGym::reset() {
        paddle_y_ = 0.5f;
        ball_y_ = static_cast<float>(rand()) / RAND_MAX;
        return {0.5f, ball_y_, paddle_y_};
    }

    State EnvGym::step(int action, float& reward, bool& done) {
        paddle_y_ += 0.1f * action;
        if (paddle_y_ < 0) paddle_y_ = 0;
        if (paddle_y_ > 1) paddle_y_ = 1;

        ball_y_ = static_cast<float>(rand()) / RAND_MAX;

        reward = std::fabs(ball_y_ - paddle_y_) < 0.2f ? +1.f : -1.f;
        done = true;

        return {0.5f, ball_y_, paddle_y_};
    }

}
