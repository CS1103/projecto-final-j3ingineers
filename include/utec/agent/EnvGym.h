#pragma once
#include "State.h"

namespace utec::nn {

    class EnvGym {
    private:
        float paddle_y_ = 0.5f;
        float ball_y_ = 0.5f;

    public:
        State reset();
        State step(int action, float& reward, bool& done);
    };

}
