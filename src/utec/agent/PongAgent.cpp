#include "utec/agent/PongAgent.h"

namespace utec::nn {

    template<typename T>
    int PongAgent<T>::act(const State& s) {
        using Tensor2D = utec::algebra::Tensor<T,2>;
        Tensor2D input(1, 3);
        input(0, 0) = s.ball_x;
        input(0, 1) = s.ball_y;
        input(0, 2) = s.paddle_y;

        auto output = forward_fn(input);

        T val = output(0,0);
        if (val > T(0.1)) return +1;
        if (val < T(-0.1)) return -1;
        return 0;
    }
    template class PongAgent<float>;
    template class PongAgent<double>;

}
