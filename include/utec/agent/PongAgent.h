#pragma once

#include "utec/nn/nn_interfaces.h"
#include "utec/agent/State.h"
#include "tensor.h"
#include <memory>
#include <functional>

namespace utec::nn {

    template<typename T>
    class PongAgent {
    private:
        std::function<utec::algebra::Tensor<T,2>(const utec::algebra::Tensor<T,2>&)> forward_fn;

    public:
        PongAgent(std::function<utec::algebra::Tensor<T,2>(const utec::algebra::Tensor<T,2>&)> fwd)
                : forward_fn(fwd) {}

        int act(const State& s);
    };

}
