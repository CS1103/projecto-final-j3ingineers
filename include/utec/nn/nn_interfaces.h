#pragma once
#include "tensor.h"

template<typename T>
class IOptimizer {
public:
    virtual void update(utec::algebra::Tensor<T,2>& params, const utec::algebra::Tensor<T,2>& grads) = 0;
    virtual void step() {}
    virtual ~IOptimizer() = default;
};

namespace utec {
    namespace neural_network {

        template<typename T>
        class ILayer {
        public:
            virtual utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) = 0;
            virtual utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& dZ) = 0;
            virtual void update_params(class IOptimizer<T>& optimizer) {}
            virtual ~ILayer() = default;
        };

        template<typename T>
        class ILoss {
        public:
            virtual T loss() const = 0;
            virtual utec::algebra::Tensor<T,2> loss_gradient() const = 0;
            virtual ~ILoss() = default;
        };



    } // namespace neural_network
} // namespace utec
