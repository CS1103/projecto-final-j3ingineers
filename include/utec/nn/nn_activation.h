#pragma once
#include "nn_interfaces.h"
#include <algorithm>

namespace utec {
    namespace neural_network {

        template<typename T>
        class ReLU final : public ILayer<T> {
        private:
            utec::algebra::Tensor<T,2> cached_;
        public:
            utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
                cached_ = z;
                utec::algebra::Tensor<T,2> out = z;
                std::transform(z.cbegin(), z.cend(), out.begin(),
                               [](T val) { return val > T(0) ? val : T(0); });
                return out;
            }

            utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
                utec::algebra::Tensor<T,2> grad = g;
                auto c_it = cached_.cbegin();
                auto g_it = g.cbegin();
                auto grad_it = grad.begin();
                for (size_t i = 0; i < grad.size(); ++i, ++c_it, ++g_it, ++grad_it) {
                    *grad_it = (*c_it > T(0)) ? *g_it : T(0);
                }
                return grad;
            }
        };

        template<typename T>
        class Sigmoid final : public ILayer<T> {
        private:
            utec::algebra::Tensor<T,2> cached_;
        public:
            utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
                cached_ = z;
                utec::algebra::Tensor<T,2> out = z;
                std::transform(z.cbegin(), z.cend(), out.begin(),
                               [](T val) { return T(1) / (T(1) + std::exp(-val)); });
                cached_ = out;
                return out;
            }

            utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
                utec::algebra::Tensor<T,2> grad = g;
                auto c_it = cached_.cbegin();
                auto g_it = g.cbegin();
                auto grad_it = grad.begin();
                for (size_t i = 0; i < grad.size(); ++i, ++c_it, ++g_it, ++grad_it) {
                    *grad_it = (*c_it) * (T(1) - *c_it) * (*g_it);
                }
                return grad;
            }
        };

    } // namespace neural_network
} // namespace utec
