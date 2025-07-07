#pragma once
#include "nn_interfaces.h"
#include <cmath>
#include <vector>

namespace utec {
    namespace neural_network {

        template<typename T>
        class SGD final : public IOptimizer<T> {
        private:
            T lr_;
        public:
            explicit SGD(T lr = 0.01) : lr_(lr) {}

            void update(utec::algebra::Tensor<T,2>& params, const utec::algebra::Tensor<T,2>& grads) override {
                size_t N = params.size();
                auto p_it = params.begin();
                auto g_it = grads.cbegin();
                for (size_t i = 0; i < N; ++i, ++p_it, ++g_it)
                    *p_it -= lr_ * (*g_it);
            }
        };

        template<typename T>
        class Adam final : public IOptimizer<T> {
        private:
            T lr_, beta1_, beta2_, epsilon_;
            std::vector<T> m_, v_;
            int t_ = 0;
        public:
            explicit Adam(T lr = 0.001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
                    : lr_(lr), beta1_(b1), beta2_(b2), epsilon_(eps) {}

            void update(utec::algebra::Tensor<T,2>& params, const utec::algebra::Tensor<T,2>& grads) override {
                size_t N = params.size();
                if (m_.empty()) {
                    m_.resize(N, T(0));
                    v_.resize(N, T(0));
                }
                ++t_;
                auto p_it = params.begin();
                auto g_it = grads.cbegin();
                for (size_t i = 0; i < N; ++i, ++p_it, ++g_it) {
                    m_[i] = beta1_ * m_[i] + (1 - beta1_) * (*g_it);
                    v_[i] = beta2_ * v_[i] + (1 - beta2_) * (*g_it) * (*g_it);

                    T m_hat = m_[i] / (1 - std::pow(beta1_, t_));
                    T v_hat = v_[i] / (1 - std::pow(beta2_, t_));

                    *p_it -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }

            void step() override {}
        };

    } // namespace neural_network
} // namespace utec

template<typename T>
using SGD = utec::neural_network::SGD<T>;

template<typename T>
using Adam = utec::neural_network::Adam<T>;
