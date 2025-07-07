#pragma once

#include "PongAgent.h"
#include "EnvGym.h"
#include "neural_network.h"
#include <random>

namespace utec::nn {

    template<typename T>
    class PongAgentTrainable : public PongAgent<T> {
    private:
        neural_network::NeuralNetwork<T>& net_;
        T gamma_;
        T lr_;

    public:
        PongAgentTrainable(
                std::function<algebra::Tensor<T,2>(const algebra::Tensor<T,2>&)> fwd,
                neural_network::NeuralNetwork<T>& net,
                T gamma = 0.95, T lr = 0.01
        ) : PongAgent<T>(fwd), net_(net), gamma_(gamma), lr_(lr) {}

        void learnOnPolicy(const State& s, int a, float r, const State& s_next, int a_next) {
            using namespace algebra;

            Tensor<T,2> x(1,3);
            x(0,0) = s.ball_x / 100.0;
            x(0,1) = s.ball_y / 100.0;
            x(0,2) = s.paddle_y / 100.0;

            Tensor<T,2> x_next(1,3);
            x_next(0,0) = s_next.ball_x / 100.0;
            x_next(0,1) = s_next.ball_y / 100.0;
            x_next(0,2) = s_next.paddle_y / 100.0;

            auto Q_pred = net_.predict(x);
            auto Q_next = net_.predict(x_next);

            Tensor<T,2> target = Q_pred;
            target(0,0) = r + gamma_ * Q_next(0,0);

            net_.train(x, target, 5, 1, lr_);
        }
    };

}
