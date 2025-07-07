#pragma once

#include "nn_interfaces.h"
#include "tensor.h"
#include <functional>
#include <random>
#include <fstream>

namespace utec::neural_network {

    template <typename T>
    using InitFunc = std::function<void(utec::algebra::Tensor<T, 2>&)>;

    template <typename T>
    class Dense : public ILayer<T> {
    private:
        using Tensor2D = utec::algebra::Tensor<T, 2>;

        Tensor2D W_;
        Tensor2D b_;
        Tensor2D input_;
        Tensor2D dW_;
        Tensor2D db_;
        InitFunc<T> weight_init_;
        InitFunc<T> bias_init_;

    public:
        Dense(size_t in, size_t out, InitFunc<T> w_init, InitFunc<T> b_init)
                : W_(in, out), b_(1, out),
                  dW_(in, out), db_(1, out),
                  weight_init_(w_init), bias_init_(b_init) {
            weight_init_(W_);
            bias_init_(b_);
        }

        // Constructor por defecto simplificado
        Dense(size_t in, size_t out)
                : Dense(in, out,
                        [](auto& W) { W.fill(0.1); },
                        [](auto& b) { b.fill(0); }) {}

        Tensor2D forward(const Tensor2D& input) override {
            input_ = input;
            auto z = utec::algebra::matrix_product(input, W_);
            for (size_t i = 0; i < z.shape()[0]; ++i)
                for (size_t j = 0; j < z.shape()[1]; ++j)
                    z(i, j) += b_(0, j);
            return z;
        }

        Tensor2D backward(const Tensor2D& grad_output) override {
            auto input_T = utec::algebra::transpose_2d(input_);
            dW_ = utec::algebra::matrix_product(input_T, grad_output);

            db_.fill(0);
            for (size_t i = 0; i < grad_output.shape()[0]; ++i)
                for (size_t j = 0; j < grad_output.shape()[1]; ++j)
                    db_(0, j) += grad_output(i, j);

            auto W_T = utec::algebra::transpose_2d(W_);
            return utec::algebra::matrix_product(grad_output, W_T);
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W_, dW_);
            optimizer.update(b_, db_);
        }

        void save(std::ostream& out) const {
            for (const auto& v : W_) out << v << " ";
            for (const auto& v : b_) out << v << " ";
            out << "\n";
        }

        void load(std::istream& in) {
            for (auto& v : W_) in >> v;
            for (auto& v : b_) in >> v;
        }
    };

} // namespace utec::neural_network
