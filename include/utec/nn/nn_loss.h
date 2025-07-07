#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "tensor.h"
#include "nn_interfaces.h"

template<typename T>
class MSELoss final : public utec::neural_network::ILoss<T> {
private:
    utec::algebra::Tensor<T,2> y_pred_, y_true_;
public:
    MSELoss(const utec::algebra::Tensor<T,2>& y_pred, const utec::algebra::Tensor<T,2>& y_true)
            : y_pred_(y_pred), y_true_(y_true) {}

    T loss() const override {
        T sum = 0;
        auto pred_it = y_pred_.cbegin();
        auto true_it = y_true_.cbegin();
        for (size_t i = 0; i < y_pred_.size(); ++i, ++pred_it, ++true_it)
            sum += (*pred_it - *true_it) * (*pred_it - *true_it);
        return sum / T(y_pred_.size());
    }

    utec::algebra::Tensor<T,2> loss_gradient() const override {
        auto shape = y_pred_.shape();
        utec::algebra::Tensor<T,2> grad(shape[0], shape[1]);
        auto pred_it = y_pred_.cbegin();
        auto true_it = y_true_.cbegin();
        auto grad_it = grad.begin();
        for (size_t i = 0; i < grad.size(); ++i, ++pred_it, ++true_it, ++grad_it)
            *grad_it = (2.0 / T(grad.size())) * (*pred_it - *true_it);
        return grad;
    }
};

template<typename T>
class BCELoss final : public utec::neural_network::ILoss<T> {
private:
    utec::algebra::Tensor<T,2> y_pred_, y_true_;
public:
    BCELoss(const utec::algebra::Tensor<T,2>& y_pred, const utec::algebra::Tensor<T,2>& y_true)
            : y_pred_(y_pred), y_true_(y_true) {}

    T loss() const override {
        T sum = 0;
        auto pred_it = y_pred_.cbegin();
        auto true_it = y_true_.cbegin();
        for (size_t i = 0; i < y_pred_.size(); ++i, ++pred_it, ++true_it) {
            T y = *true_it;
            T p = std::clamp(*pred_it, T(1e-12), T(1) - T(1e-12));
            sum += -(y * std::log(p) + (T(1) - y) * std::log(T(1) - p));
        }
        return sum / T(y_pred_.size());
    }

    utec::algebra::Tensor<T,2> loss_gradient() const override {
        auto shape = y_pred_.shape();
        utec::algebra::Tensor<T,2> grad(shape[0], shape[1]);
        auto pred_it = y_pred_.cbegin();
        auto true_it = y_true_.cbegin();
        auto grad_it = grad.begin();
        for (size_t i = 0; i < grad.size(); ++i, ++pred_it, ++true_it, ++grad_it) {
            T y = *true_it;
            T p = std::clamp(*pred_it, T(1e-12), T(1) - T(1e-12));
            *grad_it = (p - y) / (p * (T(1) - p) * T(grad.size()));
        }
        return grad;
    }
};

namespace utec::neural_network {
    template <typename T>
    using MSELoss = ::MSELoss<T>;

    template <typename T>
    using BCELoss = ::BCELoss<T>;
}
