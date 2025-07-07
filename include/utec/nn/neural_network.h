#pragma once

#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <fstream>
#include "tensor.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers_;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.emplace_back(std::move(layer));
        }

        Tensor<T,2> predict(const Tensor<T,2>& X) {
            Tensor<T,2> output = X;
            for (auto& layer : layers_)
                output = layer->forward(output);
            return output;
        }

        template <typename LossType = MSELoss<T>, typename OptimizerType = SGD<T>>
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
                   size_t epochs, size_t batch_size, T learning_rate) {

            size_t n_samples = X.shape()[0];

            OptimizerType optimizer(learning_rate);

            std::vector<size_t> indices(n_samples);
            std::iota(indices.begin(), indices.end(), 0);

            std::mt19937 rng(std::random_device{}());

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);

                for (size_t i = 0; i < n_samples; i += batch_size) {
                    size_t current_batch = std::min(batch_size, n_samples - i);

                    Tensor<T,2> X_batch(current_batch, X.shape()[1]);
                    Tensor<T,2> Y_batch(current_batch, Y.shape()[1]);

                    for (size_t j = 0; j < current_batch; ++j) {
                        X_batch[j] = X[indices[i + j]];
                        Y_batch[j] = Y[indices[i + j]];
                    }

                    Tensor<T,2> output = X_batch;
                    for (auto& layer : layers_)
                        output = layer->forward(output);

                    LossType loss_fn(output, Y_batch);
                    Tensor<T,2> grad = loss_fn.loss_gradient();

                    for (int k = layers_.size() - 1; k >= 0; --k)
                        grad = layers_[k]->backward(grad);

                    for (auto& layer : layers_)
                        layer->update_params(optimizer);
                }
            }
        }

        void save_model(const std::string& filename) const {
            std::ofstream out(filename);
            for (const auto& layer : layers_) {
                auto* d = dynamic_cast<Dense<T>*>(layer.get());
                if (d) d->save(out);
            }
        }

        void load_model(const std::string& filename) {
            std::ifstream in(filename);
            for (const auto& layer : layers_) {
                auto* d = dynamic_cast<Dense<T>*>(layer.get());
                if (d) d->load(in);
            }
        }
    };

}
