#include "utec/agent/PongAgent.h"
#include "utec/agent/EnvGym.h"
#include "neural_network.h"
#include <iostream>

using namespace utec;

int main() {
    using T = float;

    neural_network::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<neural_network::Dense<T>>(3, 6));
    net.add_layer(std::make_unique<neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<neural_network::Dense<T>>(6, 1));

    net.load_model("pesos.txt");
    std::cout << "Modelo cargado desde pesos.txt\n";

    nn::PongAgent<T> agent([&](const algebra::Tensor<T,2>& x) {
        return net.predict(x);
    });

    nn::EnvGym env;
    int victorias = 0;
    int total = 100;

    for (int episodio = 0; episodio < total; ++episodio) {
        auto s = env.reset();
        float reward;
        bool done = false;

        int a = agent.act(s);
        s = env.step(a, reward, done);

        if (reward > 0)
            ++victorias;
    }

    float porcentaje = (100.0f * victorias) / total;
    std::cout << "Winrate: " << victorias << " / " << total
              << " (" << porcentaje << "%)\n";

    return 0;
}
