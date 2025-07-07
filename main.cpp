#include "utec/agent/PongAgentTrainable.h"
#include "utec/agent/EnvGym.h"
#include "neural_network.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace utec;

int main() {
    using T = float;
    srand(time(nullptr));

    // Inicialización aleatoria
    auto init_random = [](auto& W) {
        std::default_random_engine gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5, 0.5);
        for (auto& w : W) w = dist(gen);
    };

    // Red neuronal
    neural_network::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<neural_network::Dense<T>>(3, 16, init_random, init_random));
    net.add_layer(std::make_unique<neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<neural_network::Dense<T>>(16, 8, init_random, init_random));
    net.add_layer(std::make_unique<neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<neural_network::Dense<T>>(8, 1, init_random, init_random));

    if (std::ifstream("pesos.txt").good()) {
        net.load_model("pesos.txt");
        std::cout << "📦 Pesos anteriores cargados desde pesos.txt\n";
    } else {
        std::cout << "📁 No se encontró pesos.txt, se iniciará desde cero.\n";
    }


    nn::PongAgentTrainable<T> agent(
            [&](const algebra::Tensor<T,2>& x) {
                return net.predict(x);
            },
            net,
            0.95,  // gamma
            0.005  // learning rate
    );

    nn::EnvGym env;
    const int episodios = 3000;
    const int bloque = 100;
    std::ofstream winrate_csv("winrate.csv");
    winrate_csv << "Bloque,Winrate\n";
    int victorias_bloque = 0;

    for (int episodio = 0; episodio < episodios; ++episodio) {
        auto s = env.reset();

        int a = (float(rand()) / RAND_MAX < 0.1f) ? rand() % 2 : agent.act(s);
        float total_reward = 0;
        bool done = false;

        while (!done) {
            float r;
            auto s_next = env.step(a, r, done);
            int a_next = (float(rand()) / RAND_MAX < 0.1f) ? rand() % 2 : agent.act(s_next);
            agent.learnOnPolicy(s, a, r, s_next, a_next);
            s = s_next;
            a = a_next;
            total_reward += r;
        }

        if (total_reward > 0) ++victorias_bloque;
        std::cout << "Episodio " << episodio << " | Recompensa: " << total_reward << "\n";

        if ((episodio + 1) % bloque == 0) {
            float winrate = 100.0f * victorias_bloque / bloque;
            std::cout << "🔥 Winrate: " << winrate << "%\n";
            winrate_csv << (episodio + 1) << "," << winrate << "\n";
            victorias_bloque = 0;
        }
    }

    winrate_csv.close();
    net.save_model("pesos.txt");
    std::cout << "✅ Pesos actualizados guardados en pesos.txt\n";
    return 0;
}