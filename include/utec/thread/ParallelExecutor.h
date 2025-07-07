#pragma once

#include "utec/thread/ThreadPool.h"
#include "utec/agent/PongAgent.h"
#include "utec/agent/State.h"

namespace utec::thread {

    template<typename T>
    class ParallelExecutor {
    private:
        ThreadPool pool_;
        utec::nn::PongAgent<T>& agent_;

    public:
        ParallelExecutor(size_t threads, utec::nn::PongAgent<T>& agent)
                : pool_(threads), agent_(agent) {}

        std::future<int> infer_async(const utec::nn::State& s) {
            return pool_.enqueue([this, s]() {
                return agent_.act(s);
            });
        }
    };

}
