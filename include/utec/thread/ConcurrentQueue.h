#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

namespace utec::thread {

    template<typename T>
    class ConcurrentQueue {
    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_var_;

    public:
        void push(const T& item) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(item);
            }
            cond_var_.notify_one();
        }

        T pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_var_.wait(lock, [this] { return !queue_.empty(); });
            T item = queue_.front();
            queue_.pop();
            return item;
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }
    };

}
