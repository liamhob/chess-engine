#pragma once

#include "chess/mcts.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace chess::mcts {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t worker_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::future<void> submit(std::function<void()> task);
    void shutdown() noexcept;

private:
    void worker_loop();

    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<std::packaged_task<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

struct LeafWork {
    std::uint64_t request_id = 0;
    std::uint64_t state_hash = 0;
};

class LeafQueue {
public:
    void push(LeafWork work);
    bool pop(LeafWork& work);
    void close() noexcept;
    bool closed() const noexcept;

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<LeafWork> queue_;
    bool closed_ = false;
};

}  // namespace chess::mcts
