#include "chess/concurrency.h"

#include <stdexcept>

namespace chess::mcts {

ThreadPool::ThreadPool(std::size_t worker_count) {
    if (worker_count == 0) throw std::invalid_argument("thread pool requires a worker");
    workers_.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
    std::packaged_task<void()> packaged(std::move(task));
    std::future<void> result = packaged.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) throw std::logic_error("cannot submit to stopped thread pool");
        tasks_.push(std::move(packaged));
    }
    condition_.notify_one();
    return result;
}

void ThreadPool::shutdown() noexcept {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) return;
        stopping_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::packaged_task<void()> task;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void LeafQueue::push(LeafWork work) {
    {
        std::lock_guard lock(mutex_);
        if (closed_) throw std::logic_error("cannot push to closed leaf queue");
        queue_.push(work);
    }
    condition_.notify_one();
}

bool LeafQueue::pop(LeafWork& work) {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    work = queue_.front();
    queue_.pop();
    return true;
}

void LeafQueue::close() noexcept {
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
    }
    condition_.notify_all();
}

bool LeafQueue::closed() const noexcept {
    std::lock_guard lock(mutex_);
    return closed_;
}

}  // namespace chess::mcts
