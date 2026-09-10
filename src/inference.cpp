#include "chess/inference.h"

#include <stdexcept>
#include <unordered_map>

namespace chess::bridge {

std::future<InferenceResult> InferenceQueue::submit(Position position) {
    auto promise = std::make_shared<std::promise<InferenceResult>>();
    auto future = promise->get_future();
    {
        std::lock_guard lock(mutex_);
        if (closed_) throw std::logic_error("cannot submit to a closed inference queue");
        pending_.push(InferenceRequest{next_request_id_++, std::move(position), promise});
    }
    condition_.notify_one();
    return future;
}

std::uint64_t InferenceQueue::submit_id(Position position) {
    auto promise = std::make_shared<std::promise<InferenceResult>>();
    auto future = promise->get_future();
    std::uint64_t request_id;
    {
        std::lock_guard lock(mutex_);
        if (closed_) throw std::logic_error("cannot submit to a closed inference queue");
        request_id = next_request_id_++;
        pending_.push(InferenceRequest{request_id, std::move(position), promise});
        client_futures_.emplace(request_id, std::move(future));
    }
    condition_.notify_one();
    return request_id;
}

InferenceResult InferenceQueue::wait_result(std::uint64_t request_id) {
    std::future<InferenceResult> future;
    {
        std::lock_guard lock(mutex_);
        const auto iterator = client_futures_.find(request_id);
        if (iterator == client_futures_.end()) throw std::out_of_range("unknown inference request");
        future = std::move(iterator->second);
        client_futures_.erase(iterator);
    }
    return future.get();
}

std::vector<InferenceRequest> InferenceQueue::pop_batch(std::size_t max_batch) {
    if (max_batch == 0) throw std::invalid_argument("inference batch size must be positive");
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !pending_.empty(); });
    std::vector<InferenceRequest> batch;
    while (!pending_.empty() && batch.size() < max_batch) {
        InferenceRequest request = std::move(pending_.front());
        pending_.pop();
        in_flight_.emplace(request.request_id, request.response);
        batch.push_back(std::move(request));
    }
    return batch;
}

void InferenceQueue::respond(std::uint64_t request_id, InferenceResult result) {
    std::shared_ptr<std::promise<InferenceResult>> promise;
    {
        std::lock_guard lock(mutex_);
        const auto iterator = in_flight_.find(request_id);
        if (iterator == in_flight_.end()) throw std::out_of_range("unknown inference request");
        promise = iterator->second;
        in_flight_.erase(iterator);
    }
    promise->set_value(std::move(result));
}

void InferenceQueue::close() noexcept {
    std::queue<InferenceRequest> pending;
    {
        std::lock_guard lock(mutex_);
        if (closed_) return;
        closed_ = true;
        pending.swap(pending_);
    }
    const auto error = std::make_exception_ptr(std::runtime_error("inference queue closed"));
    while (!pending.empty()) {
        pending.front().response->set_exception(error);
        pending.pop();
    }
    std::unordered_map<std::uint64_t, std::shared_ptr<std::promise<InferenceResult>>> in_flight;
    {
        std::lock_guard lock(mutex_);
        in_flight.swap(in_flight_);
    }
    for (auto& [request_id, promise] : in_flight) promise->set_exception(error);
    condition_.notify_all();
}

bool InferenceQueue::closed() const noexcept {
    std::lock_guard lock(mutex_);
    return closed_;
}

}  // namespace chess::bridge
