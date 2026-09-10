#pragma once

#include "chess/position.h"

#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace chess::bridge {

struct InferenceResult {
    std::vector<float> policy;
    float value = 0.0F;
};

struct InferenceRequest {
    std::uint64_t request_id = 0;
    Position position;
    std::shared_ptr<std::promise<InferenceResult>> response;
};

class InferenceQueue {
public:
    std::future<InferenceResult> submit(Position position);
    std::uint64_t submit_id(Position position);
    InferenceResult wait_result(std::uint64_t request_id);
    std::vector<InferenceRequest> pop_batch(std::size_t max_batch);
    void respond(std::uint64_t request_id, InferenceResult result);
    void close() noexcept;
    bool closed() const noexcept;

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<InferenceRequest> pending_;
    std::unordered_map<std::uint64_t, std::shared_ptr<std::promise<InferenceResult>>> in_flight_;
    std::unordered_map<std::uint64_t, std::future<InferenceResult>> client_futures_;
    std::uint64_t next_request_id_ = 1;
    bool closed_ = false;
};

}  // namespace chess::bridge
