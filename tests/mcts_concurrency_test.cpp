#include "chess/concurrency.h"

#include <atomic>
#include <cassert>
#include <stdexcept>

int main() {
    using namespace chess::mcts;

    ThreadPool pool(4);
    std::atomic<int> completed{0};
    std::vector<std::future<void>> futures;
    for (int index = 0; index < 32; ++index) {
        futures.push_back(pool.submit([&completed] { completed.fetch_add(1); }));
    }
    for (auto& future : futures) future.get();
    assert(completed.load() == 32);
    pool.shutdown();

    LeafQueue queue;
    queue.push({7, 0xAA});
    queue.push({8, 0xBB});
    LeafWork work;
    assert(queue.pop(work) && work.request_id == 7 && work.state_hash == 0xAA);
    assert(queue.pop(work) && work.request_id == 8 && work.state_hash == 0xBB);
    queue.close();
    assert(queue.closed());
    assert(!queue.pop(work));
    bool rejected = false;
    try {
        queue.push({9, 0xCC});
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
