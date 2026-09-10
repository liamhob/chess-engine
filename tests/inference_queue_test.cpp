#include "chess/inference.h"

#include <cassert>
#include <stdexcept>

int main() {
    using namespace chess;
    using namespace chess::bridge;

    InferenceQueue queue;
    auto first = queue.submit(Position::start());
    auto second = queue.submit(Position::start());
    const auto batch = queue.pop_batch(2);
    assert(batch.size() == 2);
    assert(batch[0].request_id != batch[1].request_id);

    queue.respond(batch[0].request_id, {{1.0F, 0.0F}, 0.25F});
    queue.respond(batch[1].request_id, {{0.0F, 1.0F}, -0.5F});
    assert(first.get().value == 0.25F);
    assert(second.get().value == -0.5F);

    queue.close();
    assert(queue.closed());
    bool rejected = false;
    try {
        queue.submit(Position::start());
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
