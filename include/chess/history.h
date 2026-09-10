#pragma once

#include "chess/position.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class PositionHistory {
public:
    explicit PositionHistory(const Position& initial) : positions_{initial} {}

    void push(const Position& position) { positions_.push_back(position); }
    void pop() {
        if (positions_.size() > 1) positions_.pop_back();
    }
    const Position& current() const noexcept { return positions_.back(); }
    std::size_t size() const noexcept { return positions_.size(); }

    std::size_t occurrences(std::uint64_t hash) const noexcept {
        std::size_t count = 0;
        for (const Position& position : positions_) {
            if (position.hash() == hash) ++count;
        }
        return count;
    }

    bool is_threefold() const noexcept {
        return occurrences(current().hash()) >= 3;
    }

private:
    std::vector<Position> positions_;
};

}  // namespace chess
