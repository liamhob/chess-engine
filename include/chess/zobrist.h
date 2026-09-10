#pragma once

#include "chess/position.h"

#include <array>
#include <cstdint>

namespace chess {

struct ZobristKeys {
    std::array<std::array<std::uint64_t, 64>, 12> pieces{};
    std::array<std::uint64_t, 16> castling{};
    std::array<std::uint64_t, 8> en_passant{};
    std::uint64_t side_to_move = 0;
};

const ZobristKeys& zobrist_keys() noexcept;

}  // namespace chess