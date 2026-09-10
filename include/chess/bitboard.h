#pragma once

#include <array>
#include <cstdint>

namespace chess {

using Bitboard = std::uint64_t;

constexpr Bitboard square_bit(std::uint8_t square) noexcept {
    return Bitboard{1} << square;
}

constexpr int rank_of(std::uint8_t square) noexcept {
    return square / 8;
}

constexpr int file_of(std::uint8_t square) noexcept {
    return square % 8;
}

constexpr Bitboard file_mask(int file) noexcept {
    return 0x0101010101010101ULL << file;
}

constexpr Bitboard rank_mask(int rank) noexcept {
    return 0xFFULL << (rank * 8);
}

constexpr std::array<Bitboard, 64> make_square_masks() noexcept {
    std::array<Bitboard, 64> masks{};
    for (std::uint8_t square = 0; square < 64; ++square) {
        masks[square] = square_bit(square);
    }
    return masks;
}

inline constexpr auto SquareMasks = make_square_masks();

}  // namespace chess
