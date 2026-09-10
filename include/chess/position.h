#pragma once

#include "chess/bitboard.h"
#include "chess/move.h"

#include <array>
#include <cstdint>

namespace chess {

enum class Color : std::uint8_t { White = 0, Black = 1 };

enum class Piece : std::uint8_t {
    WhitePawn = 0,
    WhiteKnight,
    WhiteBishop,
    WhiteRook,
    WhiteQueen,
    WhiteKing,
    BlackPawn,
    BlackKnight,
    BlackBishop,
    BlackRook,
    BlackQueen,
    BlackKing,
};

constexpr std::size_t piece_index(Color color, std::uint8_t type) noexcept {
    return static_cast<std::size_t>(static_cast<std::uint8_t>(color) * 6U + type);
}

struct Position {
    std::array<Bitboard, 12> pieces{};
    Bitboard white_occupancy = 0;
    Bitboard black_occupancy = 0;
    Color side_to_move = Color::White;
    std::uint8_t castling_rights = 0;
    Square en_passant = 64;
    std::uint16_t halfmove_clock = 0;

    void rebuild_occupancy() noexcept;
    std::uint64_t hash() const noexcept;
    static Position start() noexcept;
};

}  // namespace chess
