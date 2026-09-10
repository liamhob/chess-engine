#pragma once

#include "chess/bitboard.h"
#include "chess/move.h"

#include <cstdint>

namespace chess {

Bitboard rook_attacks(Square square, Bitboard occupied) noexcept;
Bitboard bishop_attacks(Square square, Bitboard occupied) noexcept;
Bitboard queen_attacks(Square square, Bitboard occupied) noexcept;

}  // namespace chess
