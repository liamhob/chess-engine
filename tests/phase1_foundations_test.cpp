#include "chess/bitboard.h"
#include "chess/move.h"

#include <cassert>
#include <cstdint>

int main() {
    using namespace chess;

    static_assert(square_bit(0) == 1ULL);
    static_assert(square_bit(63) == (1ULL << 63));
    static_assert(rank_of(0) == 0);
    static_assert(rank_of(63) == 7);
    static_assert(file_of(0) == 0);
    static_assert(file_of(63) == 7);

    assert(file_mask(0) == 0x0101010101010101ULL);
    assert(rank_mask(7) == 0xFF00000000000000ULL);
    assert(SquareMasks[36] == (1ULL << 36));

    const Move move = make_move(12, 28, MoveFlag::DoublePawnPush);
    assert(move_from(move) == 12);
    assert(move_to(move) == 28);
    assert(move_flag(move) == MoveFlag::DoublePawnPush);
    assert(make_move(64, 127, MoveFlag::QueenPromotionCapture) == 0xFFC0U);

    return 0;
}
