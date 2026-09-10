#include "chess/position.h"

#include <cassert>

int main() {
    using namespace chess;

    const Position position = Position::start();
    assert(position.side_to_move == Color::White);
    assert(position.pieces[piece_index(Color::White, 0)] == 0x000000000000FF00ULL);
    assert(position.pieces[piece_index(Color::Black, 5)] == 0x1000000000000000ULL);
    assert(position.white_occupancy == 0x000000000000FFFFULL);
    assert(position.black_occupancy == 0xFFFF000000000000ULL);
    assert((position.white_occupancy & position.black_occupancy) == 0);
    assert(position.castling_rights == 0x0F);
    assert(position.en_passant == 64);

    return 0;
}
