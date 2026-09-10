#include "chess/attacks.h"
#include "chess/history.h"
#include "chess/movegen.h"

#include <cassert>

int main() {
    using namespace chess;

    const Bitboard rook_blocked = rook_attacks(0, square_bit(8) | square_bit(1));
    assert(rook_blocked == (square_bit(1) | square_bit(8)));
    const Bitboard bishop_open = bishop_attacks(27, 0);
    assert((bishop_open & square_bit(0)) != 0);
    assert((bishop_open & square_bit(63)) != 0);
    assert(queen_attacks(27, 0) == (rook_attacks(27, 0) | bishop_open));

    const Position start = Position::start();
    PositionHistory history(start);
    history.push(start);
    history.push(start);
    assert(history.occurrences(start.hash()) == 3);
    assert(history.is_threefold());
    history.pop();
    assert(!history.is_threefold());

    const Position after = apply_move(start, make_move(12, 28, MoveFlag::DoublePawnPush));
    history.push(after);
    assert(history.current().hash() == after.hash());
    assert(history.occurrences(after.hash()) == 1);

    return 0;
}
