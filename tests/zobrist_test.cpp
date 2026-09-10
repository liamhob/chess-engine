#include "chess/movegen.h"
#include "chess/zobrist.h"

#include <cassert>

int main() {
    using namespace chess;

    const Position start = Position::start();
    assert(start.hash() == start.hash());

    Position side_changed = start;
    side_changed.side_to_move = Color::Black;
    assert(start.hash() != side_changed.hash());

    Position rights_changed = start;
    rights_changed.castling_rights = 0;
    assert(start.hash() != rights_changed.hash());

    Position ep_changed = start;
    ep_changed.en_passant = 24;
    assert(start.hash() != ep_changed.hash());

    const Position after_move = apply_move(start, make_move(12, 28, MoveFlag::DoublePawnPush));
    assert(start.hash() != after_move.hash());

    return 0;
}