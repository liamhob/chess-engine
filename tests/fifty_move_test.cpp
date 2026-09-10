#include "chess/movegen.h"

#include <cassert>

int main() {
    using namespace chess;

    Position position;
    position.pieces[piece_index(Color::White, 5)] = square_bit(4);
    position.pieces[piece_index(Color::Black, 5)] = square_bit(60);
    position.pieces[piece_index(Color::White, 1)] = square_bit(6);
    position.rebuild_occupancy();

    assert(position.halfmove_clock == 0);
    assert(!is_fifty_move_draw(position));
    Position next = apply_move(position, make_move(6, 21));
    assert(next.halfmove_clock == 1);
    next.halfmove_clock = 99;
    next = apply_move(next, make_move(21, 6));
    assert(next.halfmove_clock == 100);
    assert(is_fifty_move_draw(next));

    next.halfmove_clock = 99;
    next.pieces[piece_index(Color::White, 0)] = square_bit(8);
    next.rebuild_occupancy();
    next = apply_move(next, make_move(8, 16));
    assert(next.halfmove_clock == 0);

    Position capture_position;
    capture_position.pieces[piece_index(Color::White, 5)] = square_bit(4);
    capture_position.pieces[piece_index(Color::Black, 5)] = square_bit(60);
    capture_position.pieces[piece_index(Color::White, 1)] = square_bit(6);
    capture_position.pieces[piece_index(Color::Black, 2)] = square_bit(21);
    capture_position.halfmove_clock = 99;
    capture_position.rebuild_occupancy();
    const Position after_capture = apply_move(capture_position, make_move(6, 21, MoveFlag::Capture));
    assert(after_capture.halfmove_clock == 0);
    assert(!is_fifty_move_draw(after_capture));

    return 0;
}
