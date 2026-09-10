#include "chess/position.h"

namespace chess {

void Position::rebuild_occupancy() noexcept {
    white_occupancy = 0;
    black_occupancy = 0;
    for (std::uint8_t type = 0; type < 6; ++type) {
        white_occupancy |= pieces[piece_index(Color::White, type)];
        black_occupancy |= pieces[piece_index(Color::Black, type)];
    }
}

Position Position::start() noexcept {
    Position position;
    position.pieces[piece_index(Color::White, 0)] = 0x000000000000FF00ULL;
    position.pieces[piece_index(Color::White, 1)] = 0x0000000000000042ULL;
    position.pieces[piece_index(Color::White, 2)] = 0x0000000000000024ULL;
    position.pieces[piece_index(Color::White, 3)] = 0x0000000000000081ULL;
    position.pieces[piece_index(Color::White, 4)] = 0x0000000000000008ULL;
    position.pieces[piece_index(Color::White, 5)] = 0x0000000000000010ULL;
    position.pieces[piece_index(Color::Black, 0)] = 0x00FF000000000000ULL;
    position.pieces[piece_index(Color::Black, 1)] = 0x4200000000000000ULL;
    position.pieces[piece_index(Color::Black, 2)] = 0x2400000000000000ULL;
    position.pieces[piece_index(Color::Black, 3)] = 0x8100000000000000ULL;
    position.pieces[piece_index(Color::Black, 4)] = 0x0800000000000000ULL;
    position.pieces[piece_index(Color::Black, 5)] = 0x1000000000000000ULL;
    position.castling_rights = 0x0F;
    position.rebuild_occupancy();
    return position;
}

}  // namespace chess
