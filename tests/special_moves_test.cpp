#include "chess/movegen.h"

#include <cstdlib>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace chess;

    Position promotion;
    promotion.side_to_move = Color::White;
    promotion.pieces[piece_index(Color::White, 0)] = square_bit(48);
    promotion.pieces[piece_index(Color::White, 5)] = square_bit(4);
    promotion.pieces[piece_index(Color::Black, 5)] = square_bit(60);
    promotion.rebuild_occupancy();
    const auto promotion_moves = generate_legal(promotion);
    int promotion_count = 0;
    Move queen_promotion = 0;
    for (const Move move : promotion_moves) {
        if (static_cast<std::uint8_t>(move_flag(move)) >= 8) {
            ++promotion_count;
            if (move_flag(move) == MoveFlag::QueenPromotion) queen_promotion = move;
        }
    }
    check(promotion_count == 4, "promotion move count");
    const Position promoted = apply_move(promotion, queen_promotion);
    check((promoted.pieces[piece_index(Color::White, 4)] & square_bit(56)) != 0, "queen promotion placement");

    Position castling;
    castling.side_to_move = Color::White;
    castling.castling_rights = 0x0F;
    castling.pieces[piece_index(Color::White, 3)] = square_bit(0) | square_bit(7);
    castling.pieces[piece_index(Color::White, 5)] = square_bit(4);
    castling.pieces[piece_index(Color::Black, 3)] = square_bit(56) | square_bit(63);
    castling.pieces[piece_index(Color::Black, 5)] = square_bit(60);
    castling.rebuild_occupancy();
    const auto castling_moves = generate_legal(castling);
    int castle_count = 0;
    for (const Move move : castling_moves) {
        castle_count += move_flag(move) == MoveFlag::KingCastle || move_flag(move) == MoveFlag::QueenCastle;
    }
    check(castle_count == 2, "castling move count");
    const Position castled = apply_move(castling, make_move(4, 6, MoveFlag::KingCastle));
    check((castled.pieces[piece_index(Color::White, 5)] & square_bit(6)) != 0, "castled king placement");
    check((castled.pieces[piece_index(Color::White, 3)] & square_bit(5)) != 0, "castled rook placement");

    return 0;
}
