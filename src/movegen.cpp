#include "chess/movegen.h"

#include <algorithm>
#include <bit>

namespace chess {
namespace {

constexpr int direction(Color color) noexcept {
    return color == Color::White ? 1 : -1;
}

constexpr Square shifted_square(Square square, int file_delta, int rank_delta) noexcept {
    const int file = file_of(square) + file_delta;
    const int rank = rank_of(square) + rank_delta;
    return file >= 0 && file < 8 && rank >= 0 && rank < 8
        ? static_cast<Square>(rank * 8 + file)
        : 64;
}

void add_step_moves(const Position& position, std::vector<Move>& moves, Bitboard sources,
                    Color color, const int deltas[][2], int delta_count) {
    const Bitboard own = color == Color::White ? position.white_occupancy : position.black_occupancy;
    const Bitboard enemy = color == Color::White ? position.black_occupancy : position.white_occupancy;
    while (sources != 0) {
        const Square from = static_cast<Square>(std::countr_zero(sources));
        sources &= sources - 1;
        for (int index = 0; index < delta_count; ++index) {
            const Square to = shifted_square(from, deltas[index][0], deltas[index][1]);
            if (to == 64 || (own & square_bit(to)) != 0) {
                continue;
            }
            moves.push_back(make_move(from, to, (enemy & square_bit(to)) != 0 ? MoveFlag::Capture : MoveFlag::Quiet));
        }
    }
}

void add_sliding_moves(const Position& position, std::vector<Move>& moves, Bitboard sources,
                       Color color, const int deltas[][2], int delta_count) {
    const Bitboard own = color == Color::White ? position.white_occupancy : position.black_occupancy;
    const Bitboard enemy = color == Color::White ? position.black_occupancy : position.white_occupancy;
    while (sources != 0) {
        const Square from = static_cast<Square>(std::countr_zero(sources));
        sources &= sources - 1;
        for (int index = 0; index < delta_count; ++index) {
            Square to = from;
            while (true) {
                to = shifted_square(to, deltas[index][0], deltas[index][1]);
                if (to == 64 || (own & square_bit(to)) != 0) {
                    break;
                }
                const bool capture = (enemy & square_bit(to)) != 0;
                moves.push_back(make_move(from, to, capture ? MoveFlag::Capture : MoveFlag::Quiet));
                if (capture) {
                    break;
                }
            }
        }
    }
}

void add_pawn_moves(const Position& position, std::vector<Move>& moves, Bitboard pawns, Color color) {
    const Bitboard occupied = position.white_occupancy | position.black_occupancy;
    const Bitboard enemy = color == Color::White ? position.black_occupancy : position.white_occupancy;
    const int step = direction(color);
    const int start_rank = color == Color::White ? 1 : 6;
    while (pawns != 0) {
        const Square from = static_cast<Square>(std::countr_zero(pawns));
        pawns &= pawns - 1;
        const Square one = shifted_square(from, 0, step);
        if (one != 64 && (occupied & square_bit(one)) == 0) {
            if (rank_of(one) == (color == Color::White ? 7 : 0)) {
                moves.push_back(make_move(from, one, MoveFlag::KnightPromotion));
                moves.push_back(make_move(from, one, MoveFlag::BishopPromotion));
                moves.push_back(make_move(from, one, MoveFlag::RookPromotion));
                moves.push_back(make_move(from, one, MoveFlag::QueenPromotion));
            } else {
                moves.push_back(make_move(from, one));
            }
            const Square two = shifted_square(from, 0, step * 2);
            if (rank_of(from) == start_rank && (occupied & square_bit(two)) == 0) {
                moves.push_back(make_move(from, two, MoveFlag::DoublePawnPush));
            }
        }
        for (const int file_delta : {-1, 1}) {
            const Square capture = shifted_square(from, file_delta, step);
            if (capture != 64 && (enemy & square_bit(capture)) != 0) {
                if (rank_of(capture) == (color == Color::White ? 7 : 0)) {
                    moves.push_back(make_move(from, capture, MoveFlag::KnightPromotionCapture));
                    moves.push_back(make_move(from, capture, MoveFlag::BishopPromotionCapture));
                    moves.push_back(make_move(from, capture, MoveFlag::RookPromotionCapture));
                    moves.push_back(make_move(from, capture, MoveFlag::QueenPromotionCapture));
                } else {
                    moves.push_back(make_move(from, capture, MoveFlag::Capture));
                }
            }
            if (capture != 64 && capture == position.en_passant) {
                moves.push_back(make_move(from, capture, MoveFlag::EnPassant));
            }
        }
    }
}

}  // namespace

std::vector<Move> generate_pseudo_legal(const Position& position) {
    std::vector<Move> moves;
    const Color color = position.side_to_move;
    const auto index = [color](std::uint8_t type) { return piece_index(color, type); };
    add_pawn_moves(position, moves, position.pieces[index(0)], color);
    constexpr int knight_deltas[][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
    constexpr int king_deltas[][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
    constexpr int bishop_deltas[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int rook_deltas[][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    add_step_moves(position, moves, position.pieces[index(1)], color, knight_deltas, 8);
    add_sliding_moves(position, moves, position.pieces[index(2)], color, bishop_deltas, 4);
    add_sliding_moves(position, moves, position.pieces[index(3)], color, rook_deltas, 4);
    add_sliding_moves(position, moves, position.pieces[index(4)], color, bishop_deltas, 4);
    add_sliding_moves(position, moves, position.pieces[index(4)], color, rook_deltas, 4);
    add_step_moves(position, moves, position.pieces[index(5)], color, king_deltas, 8);
    const Bitboard occupied = position.white_occupancy | position.black_occupancy;
    const Bitboard enemy = color == Color::White ? position.black_occupancy : position.white_occupancy;
    const std::uint8_t king_side = color == Color::White ? 1 : 4;
    const std::uint8_t queen_side = color == Color::White ? 2 : 8;
    const Square king_square = color == Color::White ? 4 : 60;
    if ((position.pieces[index(5)] & square_bit(king_square)) != 0) {
        if ((position.castling_rights & king_side) != 0 &&
            (position.pieces[index(3)] & square_bit(king_square + 3)) != 0 &&
            (occupied & (square_bit(king_square + 1) | square_bit(king_square + 2))) == 0) {
            moves.push_back(make_move(king_square, static_cast<Square>(king_square + 2), MoveFlag::KingCastle));
        }
        if ((position.castling_rights & queen_side) != 0 &&
            (position.pieces[index(3)] & square_bit(king_square - 4)) != 0 &&
            (occupied & (square_bit(king_square - 1) | square_bit(king_square - 2) | square_bit(king_square - 3))) == 0) {
            moves.push_back(make_move(king_square, static_cast<Square>(king_square - 2), MoveFlag::QueenCastle));
        }
    }
    return moves;
}

Position apply_move(const Position& position, Move move) {
    Position next = position;
    const Color color = position.side_to_move;
    const Color enemy = color == Color::White ? Color::Black : Color::White;
    const Square from = move_from(move);
    const Square to = move_to(move);
    const MoveFlag flag = move_flag(move);
    const std::uint8_t flag_value = static_cast<std::uint8_t>(flag);
    const Bitboard from_bit = square_bit(from);
    const Bitboard to_bit = square_bit(to);
    int moving_type = -1;
    for (int type = 0; type < 6; ++type) {
        if ((next.pieces[piece_index(color, static_cast<std::uint8_t>(type))] & from_bit) != 0) {
            moving_type = type;
            next.pieces[piece_index(color, static_cast<std::uint8_t>(type))] &= ~from_bit;
            break;
        }
    }
    if (moving_type < 0) {
        return position;
    }
    for (int type = 0; type < 6; ++type) {
        next.pieces[piece_index(enemy, static_cast<std::uint8_t>(type))] &= ~to_bit;
    }
    if (flag == MoveFlag::EnPassant) {
        const Square captured = shifted_square(to, 0, color == Color::White ? -1 : 1);
        next.pieces[piece_index(enemy, 0)] &= ~square_bit(captured);
    }
    if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        const Square rook_from = flag == MoveFlag::KingCastle
            ? static_cast<Square>(from + 3) : static_cast<Square>(from - 4);
        const Square rook_to = flag == MoveFlag::KingCastle
            ? static_cast<Square>(from + 1) : static_cast<Square>(from - 1);
        const Bitboard rook_from_bit = square_bit(rook_from);
        next.pieces[piece_index(color, 3)] &= ~rook_from_bit;
        next.pieces[piece_index(color, 3)] |= square_bit(rook_to);
    }
    if (moving_type == 5) {
        next.castling_rights &= color == Color::White ? 0x0CU : 0x03U;
    }
    if (moving_type == 3 && from == 0) next.castling_rights &= ~0x02U;
    if (moving_type == 3 && from == 7) next.castling_rights &= ~0x01U;
    if (moving_type == 3 && from == 56) next.castling_rights &= ~0x08U;
    if (moving_type == 3 && from == 63) next.castling_rights &= ~0x04U;
    if (to == 0) next.castling_rights &= ~0x02U;
    if (to == 7) next.castling_rights &= ~0x01U;
    if (to == 56) next.castling_rights &= ~0x08U;
    if (to == 63) next.castling_rights &= ~0x04U;
    int placed_type = moving_type;
    if (static_cast<std::uint8_t>(flag) >= 8) {
        placed_type = 1 + (static_cast<std::uint8_t>(flag) - 8) % 4;
    }
    next.pieces[piece_index(color, static_cast<std::uint8_t>(placed_type))] |= to_bit;
    next.en_passant = 64;
    if (moving_type == 0 && std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
        next.en_passant = static_cast<Square>((from + to) / 2);
    }
    const bool capture = flag == MoveFlag::Capture || flag == MoveFlag::EnPassant || flag_value >= 12;
    next.halfmove_clock = moving_type == 0 || capture
        ? 0
        : static_cast<std::uint16_t>(std::min<std::uint32_t>(1000U, next.halfmove_clock + 1U));
    next.side_to_move = enemy;
    next.rebuild_occupancy();
    return next;
}

namespace {

bool is_attacked(const Position& position, Square target, Color by_color) {
    const auto index = [by_color](std::uint8_t type) { return piece_index(by_color, type); };
    const Bitboard target_bit = square_bit(target);
    const int pawn_direction = by_color == Color::White ? 1 : -1;
    for (const int file_delta : {-1, 1}) {
        const Square source = shifted_square(target, file_delta, -pawn_direction);
        if (source != 64 && (position.pieces[index(0)] & square_bit(source)) != 0) {
            return true;
        }
    }
    constexpr int knight_deltas[][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
    for (const auto& delta : knight_deltas) {
        const Square source = shifted_square(target, delta[0], delta[1]);
        if (source != 64 && (position.pieces[index(1)] & square_bit(source)) != 0) {
            return true;
        }
    }
    constexpr int king_deltas[][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
    for (const auto& delta : king_deltas) {
        const Square source = shifted_square(target, delta[0], delta[1]);
        if (source != 64 && (position.pieces[index(5)] & square_bit(source)) != 0) {
            return true;
        }
    }
    constexpr int bishop_deltas[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int rook_deltas[][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& delta : bishop_deltas) {
        Square source = target;
        while ((source = shifted_square(source, delta[0], delta[1])) != 64) {
            const Bitboard bit = square_bit(source);
            if ((position.pieces[index(2)] | position.pieces[index(4)]) & bit) return true;
            if ((position.white_occupancy | position.black_occupancy) & bit) break;
        }
    }
    for (const auto& delta : rook_deltas) {
        Square source = target;
        while ((source = shifted_square(source, delta[0], delta[1])) != 64) {
            const Bitboard bit = square_bit(source);
            if ((position.pieces[index(3)] | position.pieces[index(4)]) & bit) return true;
            if ((position.white_occupancy | position.black_occupancy) & bit) break;
        }
    }
    return false;
}

}  // namespace

std::vector<Move> generate_legal(const Position& position) {
    std::vector<Move> legal;
    const Color color = position.side_to_move;
    for (const Move move : generate_pseudo_legal(position)) {
        if (move_flag(move) == MoveFlag::KingCastle || move_flag(move) == MoveFlag::QueenCastle) {
            const Square from = move_from(move);
            const int transit_delta = move_flag(move) == MoveFlag::KingCastle ? 1 : -1;
            const Color enemy = color == Color::White ? Color::Black : Color::White;
            if (is_attacked(position, from, enemy) ||
                is_attacked(position, static_cast<Square>(from + transit_delta), enemy)) {
                continue;
            }
        }
        const Position next = apply_move(position, move);
        const Bitboard king = next.pieces[piece_index(color, 5)];
        if (king != 0 && !is_attacked(next, static_cast<Square>(std::countr_zero(king)), next.side_to_move)) {
            legal.push_back(move);
        }
    }
    return legal;
}

bool is_in_check(const Position& position, Color color) {
    const Bitboard king = position.pieces[piece_index(color, 5)];
    if (king == 0) return false;
    const Color enemy = color == Color::White ? Color::Black : Color::White;
    return is_attacked(position, static_cast<Square>(std::countr_zero(king)), enemy);
}

bool is_fifty_move_draw(const Position& position) noexcept {
    return position.halfmove_clock >= 100;
}

std::uint64_t perft(const Position& position, int depth) {
    if (depth == 0) return 1;
    std::uint64_t nodes = 0;
    for (const Move move : generate_legal(position)) {
        nodes += perft(apply_move(position, move), depth - 1);
    }
    return nodes;
}

}  // namespace chess
