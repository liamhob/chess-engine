#pragma once

#include "chess/move.h"
#include "chess/position.h"

#include <vector>

namespace chess {

std::vector<Move> generate_pseudo_legal(const Position& position);
Position apply_move(const Position& position, Move move);
std::vector<Move> generate_legal(const Position& position);
bool is_in_check(const Position& position, Color color);
bool is_fifty_move_draw(const Position& position) noexcept;
std::uint64_t perft(const Position& position, int depth);

}  // namespace chess
