#pragma once

#include <cstdint>

namespace chess {

using Square = std::uint8_t;
using Move = std::uint16_t;

enum class MoveFlag : std::uint8_t {
    Quiet = 0,
    DoublePawnPush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EnPassant = 5,
    KnightPromotion = 8,
    BishopPromotion = 9,
    RookPromotion = 10,
    QueenPromotion = 11,
    KnightPromotionCapture = 12,
    BishopPromotionCapture = 13,
    RookPromotionCapture = 14,
    QueenPromotionCapture = 15,
};

constexpr Move make_move(Square from, Square to, MoveFlag flag = MoveFlag::Quiet) noexcept {
    return static_cast<Move>(from & 0x3fU)
        | static_cast<Move>((to & 0x3fU) << 6U)
        | static_cast<Move>((static_cast<std::uint8_t>(flag) & 0x0fU) << 12U);
}

constexpr Square move_from(Move move) noexcept {
    return static_cast<Square>(move & 0x3fU);
}

constexpr Square move_to(Move move) noexcept {
    return static_cast<Square>((move >> 6U) & 0x3fU);
}

constexpr MoveFlag move_flag(Move move) noexcept {
    return static_cast<MoveFlag>((move >> 12U) & 0x0fU);
}

static_assert(sizeof(Move) == sizeof(std::uint16_t));
static_assert(make_move(0, 63, MoveFlag::QueenPromotionCapture) == 0xFFC0U);

}  // namespace chess
