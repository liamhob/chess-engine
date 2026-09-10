#include "chess/zobrist.h"

#include <bit>

namespace chess {
namespace {

constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

ZobristKeys make_keys() noexcept {
    ZobristKeys keys;
    std::uint64_t state = 0xC0FFEE1234567890ULL;
    for (auto& piece : keys.pieces) {
        for (auto& square : piece) square = splitmix64(state);
    }
    for (auto& value : keys.castling) value = splitmix64(state);
    for (auto& value : keys.en_passant) value = splitmix64(state);
    keys.side_to_move = splitmix64(state);
    return keys;
}

}  // namespace

const ZobristKeys& zobrist_keys() noexcept {
    static const ZobristKeys keys = make_keys();
    return keys;
}

std::uint64_t Position::hash() const noexcept {
    const auto& keys = zobrist_keys();
    std::uint64_t value = 0;
    for (std::size_t piece = 0; piece < pieces.size(); ++piece) {
        Bitboard remaining = pieces[piece];
        while (remaining != 0) {
            const std::uint8_t square = static_cast<std::uint8_t>(std::countr_zero(remaining));
            value ^= keys.pieces[piece][square];
            remaining &= remaining - 1;
        }
    }
    value ^= keys.castling[castling_rights & 0x0F];
    if (en_passant < 64) value ^= keys.en_passant[file_of(en_passant)];
    if (side_to_move == Color::Black) value ^= keys.side_to_move;
    return value;
}

}  // namespace chess