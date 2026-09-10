#include "chess/attacks.h"

#include "chess/move.h"

#include <array>
#include <bit>
#include <limits>
#include <vector>

namespace chess {
namespace {

constexpr Square step(Square square, int file_delta, int rank_delta) noexcept {
    const int file = file_of(square) + file_delta;
    const int rank = rank_of(square) + rank_delta;
    return file >= 0 && file < 8 && rank >= 0 && rank < 8
        ? static_cast<Square>(rank * 8 + file)
        : 64;
}

Bitboard ray_attacks(Square square, Bitboard occupied, const int directions[][2], int count) noexcept {
    Bitboard attacks = 0;
    for (int direction = 0; direction < count; ++direction) {
        Square target = square;
        while ((target = step(target, directions[direction][0], directions[direction][1])) != 64) {
            const Bitboard target_bit = square_bit(target);
            attacks |= target_bit;
            if ((occupied & target_bit) != 0) break;
        }
    }
    return attacks;
}

struct MagicEntry {
    Bitboard mask = 0;
    Bitboard magic = 0;
    std::uint8_t shift = 0;
    std::vector<Bitboard> attacks;
};

Bitboard relevant_mask(Square square, const int directions[][2], int count) noexcept {
    Bitboard mask = 0;
    for (int direction = 0; direction < count; ++direction) {
        Square target = square;
        while ((target = step(target, directions[direction][0], directions[direction][1])) != 64) {
            if (file_of(target) == 0 || file_of(target) == 7 || rank_of(target) == 0 || rank_of(target) == 7) break;
            mask |= square_bit(target);
        }
    }
    return mask;
}

std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state << 7U;
    state ^= state >> 9U;
    state ^= state << 8U;
    return state;
}

MagicEntry make_magic(Square square, const int directions[][2], int count, std::uint64_t& random_state) {
    MagicEntry entry;
    entry.mask = relevant_mask(square, directions, count);
    const int relevant_bits = std::popcount(entry.mask);
    entry.shift = static_cast<std::uint8_t>(64 - relevant_bits);
    const std::size_t table_size = std::size_t{1} << relevant_bits;
    std::vector<Bitboard> occupancies(table_size);
    std::vector<Bitboard> attacks(table_size);
    for (std::size_t index = 0; index < table_size; ++index) {
        Bitboard remaining = entry.mask;
        Bitboard occupancy = 0;
        for (int bit = 0; remaining != 0; ++bit) {
            const Bitboard bit_value = remaining & (~remaining + 1);
            if ((index & (std::size_t{1} << bit)) != 0) occupancy |= bit_value;
            remaining &= remaining - 1;
        }
        occupancies[index] = occupancy;
        attacks[index] = ray_attacks(square, occupancy, directions, count);
    }
    for (;;) {
        const Bitboard candidate = next_random(random_state) & next_random(random_state) & next_random(random_state);
        std::vector<Bitboard> used(table_size, std::numeric_limits<Bitboard>::max());
        bool valid = true;
        for (std::size_t index = 0; index < table_size; ++index) {
            const std::size_t magic_index = static_cast<std::size_t>((occupancies[index] * candidate) >> entry.shift);
            if (used[magic_index] != std::numeric_limits<Bitboard>::max() && used[magic_index] != attacks[index]) {
                valid = false;
                break;
            }
            used[magic_index] = attacks[index];
        }
        if (valid) {
            entry.magic = candidate;
            entry.attacks = std::move(used);
            return entry;
        }
    }
}

struct MagicTables {
    std::array<MagicEntry, 64> rooks;
    std::array<MagicEntry, 64> bishops;

    MagicTables() {
        constexpr int rook_directions[][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        constexpr int bishop_directions[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        std::uint64_t random_state = 0xA5A5F00D12345678ULL;
        for (Square square = 0; square < 64; ++square) {
            rooks[square] = make_magic(square, rook_directions, 4, random_state);
            bishops[square] = make_magic(square, bishop_directions, 4, random_state);
        }
    }
};

const MagicTables& magic_tables() {
    static const MagicTables tables;
    return tables;
}

Bitboard lookup(const MagicEntry& entry, Square square, Bitboard occupied) noexcept {
    const std::size_t index = static_cast<std::size_t>(((occupied & entry.mask) * entry.magic) >> entry.shift);
    return entry.attacks[index];
}

}  // namespace

Bitboard rook_attacks(Square square, Bitboard occupied) noexcept {
    return lookup(magic_tables().rooks[square], square, occupied);
}

Bitboard bishop_attacks(Square square, Bitboard occupied) noexcept {
    return lookup(magic_tables().bishops[square], square, occupied);
}

Bitboard queen_attacks(Square square, Bitboard occupied) noexcept {
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied);
}

}  // namespace chess
