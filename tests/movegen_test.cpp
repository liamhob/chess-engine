#include "chess/movegen.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

void check_perft(const chess::Position& position, int depth, std::uint64_t expected) {
    const std::uint64_t actual = chess::perft(position, depth);
    if (actual != expected) {
        std::cerr << "PERFT depth " << depth << ": expected " << expected
                  << ", got " << actual << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace chess;

    const Position position = Position::start();
    const auto moves = generate_pseudo_legal(position);
    if (moves.size() != 20 || generate_legal(position).size() != 20) {
        std::cerr << "start position move count mismatch\n";
        return 1;
    }
    check_perft(position, 1, 20);
    check_perft(position, 2, 400);
    check_perft(position, 3, 8902);
    check_perft(position, 4, 197281);
    check_perft(position, 5, 4865609);
    check_perft(position, 6, 119060324);

    return 0;
}
