#include "chess/movegen.h"

#include <cstdint>
#include <iostream>

int main() {
    const chess::Position position = chess::Position::start();
    constexpr std::uint64_t expected[] = {1, 20, 400, 8902, 197281, 4865609, 119060324};
    for (int depth = 0; depth <= 6; ++depth) {
        const std::uint64_t actual = chess::perft(position, depth);
        if (actual != expected[depth]) {
            std::cerr << "PERFT depth " << depth << ": expected " << expected[depth]
                      << ", got " << actual << '\n';
            return 1;
        }
        std::cout << "PERFT depth " << depth << ": " << actual << '\n';
    }
    return 0;
}
