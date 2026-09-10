#include "chess/search.h"

#include <cassert>

int main() {
    using namespace chess;
    using namespace chess::mcts;

    Search search(Position::start());
    int evaluations = 0;
    search.run(8, [&evaluations](const Position& position) {
        ++evaluations;
        Evaluation result;
        if (position.side_to_move == Color::White) {
            result.moves = {make_move(12, 28, MoveFlag::DoublePawnPush), make_move(12, 20)};
            result.priors = {0.7F, 0.3F};
        }
        result.value = 0.25F;
        return result;
    });

    assert(evaluations == 8);
    assert(search.root().visit_count == 8);
    assert(search.root().children.size() == 2);
    assert(search.root().children[0]->state_hash != 0);
    assert(search.root().children[0]->visit_count + search.root().children[1]->visit_count == 7);

    Search parallel_search(Position::start());
    parallel_search.run_parallel(32, 4, [](const Position& position) {
        Evaluation result;
        if (position.side_to_move == Color::White) {
            result.moves = {make_move(12, 28, MoveFlag::DoublePawnPush), make_move(12, 20)};
            result.priors = {0.6F, 0.4F};
        }
        return result;
    });
    assert(parallel_search.root().visit_count == 32);
    assert(parallel_search.root().children.size() == 2);
    parallel_search.add_root_noise(0.3F, 0.25F, 19);

    return 0;
}
