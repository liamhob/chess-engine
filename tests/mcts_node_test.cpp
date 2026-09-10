#include "chess/mcts.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
    using namespace chess;
    using namespace chess::mcts;

    Node root(0x1234, 1.0F);
    root.expand({make_move(0, 8), make_move(1, 9)}, {0.75F, 0.25F});
    assert(root.is_expanded());
    assert(root.children.size() == 2);
    assert(root.children[0]->parent == &root);
    assert(root.children[0]->prior_probability == 0.75F);

    root.visit_count = 10;
    root.children[0]->visit_count = 10;
    root.children[0]->value_sum = 5.0;
    root.children[1]->visit_count = 0;
    Node* selected = root.select_child(1.0F);
    assert(selected == root.children[1].get());
    selected->apply_virtual_loss();
    assert(selected->virtual_loss.load() == 1);
    selected->revert_virtual_loss();
    assert(selected->virtual_loss.load() == 0);

    selected->backpropagate(1.0F);
    assert(selected->visit_count == 1);
    assert(selected->value_sum == 1.0);
    assert(root.visit_count == 11);
    assert(root.value_sum == -1.0);

    const float first_prior = root.children[0]->prior_probability;
    root.add_dirichlet_noise(0.3F, 0.25F, 42);
    assert(std::abs(root.children[0]->prior_probability - first_prior) > 1e-6F);
    const float prior_sum = root.children[0]->prior_probability + root.children[1]->prior_probability;
    assert(std::abs(prior_sum - 1.0F) < 1e-5F);

    bool rejected = false;
    try {
        root.children[0]->add_dirichlet_noise(0.3F, 0.25F, 42);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
