#include "chess/mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace chess::mcts {

Node::Node(std::uint64_t hash, float prior, Move action, Node* owner)
    : state_hash(hash), move(action), prior_probability(prior), parent(owner) {}

bool Node::is_expanded() const noexcept {
    return !children.empty();
}

void Node::expand(const std::vector<Move>& moves, const std::vector<float>& priors) {
    if (moves.size() != priors.size()) {
        throw std::invalid_argument("MCTS moves and priors must have equal sizes");
    }
    if (is_expanded()) {
        throw std::logic_error("MCTS node cannot be expanded twice");
    }
    children.reserve(moves.size());
    for (std::size_t index = 0; index < moves.size(); ++index) {
        children.push_back(std::make_unique<Node>(0, priors[index], moves[index], this));
    }
}

Node* Node::select_child(float c_puct) noexcept {
    if (children.empty()) return nullptr;
    Node* best = children.front().get();
    double best_score = -std::numeric_limits<double>::infinity();
    const double parent_scale = std::sqrt(static_cast<double>(std::max(1, visit_count)));
    for (const auto& child : children) {
        const double q = child->visit_count == 0 ? 0.0
            : (child->value_sum - child->virtual_loss.load(std::memory_order_relaxed)) / child->visit_count;
        const double exploration = static_cast<double>(c_puct) * child->prior_probability * parent_scale
            / (1.0 + child->visit_count);
        const double score = q + exploration;
        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }
    return best;
}

void Node::apply_virtual_loss(int amount) noexcept {
    virtual_loss.fetch_add(amount, std::memory_order_relaxed);
}

void Node::revert_virtual_loss(int amount) noexcept {
    virtual_loss.fetch_sub(amount, std::memory_order_relaxed);
}

void Node::backpropagate(float leaf_value) noexcept {
    Node* node = this;
    double value = leaf_value;
    while (node != nullptr) {
        ++node->visit_count;
        node->value_sum += value;
        value = -value;
        node = node->parent;
    }
}

void Node::add_dirichlet_noise(float alpha, float epsilon, std::uint64_t seed) {
    if (parent != nullptr) {
        throw std::logic_error("Dirichlet noise is only valid at the root");
    }
    if (children.empty()) return;
    if (!(alpha > 0.0F) || !(epsilon >= 0.0F && epsilon <= 1.0F)) {
        throw std::invalid_argument("Dirichlet alpha and epsilon are invalid");
    }
    std::mt19937_64 generator(seed);
    std::gamma_distribution<float> gamma(alpha, 1.0F);
    std::vector<float> noise(children.size());
    float total = 0.0F;
    for (float& value : noise) {
        value = gamma(generator);
        total += value;
    }
    if (total == 0.0F) return;
    for (std::size_t index = 0; index < children.size(); ++index) {
        children[index]->prior_probability = (1.0F - epsilon) * children[index]->prior_probability
            + epsilon * noise[index] / total;
    }
}

}  // namespace chess::mcts
