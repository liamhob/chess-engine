#pragma once

#include "chess/move.h"

#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>

namespace chess::mcts {

struct Node {
    std::uint64_t state_hash = 0;
    Move move = 0;
    float prior_probability = 0.0F;
    int visit_count = 0;
    double value_sum = 0.0;
    std::atomic<int> virtual_loss{0};
    bool terminal = false;
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;

    explicit Node(std::uint64_t hash = 0, float prior = 1.0F, Move action = 0, Node* owner = nullptr);

    bool is_expanded() const noexcept;
    void expand(const std::vector<Move>& moves, const std::vector<float>& priors);
    Node* select_child(float c_puct) noexcept;
    void backpropagate(float leaf_value) noexcept;
    void apply_virtual_loss(int amount = 1) noexcept;
    void revert_virtual_loss(int amount = 1) noexcept;
    void add_dirichlet_noise(float alpha, float epsilon, std::uint64_t seed);
};

}  // namespace chess::mcts
