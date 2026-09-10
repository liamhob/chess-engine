#include "chess/search.h"

#include "chess/concurrency.h"
#include "chess/movegen.h"

#include <stdexcept>
#include <future>

namespace chess::mcts {

Search::Search(Position root_position, float c_puct)
    : root_position_(std::move(root_position)), root_(root_position_.hash()), c_puct_(c_puct) {
    if (!(c_puct > 0.0F)) throw std::invalid_argument("c_puct must be positive");
}

void Search::run(int simulations, const Evaluator& evaluator) {
    if (simulations < 0) throw std::invalid_argument("simulation count cannot be negative");
    if (!evaluator) throw std::invalid_argument("MCTS evaluator is required");
    for (int simulation = 0; simulation < simulations; ++simulation) {
        Node* node = &root_;
        Position position = root_position_;
        while (node->is_expanded() && !node->terminal) {
            node = node->select_child(c_puct_);
            if (node == nullptr) break;
            position = apply_move(position, node->move);
        }
        if (node == nullptr) continue;
        const Evaluation evaluation = evaluator(position);
        node->terminal = evaluation.terminal;
        if (!node->terminal && !node->is_expanded()) {
            if (evaluation.moves.size() != evaluation.priors.size()) {
                throw std::invalid_argument("MCTS evaluator returned mismatched moves and priors");
            }
            node->expand(evaluation.moves, evaluation.priors);
            for (auto& child : node->children) {
                Position child_position = apply_move(position, child->move);
                child->state_hash = child_position.hash();
            }
        }
        node->backpropagate(evaluation.value);
    }
}

void Search::run_parallel(int simulations, std::size_t worker_count, const Evaluator& evaluator) {
    if (worker_count == 0) throw std::invalid_argument("worker count must be positive");
    ThreadPool pool(worker_count);
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(simulations));
    for (int simulation = 0; simulation < simulations; ++simulation) {
        futures.push_back(pool.submit([this, &evaluator] {
            run_parallel_simulation(evaluator);
        }));
    }
    for (auto& future : futures) future.get();
    pool.shutdown();
}

void Search::run_parallel_simulation(const Evaluator& evaluator) {
    Node* node = &root_;
    Position position = root_position_;
    std::vector<Node*> reserved_path;
    {
        std::lock_guard lock(search_mutex_);
        while (node->is_expanded() && !node->terminal) {
            node = node->select_child(c_puct_);
            if (node == nullptr) return;
            node->apply_virtual_loss();
            reserved_path.push_back(node);
            position = apply_move(position, node->move);
        }
    }

    Evaluation evaluation;
    try {
        evaluation = evaluator(position);
    } catch (...) {
        std::lock_guard lock(search_mutex_);
        for (Node* reserved : reserved_path) reserved->revert_virtual_loss();
        throw;
    }

    std::lock_guard lock(search_mutex_);
    node->terminal = evaluation.terminal;
    if (!node->terminal && !node->is_expanded()) {
        if (evaluation.moves.size() != evaluation.priors.size()) {
            for (Node* reserved : reserved_path) reserved->revert_virtual_loss();
            throw std::invalid_argument("MCTS evaluator returned mismatched moves and priors");
        }
        node->expand(evaluation.moves, evaluation.priors);
        for (auto& child : node->children) {
            child->state_hash = apply_move(position, child->move).hash();
        }
    }
    for (Node* reserved : reserved_path) reserved->revert_virtual_loss();
    node->backpropagate(evaluation.value);
}

void Search::add_root_noise(float alpha, float epsilon, std::uint64_t seed) {
    root_.add_dirichlet_noise(alpha, epsilon, seed);
}

Node& Search::root() noexcept {
    return root_;
}

const Node& Search::root() const noexcept {
    return root_;
}

std::vector<std::pair<Move, int>> Search::root_visit_distribution() const {
    std::lock_guard lock(search_mutex_);
    std::vector<std::pair<Move, int>> distribution;
    distribution.reserve(root_.children.size());
    for (const auto& child : root_.children) {
        distribution.emplace_back(child->move, child->visit_count);
    }
    return distribution;
}

}  // namespace chess::mcts
