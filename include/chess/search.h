#pragma once

#include "chess/mcts.h"
#include "chess/position.h"

#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace chess::mcts {

struct Evaluation {
    std::vector<Move> moves;
    std::vector<float> priors;
    float value = 0.0F;
    bool terminal = false;
};

using Evaluator = std::function<Evaluation(const Position&)>;

class Search {
public:
    Search(Position root_position, float c_puct = 1.5F);

    void run(int simulations, const Evaluator& evaluator);
    void run_parallel(int simulations, std::size_t worker_count, const Evaluator& evaluator);
    void add_root_noise(float alpha, float epsilon, std::uint64_t seed);
    Node& root() noexcept;
    const Node& root() const noexcept;
    std::vector<std::pair<Move, int>> root_visit_distribution() const;

private:
    void run_parallel_simulation(const Evaluator& evaluator);

    Position root_position_;
    Node root_;
    float c_puct_;
    mutable std::mutex search_mutex_;
};

}  // namespace chess::mcts
