#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <bit>
#include <stdexcept>

#include "chess/inference.h"
#include "chess/movegen.h"
#include "chess/search.h"

namespace py = pybind11;

namespace {

class GameState {
public:
    GameState() : position_(chess::Position::start()) {}
    explicit GameState(chess::Position position) : position_(std::move(position)) {}

    std::uint64_t hash() const noexcept { return position_.hash(); }
    std::vector<chess::Move> legal_moves() const { return chess::generate_legal(position_); }
    GameState apply(chess::Move move) const { return GameState(chess::apply_move(position_, move)); }
    const chess::Position& position() const noexcept { return position_; }

    py::array_t<float> planes() const {
        py::array_t<float> result({14, 8, 8});
        auto view = result.mutable_unchecked<3>();
        for (int plane = 0; plane < 14; ++plane) {
            for (int rank = 0; rank < 8; ++rank) {
                for (int file = 0; file < 8; ++file) {
                    view(plane, rank, file) = 0.0F;
                }
            }
        }
        for (int piece = 0; piece < 12; ++piece) {
            chess::Bitboard remaining = position_.pieces[piece];
            while (remaining != 0) {
                const int square = static_cast<int>(std::countr_zero(remaining));
                view(piece, square / 8, square % 8) = 1.0F;
                remaining &= remaining - 1;
            }
        }
        view(12, 0, 0) = static_cast<float>(position_.castling_rights);
        view(13, 0, 0) = position_.side_to_move == chess::Color::Black ? 1.0F : 0.0F;
        return result;
    }

    int castling_rights() const noexcept { return position_.castling_rights; }
    int side_to_move() const noexcept { return static_cast<int>(position_.side_to_move); }
    bool in_check() const noexcept { return chess::is_in_check(position_, position_.side_to_move); }
    int halfmove_clock() const noexcept { return position_.halfmove_clock; }
    bool fifty_move_draw() const noexcept { return chess::is_fifty_move_draw(position_); }

private:
    chess::Position position_;
};

class MctsBinding {
public:
    explicit MctsBinding(const GameState& state) : search_(state.position()) {}

    void run(int simulations, const py::function& evaluator) {
        search_.run(simulations, [&evaluator](const chess::Position& position) {
            py::gil_scoped_acquire gil;
            const py::tuple result = evaluator(GameState(position)).cast<py::tuple>();
            chess::mcts::Evaluation evaluation;
            evaluation.moves = result[0].cast<std::vector<chess::Move>>();
            evaluation.priors = result[1].cast<std::vector<float>>();
            evaluation.value = result[2].cast<float>();
            evaluation.terminal = result.size() > 3 ? result[3].cast<bool>() : false;
            return evaluation;
        });
    }

    void run_parallel(int simulations, std::size_t worker_count, chess::bridge::InferenceQueue& queue) {
        py::gil_scoped_release release;
        search_.run_parallel(simulations, worker_count, [&queue](const chess::Position& position) {
            chess::mcts::Evaluation evaluation;
            if (chess::is_fifty_move_draw(position)) {
                evaluation.terminal = true;
                evaluation.value = 0.0F;
                return evaluation;
            }
            evaluation.moves = chess::generate_legal(position);
            if (evaluation.moves.empty()) {
                evaluation.terminal = true;
                evaluation.value = chess::is_in_check(position, position.side_to_move) ? 1.0F : 0.0F;
                return evaluation;
            }
            auto future = queue.submit(position);
            chess::bridge::InferenceResult result = future.get();
            if (result.policy.size() != evaluation.moves.size()) {
                throw std::invalid_argument("batched evaluator returned priors that do not match legal moves");
            }
            evaluation.priors = std::move(result.policy);
            evaluation.value = result.value;
            return evaluation;
        });
    }

    int visit_count() const noexcept { return search_.root().visit_count; }
    std::vector<std::pair<chess::Move, int>> visit_distribution() const {
        return search_.root_visit_distribution();
    }
    void add_root_noise(float alpha, float epsilon, std::uint64_t seed) {
        search_.add_root_noise(alpha, epsilon, seed);
    }

private:
    chess::mcts::Search search_;
};

}  // namespace

PYBIND11_MODULE(alphazero_cpp, module) {
    py::class_<GameState>(module, "GameState")
        .def(py::init<>())
        .def("hash", &GameState::hash)
        .def("legal_moves", &GameState::legal_moves)
        .def("apply", &GameState::apply)
        .def("planes", &GameState::planes)
        .def("castling_rights", &GameState::castling_rights)
        .def("side_to_move", &GameState::side_to_move)
        .def("in_check", &GameState::in_check)
        .def("halfmove_clock", &GameState::halfmove_clock)
        .def("fifty_move_draw", &GameState::fifty_move_draw);
        

    py::class_<MctsBinding>(module, "MCTS")
        .def(py::init<const GameState&>())
        .def("run", &MctsBinding::run)
        .def("run_parallel", &MctsBinding::run_parallel)
        .def("visit_count", &MctsBinding::visit_count)
        .def("visit_distribution", &MctsBinding::visit_distribution)
        .def("add_root_noise", &MctsBinding::add_root_noise);

    py::class_<chess::bridge::InferenceQueue>(module, "InferenceQueue")
        .def(py::init<>())
        .def("submit", [](chess::bridge::InferenceQueue& queue, const GameState& state) {
            return queue.submit_id(state.position());
        })
        .def("pop_batch", [](chess::bridge::InferenceQueue& queue, std::size_t max_batch) {
            std::vector<chess::bridge::InferenceRequest> batch;
            {
                py::gil_scoped_release release;
                batch = queue.pop_batch(max_batch);
            }
            py::list result;
            for (auto& request : batch) {
                result.append(py::make_tuple(request.request_id, GameState(request.position)));
            }
            return result;
        })
        .def("respond", [](chess::bridge::InferenceQueue& queue, std::uint64_t request_id,
                            std::vector<float> policy, float value) {
            queue.respond(request_id, {std::move(policy), value});
        })
        .def("wait_result", [](chess::bridge::InferenceQueue& queue, std::uint64_t request_id) {
            chess::bridge::InferenceResult result;
            {
                py::gil_scoped_release release;
                result = queue.wait_result(request_id);
            }
            return py::make_tuple(result.policy, result.value);
        })
        .def("close", &chess::bridge::InferenceQueue::close)
        .def("closed", &chess::bridge::InferenceQueue::closed);
}
