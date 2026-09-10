import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "build" / "Release"))

import alphazero_cpp
from alphazero.selfplay import ArenaResult, mcts_policy_target


def test_game_state_and_mcts():
    state = alphazero_cpp.GameState()
    assert state.hash() != 0
    assert len(state.legal_moves()) == 20
    knight_move = next(
        move for move in state.legal_moves()
        if move & 0x3F == 6 and (move >> 6) & 0x3F == 21
    )
    next_state = state.apply(knight_move)
    assert next_state.halfmove_clock() == 1
    assert not next_state.fifty_move_draw()
    search = alphazero_cpp.MCTS(state)
    search.run(2, lambda current: ([current.legal_moves()[0]], [1.0], 0.0))
    assert search.visit_count() == 2
    target = mcts_policy_target(search)
    assert target.shape == (4672,)
    assert target.sum() == 1.0


def test_batched_inference_queue():
    queue = alphazero_cpp.InferenceQueue()
    request_id = queue.submit(alphazero_cpp.GameState())
    batch = queue.pop_batch(1)
    assert len(batch) == 1
    assert batch[0][0] == request_id
    queue.respond(request_id, [0.25, 0.75], 0.5)
    policy, value = queue.wait_result(request_id)
    assert policy == [0.25, 0.75]
    assert value == 0.5
    queue.close()


def test_arena_requires_declared_minimum_games():
    assert not ArenaResult(1, 0, 0).promoted
