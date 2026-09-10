import numpy as np

from alphazero.selfplay import (
    EvaluatorArena,
    collect_self_play,
    collect_self_play_parallel,
    load_experiences,
    save_experiences,
)
from alphazero.training import Experience


def record(value: float) -> Experience:
    policy = np.zeros(4672, dtype=np.float32)
    policy[0] = 1.0
    return Experience(np.zeros((119, 8, 8), dtype=np.float32), policy, value)


def pick_seeded_record(random_source):
    return [record(float(random_source.randrange(10)))]


def test_self_play_is_seeded_and_persistable(tmp_path):
    first = collect_self_play(pick_seeded_record, games=3, seed=4)
    second = collect_self_play(pick_seeded_record, games=3, seed=4)
    assert [item.outcome for item in first] == [item.outcome for item in second]
    path = tmp_path / "games.npz"
    save_experiences(path, first)
    restored = load_experiences(path)
    assert len(restored) == 3
    assert restored[0].policy[0] == 1.0


def test_parallel_self_play_uses_deterministic_workers():
    records = collect_self_play_parallel(pick_seeded_record, games=2, workers=2, seed=8)
    repeated = collect_self_play_parallel(pick_seeded_record, games=2, workers=2, seed=8)
    assert [item.outcome for item in records] == [item.outcome for item in repeated]


def test_evaluator_arena_tracks_promotion():
    arena = EvaluatorArena(lambda candidate, incumbent, seed: 1 if seed < 55 else -1)
    result = arena.play("candidate", "incumbent", games=100, seed=0)
    assert result.wins == 55
    assert result.losses == 45
    assert result.promoted
