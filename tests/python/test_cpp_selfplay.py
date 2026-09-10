import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "build" / "Release"))

from alphazero.cpp_selfplay import play_cpp_game
from alphazero.network import AlphaZeroNet


def test_cpp_selfplay_produces_training_records():
    model = AlphaZeroNet(residual_blocks=9, channels=8)
    records = play_cpp_game(model, simulations=2, max_plies=4, seed=2, temperature_moves=2)
    assert len(records) == 4
    assert records[0].observation.shape == (119, 8, 8)
    assert records[0].policy.shape == (4672,)
    assert torch.isfinite(torch.from_numpy(records[0].observation)).all()
    assert abs(float(records[0].policy.sum()) - 1.0) < 1e-6


def test_cpp_selfplay_parallel_batched():
    model = AlphaZeroNet(residual_blocks=9, channels=8)
    records = play_cpp_game(
        model,
        simulations=16,
        max_plies=4,
        seed=3,
        temperature_moves=2,
        mcts_workers=4,
        inference_batch_size=8,
    )
    assert len(records) == 4
    assert records[0].observation.shape == (119, 8, 8)
    assert records[0].policy.shape == (4672,)
    assert torch.isfinite(torch.from_numpy(records[0].observation)).all()
    assert abs(float(records[0].policy.sum()) - 1.0) < 1e-6

