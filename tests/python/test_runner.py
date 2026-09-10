import numpy as np
import torch

from alphazero.network import AlphaZeroNet
from alphazero.runner import run_training_iteration
from alphazero.training import Experience, ReplayBuffer


def play_game(random_source):
    policy = np.zeros(4672, dtype=np.float32)
    policy[random_source.randrange(2)] = 1.0
    return [Experience(np.zeros((119, 8, 8), dtype=np.float32), policy, 0.0)]


def test_training_iteration_collects_augments_trains_and_checkpoints(tmp_path):
    model = AlphaZeroNet(residual_blocks=9, channels=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    replay = ReplayBuffer(capacity=16, seed=2)
    loss = run_training_iteration(
        model, optimizer, replay, play_game, games=2, batch_size=2, seed=4,
        checkpoint_path=tmp_path / "iteration.pt", iteration=1,
    )
    assert np.isfinite(loss)
    assert len(replay) == 4
    assert (tmp_path / "iteration.pt").exists()