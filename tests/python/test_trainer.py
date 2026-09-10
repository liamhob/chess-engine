import numpy as np
import torch

from alphazero.network import AlphaZeroNet
from alphazero.trainer import load_checkpoint, save_checkpoint, train_step
from alphazero.training import Experience


def make_item() -> Experience:
    policy = np.zeros(4672, dtype=np.float32)
    policy[0] = 1.0
    return Experience(np.zeros((119, 8, 8), dtype=np.float32), policy, 1.0)


def test_train_step_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(3)
    model = AlphaZeroNet(residual_blocks=9, channels=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = train_step(model, optimizer, [make_item(), make_item()])
    assert np.isfinite(loss)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, iteration=4, seed=3)
    restored = AlphaZeroNet(residual_blocks=9, channels=8)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    metadata = load_checkpoint(path, restored, restored_optimizer)
    assert metadata == {"iteration": 4, "seed": 3}
