import torch

from alphazero.network import AlphaZeroNet, combined_loss


def test_network_shapes_and_value_range():
    model = AlphaZeroNet(residual_blocks=9, channels=32)
    observations = torch.randn(2, 119, 8, 8)
    log_policy, value = model(observations)
    assert tuple(log_policy.shape) == (2, 4672)
    assert tuple(value.shape) == (2, 1)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)
    assert torch.allclose(log_policy.exp().sum(dim=1), torch.ones(2), atol=1e-5)


def test_loss_is_finite_and_has_gradients():
    model = AlphaZeroNet(residual_blocks=9, channels=16)
    observations = torch.randn(2, 119, 8, 8)
    target_policy = torch.softmax(torch.randn(2, 4672), dim=1)
    target_value = torch.tensor([[1.0], [-1.0]])
    log_policy, value = model(observations)
    loss = combined_loss(log_policy, value, target_policy, target_value, model, l2_weight=1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
