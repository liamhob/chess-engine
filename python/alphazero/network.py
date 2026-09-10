from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class AlphaZeroNet(nn.Module):
    def __init__(self, residual_blocks: int = 9, channels: int = 256) -> None:
        super().__init__()
        if not 9 <= residual_blocks <= 19:
            raise ValueError("residual_blocks must be between 9 and 19")
        self.stem = nn.Sequential(
            nn.Conv2d(119, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*(ResidualBlock(channels) for _ in range(residual_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 73, 1),
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self.stem(observations))
        return self.policy_head(features), self.value_head(features)


def combined_loss(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    model: nn.Module,
    l2_weight: float = 1e-4,
) -> torch.Tensor:
    value_loss = torch.nn.functional.mse_loss(value, target_value)
    policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
    l2_loss = sum(parameter.square().sum() for parameter in model.parameters())
    return value_loss + policy_loss + l2_weight * l2_loss
