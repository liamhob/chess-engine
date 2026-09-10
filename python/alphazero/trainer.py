from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .network import combined_loss
from .training import Experience


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[Experience],
    device: torch.device | str = "cpu",
    amp: bool = False,
) -> float:
    if not batch:
        raise ValueError("training batch must not be empty")
    observations = torch.from_numpy(np.stack([item.observation for item in batch])).to(device)
    policies = torch.from_numpy(np.stack([item.policy for item in batch])).to(device)
    outcomes = torch.tensor([item.outcome for item in batch], dtype=torch.float32, device=device).unsqueeze(1)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
        log_policy, value = model(observations)
    loss = combined_loss(log_policy, value, policies, outcomes, model)
    if not torch.isfinite(loss):
        raise FloatingPointError("training loss is not finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    seed: int,
) -> None:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    checkpoint_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model": checkpoint_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "seed": seed,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    target_model = getattr(model, "_orig_mod", model)
    target_model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return {"iteration": int(checkpoint["iteration"]), "seed": int(checkpoint["seed"])}
