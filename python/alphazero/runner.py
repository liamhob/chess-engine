from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch

from .selfplay import collect_self_play
from .trainer import save_checkpoint, train_step
from .training import Experience, ReplayBuffer, mirror_experience


def run_training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    play_game: Callable,
    games: int,
    batch_size: int,
    seed: int,
    checkpoint_path: str | Path | None = None,
    iteration: int = 0,
    augment: bool = True,
) -> float:
    records: Sequence[Experience] = collect_self_play(play_game, games, seed)
    augmented = list(records)
    if augment:
        augmented.extend(mirror_experience(record) for record in records)
    replay.extend(augmented)
    batch = replay.sample(batch_size)
    loss = train_step(model, optimizer, batch)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, iteration, seed)
    return loss
