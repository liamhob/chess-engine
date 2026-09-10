from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np

from .policy import Move, index_to_move, move_to_index


@dataclass(frozen=True)
class Experience:
    observation: np.ndarray
    policy: np.ndarray
    outcome: float


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Experience] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def append(self, item: Experience) -> None:
        self._validate(item)
        self._items.append(item)

    def extend(self, items: Iterable[Experience]) -> None:
        for item in items:
            self.append(item)

    def sample(self, batch_size: int) -> list[Experience]:
        if not 0 < batch_size <= len(self._items):
            raise ValueError("batch_size must be within the current buffer size")
        return self._random.sample(list(self._items), batch_size)

    def snapshot(self) -> list[Experience]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _validate(item: Experience) -> None:
        if item.observation.shape != (119, 8, 8):
            raise ValueError("observation must have shape (119, 8, 8)")
        if item.policy.shape != (4672,):
            raise ValueError("policy must have shape (4672,)")
        if not np.isclose(item.policy.sum(), 1.0):
            raise ValueError("policy target must sum to one")
        if not -1.0 <= item.outcome <= 1.0:
            raise ValueError("outcome must be in [-1, 1]")


def mirror_experience(item: Experience) -> Experience:
    ReplayBuffer._validate(item)
    observation = item.observation[:, :, ::-1].copy()
    policy = np.zeros_like(item.policy)
    for index, probability in enumerate(item.policy):
        if probability == 0.0:
            continue
        move = index_to_move(index)
        mirrored_from = (move.from_square // 8) * 8 + (7 - move.from_square % 8)
        mirrored_to = (move.to_square // 8) * 8 + (7 - move.to_square % 8)
        mirrored_index = move_to_index(Move(mirrored_from, mirrored_to, move.promotion))
        policy[mirrored_index] += probability
    return Experience(observation, policy, item.outcome)


def promote_candidate(
    wins: int,
    games: int,
    minimum_win_rate: float = 0.55,
    minimum_games: int = 1,
) -> bool:
    if games <= 0 or wins < 0 or wins > games:
        raise ValueError("wins and games must describe a valid arena")
    if not 0.0 <= minimum_win_rate <= 1.0:
        raise ValueError("minimum_win_rate must be in [0, 1]")
    if minimum_games <= 0:
        raise ValueError("minimum_games must be positive")
    return games >= minimum_games and wins / games >= minimum_win_rate
