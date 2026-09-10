from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from multiprocessing import get_context
from typing import Callable, Iterable, Sequence

import numpy as np

from .policy import Move, move_to_index
from .training import Experience, promote_candidate


@dataclass(frozen=True)
class ArenaResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def promoted(self) -> bool:
        return promote_candidate(self.wins, self.games, minimum_games=100)


def mcts_policy_target(search: object) -> np.ndarray:
    """Convert bound C++ root visit counts into a normalized 4672-slot target."""
    target = np.zeros(4672, dtype=np.float32)
    distribution = search.visit_distribution()
    total = sum(visits for _, visits in distribution)
    if total <= 0:
        raise ValueError("MCTS must have at least one root visit")
    for move, visits in distribution:
        from_square = move & 0x3F
        to_square = (move >> 6) & 0x3F
        flag = (move >> 12) & 0x0F
        promotion = None
        if 8 <= flag <= 15:
            promotion = ("n", "b", "r", "q")[(flag - 8) % 4]
        target[move_to_index(Move(from_square, to_square, promotion))] += visits / total
    return target


def collect_self_play(
    play_game: Callable[[random.Random], Sequence[Experience]],
    games: int,
    seed: int,
) -> list[Experience]:
    if games < 0:
        raise ValueError("games must be non-negative")
    random_source = random.Random(seed)
    records: list[Experience] = []
    for _ in range(games):
        records.extend(play_game(random_source))
    return records


def _play_one_game(arguments: tuple[Callable[[random.Random], Sequence[Experience]], int]) -> list[Experience]:
    play_game, seed = arguments
    return list(play_game(random.Random(seed)))


def collect_self_play_parallel(
    play_game: Callable[[random.Random], Sequence[Experience]],
    games: int,
    workers: int,
    seed: int,
) -> list[Experience]:
    if games < 0 or workers <= 0:
        raise ValueError("games must be non-negative and workers must be positive")
    if games == 0:
        return []
    context = get_context("spawn")
    with context.Pool(processes=workers) as pool:
        results = pool.map(_play_one_game, [(play_game, seed + index) for index in range(games)])
    return [record for game in results for record in game]


def save_experiences(
    path: str | Path,
    records: Iterable[Experience],
    compressed: bool = True,
) -> None:
    materialized = list(records)
    if not materialized:
        raise ValueError("cannot save an empty experience set")
    payload = {
        "observations": np.stack([record.observation for record in materialized]),
        "policies": np.stack([record.policy for record in materialized]),
        "outcomes": np.asarray([record.outcome for record in materialized], dtype=np.float32),
    }
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            if compressed:
                np.savez_compressed(stream, **payload)
            else:
                np.savez(stream, **payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_experiences(path: str | Path) -> list[Experience]:
    with np.load(path) as data:
        observations = data["observations"]
        policies = data["policies"]
        outcomes = data["outcomes"]
        if len(observations) != len(policies) or len(policies) != len(outcomes):
            raise ValueError("experience arrays have mismatched lengths")
        return [
            Experience(observations[index], policies[index], float(outcomes[index]))
            for index in range(len(outcomes))
        ]


class EvaluatorArena:
    def __init__(self, play_match: Callable[[object, object, int], int]) -> None:
        self._play_match = play_match

    def play(self, candidate: object, incumbent: object, games: int, seed: int = 0) -> ArenaResult:
        if games <= 0:
            raise ValueError("arena requires at least one game")
        wins = draws = losses = 0
        for game_index in range(games):
            outcome = self._play_match(candidate, incumbent, seed + game_index)
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1
        return ArenaResult(wins, draws, losses)
