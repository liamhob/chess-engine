import numpy as np
import pytest

from alphazero.training import (
    Experience,
    ReplayBuffer,
    mirror_experience,
    promote_candidate,
)


def experience(index: int) -> Experience:
    observation = np.full((119, 8, 8), index, dtype=np.float32)
    policy = np.zeros(4672, dtype=np.float32)
    policy[index] = 1.0
    return Experience(observation, policy, float(index % 3 - 1))


def test_replay_buffer_is_bounded_and_seeded():
    buffer = ReplayBuffer(capacity=2, seed=7)
    buffer.extend([experience(0), experience(1), experience(2)])
    assert len(buffer) == 2
    assert [item.observation[0, 0, 0] for item in buffer.snapshot()] == [1.0, 2.0]
    sample_a = buffer.sample(2)
    buffer_again = ReplayBuffer(capacity=2, seed=7)
    buffer_again.extend([experience(0), experience(1), experience(2)])
    assert [item.observation[0, 0, 0] for item in sample_a] == [
        item.observation[0, 0, 0] for item in buffer_again.sample(2)
    ]


def test_mirror_preserves_shapes_and_probability_mass():
    item = experience(3)
    mirrored = mirror_experience(item)
    assert mirrored.observation.shape == item.observation.shape
    assert mirrored.policy.shape == item.policy.shape
    assert np.isclose(mirrored.policy.sum(), 1.0)
    assert mirrored.observation[0, 0, 7] == item.observation[0, 0, 0]


def test_promotion_requires_at_least_55_percent():
    assert promote_candidate(wins=55, games=100, minimum_win_rate=0.55)
    assert not promote_candidate(wins=54, games=100, minimum_win_rate=0.55)
    with pytest.raises(ValueError):
        promote_candidate(wins=1, games=0)
