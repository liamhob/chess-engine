import numpy as np

from alphazero.selfplay import Experience, load_experiences, save_experiences


def test_experience_save_is_readable(tmp_path):
    policy = np.zeros(4672, dtype=np.float32)
    policy[0] = 1.0
    item = Experience(np.zeros((119, 8, 8), dtype=np.float32), policy, 0.0)
    path = tmp_path / "experiences.npz"
    save_experiences(path, [item])
    assert len(load_experiences(path)) == 1
    assert not (tmp_path / "experiences.npz.tmp").exists()
