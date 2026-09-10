import numpy as np
import torch

from alphazero.observation import encode_history


def test_observation_shape_and_history_order():
    frames = []
    for history_index in range(8):
        frame = np.zeros((14, 8, 8), dtype=np.float32)
        frame[history_index % 14, 0, history_index] = 1.0
        frames.append(frame)

    encoded = encode_history(frames, castling_rights=0b0101, side_to_move=1)
    assert tuple(encoded.shape) == (119, 8, 8)
    assert encoded.dtype == torch.float32
    assert encoded[0, 0, 0].item() == 1.0
    assert encoded[15, 0, 1].item() == 1.0
    assert encoded[112, :, :].max().item() == 1.0
    assert encoded[116, :, :].max().item() == 1.0
    assert encoded[117, :, :].max().item() == 0.0
    assert encoded[118, :, :].max().item() == 0.0


def test_history_is_zero_padded():
    encoded = encode_history([], castling_rights=0, side_to_move=0)
    assert tuple(encoded.shape) == (119, 8, 8)
    assert float(encoded.sum()) == 0.0
