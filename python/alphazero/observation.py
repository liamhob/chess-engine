from __future__ import annotations

from collections.abc import Sequence

import torch


HISTORY_LENGTH = 8
PLANES_PER_POSITION = 14
AUXILIARY_PLANES = 7
OBSERVATION_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + AUXILIARY_PLANES


def encode_history(
    frames: Sequence[torch.Tensor], castling_rights: int, side_to_move: int
) -> torch.Tensor:
    """Encode newest-first 14-plane frames as (119, 8, 8) float32 channels.

    Planes 0-111 are eight newest-first historical positions. Auxiliary planes
    112-115 are white/black king-side and queen-side castling rights, plane 116
    is side-to-move, and planes 117-118 are reserved for repetition metadata.
    """
    encoded = torch.zeros((OBSERVATION_PLANES, 8, 8), dtype=torch.float32)
    for history_index, frame in enumerate(frames[:HISTORY_LENGTH]):
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tuple(tensor.shape) != (PLANES_PER_POSITION, 8, 8):
            raise ValueError("each history frame must have shape (14, 8, 8)")
        start = history_index * PLANES_PER_POSITION
        encoded[start : start + PLANES_PER_POSITION] = tensor
    encoded[112].fill_(1.0 if castling_rights & 1 else 0.0)
    encoded[113].fill_(1.0 if castling_rights & 2 else 0.0)
    encoded[114].fill_(1.0 if castling_rights & 4 else 0.0)
    encoded[115].fill_(1.0 if castling_rights & 8 else 0.0)
    encoded[116].fill_(float(side_to_move))
    return encoded
