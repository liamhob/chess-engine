from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: str | None = None


_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_KNIGHTS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_PROMOTION_PIECES = ("n", "b", "r")


def _square(square: int) -> tuple[int, int]:
    return square % 8, square // 8


def _plane_for_move(move: Move) -> int:
    file_delta = (move.to_square % 8) - (move.from_square % 8)
    rank_delta = (move.to_square // 8) - (move.from_square // 8)
    if move.promotion in _PROMOTION_PIECES:
        direction = 0 if file_delta == 0 else (1 if file_delta > 0 else 2)
        return 64 + direction * 3 + _PROMOTION_PIECES.index(move.promotion)
    for direction_index, (file_step, rank_step) in enumerate(_DIRECTIONS):
        distance = abs(file_delta) if file_step else abs(rank_delta)
        if distance and (file_delta, rank_delta) == (file_step * distance, rank_step * distance):
            return direction_index * 7 + distance - 1
    if (file_delta, rank_delta) in _KNIGHTS:
        return 56 + _KNIGHTS.index((file_delta, rank_delta))
    if move.promotion == "q":
        raise ValueError("queen promotions use the queen-move slot")
    raise ValueError(f"move cannot be represented by the 73-plane mapping: {move}")


def move_to_index(move: Move) -> int:
    if not 0 <= move.from_square < 64 or not 0 <= move.to_square < 64:
        raise ValueError("squares must be in [0, 63]")
    return move.from_square * 73 + _plane_for_move(move)


def index_to_move(index: int) -> Move:
    if not 0 <= index < 4672:
        raise ValueError("policy index must be in [0, 4671]")
    from_square, plane = divmod(index, 73)
    file, rank = _square(from_square)
    if plane < 56:
        direction_index, distance_index = divmod(plane, 7)
        file_step, rank_step = _DIRECTIONS[direction_index]
        distance = distance_index + 1
        to_file, to_rank = file + file_step * distance, rank + rank_step * distance
        promotion = "q" if to_rank in (0, 7) else None
    elif plane < 64:
        file_step, rank_step = _KNIGHTS[plane - 56]
        to_file, to_rank = file + file_step, rank + rank_step
        promotion = None
    else:
        direction, piece_index = divmod(plane - 64, 3)
        file_step = 0 if direction == 0 else (1 if direction == 1 else -1)
        to_file, to_rank = file + file_step, rank + (1 if rank == 6 else -1)
        promotion = _PROMOTION_PIECES[piece_index]
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError("policy index decodes to an off-board move")
    return Move(from_square, to_rank * 8 + to_file, promotion)
