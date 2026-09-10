from alphazero.policy import Move, index_to_move, move_to_index


def test_move_slots_round_trip():
    moves = [
        Move(0, 8),
        Move(0, 56, "q"),
        Move(27, 44),
        Move(18, 35),
        Move(48, 56, "n"),
        Move(48, 57, "r"),
        Move(55, 63, "b"),
    ]
    for move in moves:
        assert index_to_move(move_to_index(move)) == move


def test_policy_space_has_4672_slots():
    assert 64 * 73 == 4672
    assert 0 <= move_to_index(Move(63, 7)) < 4672
