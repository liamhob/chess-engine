from __future__ import annotations

from collections import Counter, deque
import random
import threading
from typing import Any

import numpy as np
import torch

from .observation import encode_history
from .policy import Move, move_to_index
from .selfplay import mcts_policy_target
from .training import Experience


def _move_value(value: int) -> Move:
    flag = (value >> 12) & 0x0F
    promotion = None
    if 8 <= flag <= 15:
        promotion = ("n", "b", "r", "q")[(flag - 8) % 4]
    return Move(value & 0x3F, (value >> 6) & 0x3F, promotion)


def _move_flag(value: int) -> int:
    return (value >> 12) & 0x0F


def _is_capture(value: int) -> bool:
    return _move_flag(value) in {4, 5, 12, 13, 14, 15}


PIECE_WEIGHTS = np.array(
    [1.0, 3.0, 3.2, 5.0, 9.0, 0.0, -1.0, -3.0, -3.2, -5.0, -9.0, 0.0],
    dtype=np.float32,
)


def compute_material_diff(planes: np.ndarray) -> float:
    """Compute White material minus Black material from piece planes 0..11."""
    if planes.ndim == 3:
        counts = planes[:12].sum(axis=(1, 2))
        return float(np.dot(counts, PIECE_WEIGHTS))
    counts = planes[:, :12].sum(axis=(2, 3))
    return np.dot(counts, PIECE_WEIGHTS)


def compute_material_value(child: Any) -> float:
    """Return normalized material balance in [-1, 1] relative to child's side_to_move."""
    diff = compute_material_diff(np.asarray(child.planes(), dtype=np.float32))
    perspective_diff = diff if child.side_to_move() == 0 else -diff
    return float(np.tanh(perspective_diff / 6.0))


def play_cpp_game(
    model: torch.nn.Module,
    simulations: int = 64,
    max_plies: int | None = None,
    seed: int = 0,
    temperature_moves: int = 20,
    mcts_workers: int = 1,
    inference_batch_size: int = 16,
    capture_prior_bonus: float = 1.0,
    check_prior_bonus: float = 1.0,
    material_weight: float = 0.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    adjudicate_material: bool = True,
    on_position: Any = None,
    on_game_end: Any = None,
    opponent_model: torch.nn.Module | None = None,
    return_outcome: bool = False,
) -> list[Experience] | float:
    import alphazero_cpp

    random_source = random.Random(seed)
    state = alphazero_cpp.GameState()
    history: deque[np.ndarray] = deque(maxlen=8)
    records: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_stats: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    if mcts_workers <= 0:
        raise ValueError("mcts_workers must be positive")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    def finish(reason: str, outcome: float, ply_count: int):
        if on_game_end is not None:
            on_game_end(reason, outcome, ply_count, len(records), dict(move_stats))
        if return_outcome:
            return outcome
        return [
            Experience(observation, policy, float(outcome if side == 0 else -outcome))
            for observation, policy, side in records
        ]

    def active_model_for(child: Any) -> torch.nn.Module:
        if child.side_to_move() == 1 and opponent_model is not None:
            return opponent_model
        return model

    def terminal_value(child: Any) -> tuple[float, bool]:
        if child.fifty_move_draw():
            if adjudicate_material:
                mat_diff = compute_material_diff(np.asarray(child.planes(), dtype=np.float32))
                persp = mat_diff if child.side_to_move() == 1 else -mat_diff
                val = 0.0 if abs(persp) < 1.0 else float(np.tanh(persp / 4.0))
                return val, True
            return 0.0, True
        legal_values = list(child.legal_moves())
        if not legal_values:
            # If child is in check with no legal moves, child was checkmated!
            # The player who just moved won (+1.0)!
            return (1.0 if child.in_check() else 0.0), True
        return 0.0, False

    def evaluate_batch(children: list[Any]):
        evaluations: list[Any] = [None] * len(children)
        groups: dict[int, tuple[torch.nn.Module, list[int]]] = {}
        legal_by_child: list[list[int]] = []
        for child_index, child in enumerate(children):
            value, terminal = terminal_value(child)
            legal_values = [] if terminal else list(child.legal_moves())
            legal_by_child.append(legal_values)
            if terminal:
                evaluations[child_index] = ([], [], value, True)
                continue
            active_model = active_model_for(child)
            group = groups.setdefault(id(active_model), (active_model, []))
            group[1].append(child_index)

        for active_model, child_indexes in groups.values():
            active_device = next(active_model.parameters()).device
            observations = []
            for child_index in child_indexes:
                child = children[child_index]
                frame = np.asarray(child.planes(), dtype=np.float32)
                observations.append(
                    encode_history(
                        [frame], child.castling_rights(), child.side_to_move()
                    )
                )
            observation_batch = torch.stack(observations).to(active_device)
            with torch.inference_mode(), torch.autocast(
                device_type=active_device.type,
                dtype=torch.float16,
                enabled=active_device.type == "cuda",
            ):
                log_policy_batch, value_batch = active_model(observation_batch)
            for row, child_index in enumerate(child_indexes):
                child = children[child_index]
                moves = legal_by_child[child_index]
                priors = []
                for value_move in moves:
                    prior = float(
                        log_policy_batch[
                            row, move_to_index(_move_value(value_move))
                        ].exp()
                    )
                    if capture_prior_bonus != 1.0 and _is_capture(value_move):
                        prior *= capture_prior_bonus
                    if check_prior_bonus != 1.0 and child.apply(value_move).in_check():
                        prior *= check_prior_bonus
                    priors.append(prior)

                # Root exploration noise (Dirichlet)
                if (
                    dirichlet_alpha > 0.0
                    and child.hash() == state.hash()
                    and len(moves) > 1
                ):
                    dir_noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
                    priors = [
                        float((1.0 - dirichlet_epsilon) * p + dirichlet_epsilon * n)
                        for p, n in zip(priors, dir_noise)
                    ]

                total = sum(priors)
                if total <= 0.0:
                    priors = [1.0 / len(moves)] * len(moves)
                else:
                    priors = [prior / total for prior in priors]

                nn_val = float(value_batch[row].item())
                if material_weight > 0.0:
                    mat_val = compute_material_value(child)
                    leaf_val = (1.0 - material_weight) * nn_val + material_weight * mat_val
                else:
                    leaf_val = nn_val

                # leaf_val is evaluated from child.side_to_move's perspective (the opponent).
                # For the move that led to child, its value for the player who made it is -leaf_val!
                evaluations[child_index] = (moves, priors, -leaf_val, False)
        return evaluations

    def evaluate(child: Any):
        return evaluate_batch([child])[0]

    def run_search(search: Any):
        if mcts_workers == 1:
            search.run(simulations, evaluate)
            return

        queue = alphazero_cpp.InferenceQueue()
        worker_error: list[BaseException] = []

        def process_batches():
            try:
                while True:
                    batch = queue.pop_batch(inference_batch_size)
                    if not batch:
                        return
                    states = [state for _, state in batch]
                    for (request_id, _), (_, priors, value, _) in zip(
                        batch, evaluate_batch(states)
                    ):
                        queue.respond(request_id, priors, value)
            except BaseException as error:
                worker_error.append(error)
                queue.close()

        batch_thread = threading.Thread(target=process_batches, daemon=True)
        batch_thread.start()
        search_error = None
        try:
            search.run_parallel(simulations, mcts_workers, queue)
        except BaseException as error:
            search_error = error
        finally:
            queue.close()
            batch_thread.join()
        if worker_error:
            raise worker_error[0]
        if search_error is not None:
            raise search_error

    ply = 0
    while max_plies is None or ply < max_plies:
        # Check threefold repetition
        curr_hash = state.hash()
        position_counts[curr_hash] += 1
        if position_counts[curr_hash] >= 3:
            return finish("threefold_repetition", 0.0, ply)

        if state.fifty_move_draw():
            if adjudicate_material:
                mat_diff = compute_material_diff(np.asarray(state.planes(), dtype=np.float32))
                outcome = 0.0 if abs(mat_diff) < 1.0 else float(np.tanh(mat_diff / 4.0))
                return finish("fifty_move", outcome, ply)
            return finish("fifty_move", 0.0, ply)

        legal_values = list(state.legal_moves())
        if not legal_values:
            outcome = -1.0 if state.in_check() and state.side_to_move() == 0 else 1.0 if state.in_check() else 0.0
            reason = "checkmate" if state.in_check() else "stalemate"
            return finish(reason, outcome, ply)

        # Early adjudication on overwhelming material advantage (saving time on runaway games)
        if adjudicate_material and ply >= 60:
            mat_diff = compute_material_diff(np.asarray(state.planes(), dtype=np.float32))
            if abs(mat_diff) >= 12.0:
                outcome = 1.0 if mat_diff > 0 else -1.0
                return finish("material_resignation", outcome, ply)

        frame = np.asarray(state.planes(), dtype=np.float32)
        observation = encode_history(
            [frame, *list(history)], state.castling_rights(), state.side_to_move()
        ).numpy()
        search = alphazero_cpp.MCTS(state)
        run_search(search)
        policy = mcts_policy_target(search)
        distribution = list(search.visit_distribution())
        if ply < temperature_moves:
            weights = [max(0, visits) for _, visits in distribution]
            selected_value = random_source.choices(
                [value for value, _ in distribution], weights=weights
            )[0]
        else:
            selected_value = max(distribution, key=lambda item: item[1])[0]
        records.append((observation, policy, state.side_to_move()))
        move_stats["captures" if _is_capture(selected_value) else "quiet_moves"] += 1
        next_state = state.apply(selected_value)
        if next_state.in_check():
            move_stats["checks"] += 1
        if on_position is not None:
            on_position(state, policy, selected_value, ply)
        history.appendleft(frame)
        state = next_state
        ply += 1

    if adjudicate_material:
        mat_diff = compute_material_diff(np.asarray(state.planes(), dtype=np.float32))
        outcome = 0.0 if abs(mat_diff) < 1.0 else float(np.tanh(mat_diff / 4.0))
        return finish("max_plies", outcome, ply)
    return finish("max_plies", 0.0, ply)


def play_cpp_match(
    candidate: torch.nn.Module,
    incumbent: torch.nn.Module,
    simulations: int,
    max_plies: int | None,
    seed: int,
    mcts_workers: int = 1,
    inference_batch_size: int = 16,
    capture_prior_bonus: float = 1.5,
    check_prior_bonus: float = 1.2,
    material_weight: float = 0.5,
    on_position: Any = None,
    on_game_end: Any = None,
) -> int:
    """Return +1 if candidate wins, -1 if it loses, 0 for a draw."""
    raw_outcome = float(
        play_cpp_game(
            candidate,
            simulations=simulations,
            max_plies=max_plies,
            seed=seed,
            temperature_moves=0,
            mcts_workers=mcts_workers,
            inference_batch_size=inference_batch_size,
            capture_prior_bonus=capture_prior_bonus,
            check_prior_bonus=check_prior_bonus,
            material_weight=material_weight,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            adjudicate_material=True,
            opponent_model=incumbent,
            return_outcome=True,
            on_position=on_position,
            on_game_end=on_game_end,
        )
    )
    if raw_outcome > 0.25:
        return 1
    if raw_outcome < -0.25:
        return -1
    return 0
