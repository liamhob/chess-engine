from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import sys
import json
import signal
import time

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "build" / "Release"))

from alphazero.cpp_selfplay import play_cpp_game, play_cpp_match
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import load_experiences, save_experiences
from alphazero.trainer import load_checkpoint, save_checkpoint, train_step
from alphazero.training import ReplayBuffer, mirror_experience
from alphazero.visualizer import TrainingVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZeroNet from saved experience records")
    parser.add_argument("--records", type=Path, required=True, help="Experience file to load and update")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=500_000)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-play-games", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--mcts-workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=140, help="hard ply cutoff; 0 disables the cutoff")
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--capture-prior-bonus", type=float, default=1.5, help="self-play prior multiplier for captures")
    parser.add_argument("--check-prior-bonus", type=float, default=1.2, help="self-play prior multiplier for checking moves")
    parser.add_argument("--material-weight", type=float, default=0.5, help="blend ratio for material evaluation at MCTS leaves")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha parameter at root")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Dirichlet noise mixing weight at root")
    parser.add_argument("--train-steps", type=int, default=50, help="number of gradient steps per iteration")
    parser.add_argument("--no-adjudicate", action="store_true", help="disable material adjudication on draws/cutoffs")
    parser.add_argument("--profile", choices=("default", "rtx4070", "9070xt"), default="default", help="hardware optimization profile")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "directml"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="show every self-play position")
    parser.add_argument("--visualize-delay", type=float, default=0.03)
    parser.add_argument("--amp", action="store_true", help="use CUDA/ROCm mixed precision")
    parser.add_argument("--compile", action="store_true", help="compile the network with torch.compile")
    parser.add_argument("--metrics-file", type=Path, default=None, help="append JSONL metrics")
    parser.add_argument("--progress-every", type=int, default=1, help="print self-play progress every N plies")
    parser.add_argument("--arena-games", type=int, default=0, help="evaluate against a frozen pre-update model")
    parser.add_argument("--arena-simulations", type=int, default=0)
    parser.add_argument("--arena-every", type=int, default=5)
    parser.add_argument("--arena-min-games", type=int, default=4)
    parser.add_argument("--visualize-arena", action="store_true", help="show arena boards")
    args = parser.parse_args()

    if args.profile == "9070xt":
        if args.mcts_workers == 4:
            args.mcts_workers = 16
        if args.inference_batch_size == 32:
            args.inference_batch_size = 64
        if args.batch_size == 128:
            args.batch_size = 256
        print(
            "Configured optimized profile for AMD Radeon RX 9070 XT: "
            f"mcts_workers={args.mcts_workers}, "
            f"inference_batch_size={args.inference_batch_size}, "
            f"training_batch_size={args.batch_size}",
            flush=True,
        )

    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch-size must be positive")
    if args.max_plies < 0:
        parser.error("max-plies must be non-negative")
    if args.mcts_workers <= 0 or args.inference_batch_size <= 0:
        parser.error("mcts-workers and inference-batch-size must be positive")
    if args.capture_prior_bonus <= 0.0 or args.check_prior_bonus <= 0.0:
        parser.error("prior bonuses must be positive")

    torch.manual_seed(args.seed)

    def resolve_device(requested: str):
        if requested == "cuda":
            if not torch.cuda.is_available():
                parser.error("CUDA/ROCm requested, but torch.cuda.is_available() is False")
            return torch.device("cuda")
        if requested == "directml":
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                parser.error("DirectML requested, but 'torch-directml' is not installed. Run: pip install torch-directml")
        if requested == "cpu":
            return torch.device("cpu")
        # auto detection:
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            print("Detected AMD GPU via DirectML hardware acceleration.", flush=True)
            return torch_directml.device()
        except ImportError:
            pass
        return torch.device("cpu")

    device = resolve_device(args.device)
    if args.amp and getattr(device, "type", str(device)) != "cuda":
        print(f"--amp mixed precision is only supported on CUDA/ROCm; continuing without AMP on {device}.", flush=True)
        args.amp = False
    if args.compile and sys.platform == "win32":
        print("torch.compile is disabled on Windows without Triton; continuing.", flush=True)
        args.compile = False

    replay = ReplayBuffer(capacity=args.capacity, seed=args.seed)
    if args.records.exists():
        try:
            replay.extend(load_experiences(args.records))
        except Exception as error:
            corrupt_path = args.records.with_name(f"{args.records.name}.corrupt")
            if corrupt_path.exists():
                corrupt_path.unlink()
            args.records.replace(corrupt_path)
            print(
                f"Warning: could not read {args.records.name} ({error}). "
                f"Moved it to {corrupt_path.name} and continuing with recovery data.",
                flush=True,
            )
    recovery_path = args.records.with_name(f"{args.records.stem}.recovery.npz")
    if recovery_path.exists():
        recovered = load_experiences(recovery_path)
        replay.extend(recovered)
        print(f"Recovered {len(recovered)} interrupted self-play positions.", flush=True)
        recovery_path.unlink()

    model = AlphaZeroNet(residual_blocks=args.blocks, channels=args.channels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iteration = 0
    if args.resume and args.checkpoint.exists():
        metadata = load_checkpoint(args.checkpoint, model, optimizer)
        start_iteration = metadata["iteration"]
    if args.compile:
        if not hasattr(torch, "compile"):
            parser.error("this PyTorch build does not support torch.compile")
        print("Compiling model; the first iteration will be slower.", flush=True)
        model = torch.compile(model, mode="max-autotune")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    visualizer = TrainingVisualizer(args.visualize_delay) if args.visualize else None
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    if args.arena_every <= 0:
        parser.error("arena-every must be positive")
    if args.arena_min_games <= 0:
        parser.error("arena-min-games must be positive")
    if args.metrics_file is not None:
        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    max_plies = args.max_plies if args.max_plies > 0 else None
    max_plies_label = str(args.max_plies) if max_plies is not None else "no cutoff"
    total_start = time.perf_counter()
    for iteration in range(start_iteration, start_iteration + args.iterations):
        iteration_start = time.perf_counter()
        generated_positions = 0
        self_play_reasons: Counter[str] = Counter()
        self_play_move_stats: Counter[str] = Counter()
        self_play_plies: list[int] = []
        if args.self_play_games:
            records = []
            for game_index in range(args.self_play_games):
                game_start = time.perf_counter()
                game_end: dict[str, int | str] = {}

                def report_position(state, policy, selected_move, ply):
                    if visualizer is not None:
                        visualizer.show(state, policy, selected_move, ply)
                    if (ply + 1) % args.progress_every == 0:
                        elapsed = max(time.perf_counter() - game_start, 1e-6)
                        print(
                            f"iteration={iteration + 1} game={game_index + 1}/{args.self_play_games} "
                            f"ply={ply + 1}/{max_plies_label} "
                            f"game_positions_per_sec={(ply + 1) / elapsed:.2f}",
                            flush=True,
                        )

                def report_game_end(reason, outcome, ply_count, positions, move_stats):
                    game_end["reason"] = reason
                    game_end["outcome"] = outcome
                    game_end["plies"] = ply_count
                    game_end["positions"] = positions
                    game_end["captures"] = move_stats.get("captures", 0)
                    game_end["checks"] = move_stats.get("checks", 0)
                    game_end["quiet_moves"] = move_stats.get("quiet_moves", 0)

                try:
                    game_records = play_cpp_game(
                        model,
                        simulations=args.simulations,
                        max_plies=max_plies,
                        seed=args.seed + iteration * args.self_play_games + game_index,
                        temperature_moves=args.temperature_moves,
                        mcts_workers=args.mcts_workers,
                        inference_batch_size=args.inference_batch_size,
                        capture_prior_bonus=args.capture_prior_bonus,
                        check_prior_bonus=args.check_prior_bonus,
                        material_weight=args.material_weight,
                        dirichlet_alpha=args.dirichlet_alpha,
                        dirichlet_epsilon=args.dirichlet_epsilon,
                        adjudicate_material=not args.no_adjudicate,
                        on_position=report_position,
                        on_game_end=report_game_end,
                    )
                except KeyboardInterrupt:
                    previous_handler = signal.getsignal(signal.SIGINT)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    try:
                        if records:
                            recovery_records = list(records)
                            recovery_records.extend(mirror_experience(record) for record in records)
                            save_experiences(recovery_path, recovery_records, compressed=False)
                        save_checkpoint(args.checkpoint, model, optimizer, iteration, args.seed)
                        print(
                            "Interrupted cleanly; recovery records and checkpoint were saved. "
                            "Restart with --resume to continue.",
                            flush=True,
                        )
                    finally:
                        signal.signal(signal.SIGINT, previous_handler)
                    return
                records.extend(game_records)
                generated_positions += len(game_records)
                reason = str(game_end.get("reason", "unknown"))
                plies = int(game_end.get("plies", len(game_records)))
                captures = int(game_end.get("captures", 0))
                checks = int(game_end.get("checks", 0))
                quiet_moves = int(game_end.get("quiet_moves", 0))
                self_play_reasons[reason] += 1
                self_play_move_stats.update({
                    "captures": captures,
                    "checks": checks,
                    "quiet_moves": quiet_moves,
                })
                self_play_plies.append(plies)
                elapsed = max(time.perf_counter() - iteration_start, 1e-6)
                print(
                    f"iteration={iteration + 1} game={game_index + 1}/{args.self_play_games} "
                    f"ended={reason} plies={plies} "
                    f"captures={captures} checks={checks} "
                    f"positions={generated_positions} positions_per_sec={generated_positions / elapsed:.2f} "
                    f"replay_before={len(replay)}",
                    flush=True,
                )
            replay.extend(records)
            replay.extend(mirror_experience(record) for record in records)
            save_experiences(args.records, replay.snapshot())
            if recovery_path.exists():
                recovery_path.unlink()
        if len(replay) < args.batch_size:
            parser.error(f"replay contains {len(replay)} positions; need {args.batch_size}")
        total_loss = 0.0
        steps = max(1, args.train_steps)
        for _ in range(steps):
            total_loss += train_step(
                model, optimizer, replay.sample(args.batch_size), device=device, amp=args.amp
            )
        loss = total_loss / steps
        arena = None
        current_iteration = iteration + 1
        save_checkpoint(args.checkpoint, model, optimizer, current_iteration, args.seed)
        if current_iteration % 5 == 0:
            periodic_ckpt = args.checkpoint.with_name(
                f"{args.checkpoint.stem}_iter_{current_iteration}{args.checkpoint.suffix}"
            )
            save_checkpoint(periodic_ckpt, model, optimizer, current_iteration, args.seed)
            print(f"Saved 5-iteration checkpoint: {periodic_ckpt.name}", flush=True)
        boundary_due = args.arena_games > 0 and current_iteration % args.arena_every == 0
        boundary_checkpoint = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_iter_{current_iteration}{args.checkpoint.suffix}"
        )
        best_checkpoint = args.checkpoint.with_name(f"{args.checkpoint.stem}_best{args.checkpoint.suffix}")
        if boundary_due:
            save_checkpoint(boundary_checkpoint, model, optimizer, current_iteration, args.seed)
            arena_games_to_play = args.arena_games
            if not best_checkpoint.exists():
                save_checkpoint(best_checkpoint, model, optimizer, current_iteration, args.seed)
                arena_games_to_play = 0
                print(
                    f"checkpoint boundary={current_iteration}; initialized best checkpoint "
                    f"at {best_checkpoint.name}",
                    flush=True,
                )
            else:
                incumbent = AlphaZeroNet(residual_blocks=args.blocks, channels=args.channels).to(device)
                incumbent_optimizer = torch.optim.Adam(incumbent.parameters(), lr=args.lr)
                load_checkpoint(best_checkpoint, incumbent, incumbent_optimizer)
                incumbent.eval()
            arena_wins = arena_draws = arena_losses = 0
            arena_reasons: Counter[str] = Counter()
            arena_move_stats: Counter[str] = Counter()
            arena_plies: list[int] = []
            arena_simulations = args.arena_simulations or args.simulations
            try:
                for arena_index in range(arena_games_to_play):
                    arena_start = time.perf_counter()
                    arena_end: dict[str, int | str] = {}

                    def report_arena_position(state, policy, selected_move, ply):
                        if args.visualize_arena and visualizer is not None:
                            visualizer.show(state, policy, selected_move, ply)

                    def report_arena_end(reason, outcome, ply_count, positions, move_stats):
                        arena_end["reason"] = reason
                        arena_end["outcome"] = outcome
                        arena_end["plies"] = ply_count
                        arena_end["positions"] = positions
                        arena_end["captures"] = move_stats.get("captures", 0)
                        arena_end["checks"] = move_stats.get("checks", 0)
                        arena_end["quiet_moves"] = move_stats.get("quiet_moves", 0)

                    candidate_plays_white = (arena_index % 2 == 0)
                    outcome = play_cpp_match(
                        model, incumbent, arena_simulations, max_plies,
                        args.seed + current_iteration * arena_games_to_play + arena_index,
                        candidate_plays_white=candidate_plays_white,
                        temperature_moves=4,
                        mcts_workers=args.mcts_workers,
                        inference_batch_size=args.inference_batch_size,
                        material_weight=args.material_weight,
                        on_position=report_arena_position if args.visualize_arena else None,
                        on_game_end=report_arena_end,
                    )
                    if outcome > 0:
                        arena_wins += 1
                    elif outcome < 0:
                        arena_losses += 1
                    else:
                        arena_draws += 1
                    reason = str(arena_end.get("reason", "unknown"))
                    plies = int(arena_end.get("plies", 0))
                    captures = int(arena_end.get("captures", 0))
                    checks = int(arena_end.get("checks", 0))
                    quiet_moves = int(arena_end.get("quiet_moves", 0))
                    arena_reasons[reason] += 1
                    arena_move_stats.update({
                        "captures": captures,
                        "checks": checks,
                        "quiet_moves": quiet_moves,
                    })
                    arena_plies.append(plies)
                    result = "WIN" if outcome > 0 else "LOSS" if outcome < 0 else "DRAW"
                    arena_elapsed = max(time.perf_counter() - arena_start, 1e-6)
                    print(
                        f"arena game={arena_index + 1}/{args.arena_games} result={result} "
                        f"ended={reason} plies={plies} "
                        f"captures={captures} checks={checks} "
                        f"running_score={arena_wins}-{arena_draws}-{arena_losses} "
                        f"elapsed={arena_elapsed:.1f}s",
                        flush=True,
                    )
            except KeyboardInterrupt:
                print(
                    f"Arena interrupted cleanly. Latest checkpoint is {args.checkpoint.name}; "
                    f"candidate boundary checkpoint is {boundary_checkpoint.name}.",
                    flush=True,
                )
                return
            arena_rate = arena_wins / arena_games_to_play if arena_games_to_play else 0.0
            promoted = arena_games_to_play >= args.arena_min_games and arena_rate >= 0.55
            arena = {
                "wins": arena_wins,
                "draws": arena_draws,
                "losses": arena_losses,
                "games": arena_games_to_play,
                "win_rate": arena_rate,
                "promoted": promoted,
                "best_checkpoint": best_checkpoint.name,
                "termination_reasons": dict(arena_reasons),
                "move_stats": dict(arena_move_stats),
                "average_plies": sum(arena_plies) / len(arena_plies) if arena_plies else 0.0,
            }
            if promoted:
                save_checkpoint(best_checkpoint, model, optimizer, current_iteration, args.seed)
            print(
                f"arena games={arena_games_to_play} wins={arena_wins} draws={arena_draws} "
                f"losses={arena_losses} win_rate={arena_rate:.3f} promoted={promoted} "
                f"best={best_checkpoint.name}",
                flush=True,
            )
        elapsed = max(time.perf_counter() - iteration_start, 1e-6)
        metrics = {
            "iteration": iteration + 1,
            "loss": loss,
            "device": str(device),
            "amp": args.amp,
            "games": args.self_play_games,
            "generated_positions": generated_positions,
            "replay_size": len(replay),
            "elapsed_seconds": elapsed,
            "positions_per_second": generated_positions / elapsed,
            "total_elapsed_seconds": time.perf_counter() - total_start,
            "mcts_workers": args.mcts_workers,
            "inference_batch_size": args.inference_batch_size,
        }
        if args.self_play_games:
            metrics["self_play"] = {
                "termination_reasons": dict(self_play_reasons),
                "move_stats": dict(self_play_move_stats),
                "average_plies": sum(self_play_plies) / len(self_play_plies) if self_play_plies else 0.0,
            }
        if arena is not None:
            metrics["arena"] = arena
        if device.type == "cuda":
            metrics["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        print(
            f"iteration={metrics['iteration']} loss={loss:.6f} device={device} "
            f"replay={len(replay)} positions_per_sec={metrics['positions_per_second']:.2f}",
            flush=True,
        )
        if args.metrics_file is not None:
            with args.metrics_file.open("a", encoding="utf-8") as metrics_stream:
                metrics_stream.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()
