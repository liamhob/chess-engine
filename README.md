# AlphaZero Chess Engine

C++ chess environment and tree search with Python/PyTorch training through Pybind11.

## Current status

Phase 1 is complete. The C++ environment includes bitboards, legal move generation, deterministic magic-bitboard sliding attacks, Zobrist hashing, repetition history, and PERFT validation.

## Build and test

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Phase 5 is complete with a bounded 500,000-position replay buffer, deterministic sampling, geometry-correct policy augmentation, real C++ MCTS visit-to-policy targets, 55% evaluator promotion after a declared 100-game minimum, optimizer steps, resumable checkpoints, spawn-based self-play workers, persistent experience files, evaluator arena accounting, and an integrated training iteration runner. Phase 4 is complete with the PyTorch residual network, 119-plane observation encoder, reversible 8x8x73 policy mapping, combined loss, Pybind11 `GameState` and `MCTS` bindings, and a thread-safe batched inference queue. Phase 2 remains complete with tested PUCT nodes, sign-aware backpropagation, root noise, virtual-loss path reservation, parallel leaf evaluation, a thread pool, a leaf queue, and deterministic search orchestration.

The Windows bridge is generated as `build/Release/alphazero_cpp.cp311-win_amd64.pyd`.

For the RTX 4070 environment, install the CUDA build explicitly:

```powershell
python -m pip install -r requirements-cuda.txt
```

## Training from records

Once self-play has produced an experience file, run:

```powershell
python python/train.py --records data/experiences.npz --checkpoint checkpoints/model.pt --batch-size 128 --iterations 100
```

The records-only mode trains from an existing `.npz` file. The real self-play mode below connects C++ `GameState` history to the Python 119-plane encoder and calls bound MCTS for each move; do not use random records as a training substitute.

The real self-play loop is now available:

```powershell
python python/train.py `
	--records data/experiences.npz `
	--checkpoint checkpoints/model.pt `
	--self-play-games 32 `
	--simulations 128 `
	--max-plies 512 `
	--batch-size 128 `
	--iterations 10000 `
	--blocks 9 `
	--channels 256 `
	--seed 7
```

The CLI automatically selects CUDA when available and places the model and batches on the GPU. Start with `--simulations 32` to smoke-test the loop, then increase it after confirming checkpoints are being written.

To watch every self-play position in a live board window, add:

```powershell
--visualize --visualize-delay 0.03
```

The viewer is intentionally attached to self-play positions, so it shows the board, side to move, selected action, and current training game while the model trains on CUDA.

Training prints per-game and per-iteration throughput, replay size, loss, device, and CUDA peak memory. Add `--metrics-file logs/training.jsonl` to append the same metrics as JSONL for later plotting. Add `--amp` to enable CUDA mixed precision on the RTX 4070.

Arena checkpoints are maintained separately: `model.pt` is the latest candidate, `model_iter_N.pt` stores each boundary model, and `model_best.pt` is the last promoted baseline. At the first boundary, the best baseline is initialized; later boundaries compare against that saved best model. `--arena-min-games` controls the minimum games required for promotion and defaults to 100.
