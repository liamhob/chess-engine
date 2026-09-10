import glob
import http.server
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
LOGS_DIR = WORKSPACE / "logs"
CHECKPOINTS_DIR = WORKSPACE / "checkpoints"
METRICS_FILE = LOGS_DIR / "training.jsonl"
TASKS_DIR_PATTERN = os.path.expanduser(
    r"~/.gemini/antigravity/brain/*/.system_generated/tasks/task-*.log"
)

# Insert build path for alphazero_cpp
sys.path.insert(0, str(WORKSPACE / "python"))
sys.path.insert(0, str(WORKSPACE / "build" / "Release"))

import torch
import alphazero_cpp
from alphazero.network import AlphaZeroNet
from alphazero.trainer import load_checkpoint
from alphazero.observation import encode_history
from alphazero.policy import Move, move_to_index
from alphazero.cpp_selfplay import compute_material_value

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADED_MODELS: dict[str, AlphaZeroNet] = {}


def get_model(checkpoint_name: str) -> AlphaZeroNet:
    if checkpoint_name not in LOADED_MODELS:
        ckpt_path = CHECKPOINTS_DIR / checkpoint_name
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINTS_DIR / "model.pt"
        m = AlphaZeroNet(residual_blocks=9, channels=256).to(DEVICE)
        opt = torch.optim.Adam(m.parameters())
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, m, opt)
        m.eval()
        LOADED_MODELS[checkpoint_name] = m
    return LOADED_MODELS[checkpoint_name]


def move_to_uci(m: int) -> str:
    f_sq = m & 0x3F
    t_sq = (m >> 6) & 0x3F
    flag = (m >> 12) & 0x0F
    promo = ["n", "b", "r", "q"][(flag - 8) % 4] if 8 <= flag <= 15 else ""
    return (
        chr(ord("a") + (f_sq % 8))
        + str(1 + (f_sq // 8))
        + chr(ord("a") + (t_sq % 8))
        + str(1 + (t_sq // 8))
        + promo
    )


def build_state_from_moves(move_list: list[str]):
    state = alphazero_cpp.GameState()
    counts = {state.hash(): 1}
    for uci in move_list:
        uci = uci.strip().lower()
        if not uci:
            continue
        matched = None
        for m in state.legal_moves():
            if move_to_uci(m) == uci:
                matched = m
                break
        if matched is None:
            raise ValueError(f"Illegal move in sequence: {uci}")
        state = state.apply(matched)
        h = state.hash()
        counts[h] = counts.get(h, 0) + 1
    return state, counts


def is_repetition_draw(counts: dict[int, int]) -> bool:
    return any(c >= 3 for c in counts.values())


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_latest_task_log() -> Path | None:
    files = glob.glob(TASKS_DIR_PATTERN)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def parse_active_log(log_path: Path | None):
    status = {
        "is_active": False,
        "iteration": None,
        "game": None,
        "total_games": 8,
        "ply": None,
        "positions_per_sec": None,
        "recent_games": [],
        "raw_lines": [],
    }
    if not log_path or not log_path.exists():
        return status

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().splitlines()

        mtime = os.path.getmtime(log_path)
        status["is_active"] = (time.time() - mtime) < 90
        status["raw_lines"] = lines[-12:]

        ply_pattern = re.compile(
            r"iteration=(\d+)\s+game=(\d+)/(\d+)\s+ply=(\d+)/.+?\s+game_positions_per_sec=([0-9.]+)"
        )
        end_pattern = re.compile(
            r"iteration=(\d+)\s+game=(\d+)/(\d+)\s+ended=(\w+)\s+plies=(\d+)\s+captures=(\d+)\s+checks=(\d+)\s+positions=(\d+)\s+positions_per_sec=([0-9.]+)"
        )

        for line in reversed(lines):
            m_ply = ply_pattern.search(line)
            if m_ply and status["iteration"] is None:
                status["iteration"] = int(m_ply.group(1))
                status["game"] = int(m_ply.group(2))
                status["total_games"] = int(m_ply.group(3))
                status["ply"] = int(m_ply.group(4))
                status["positions_per_sec"] = float(m_ply.group(5))

            m_end = end_pattern.search(line)
            if m_end and len(status["recent_games"]) < 6:
                status["recent_games"].append({
                    "iteration": int(m_end.group(1)),
                    "game": int(m_end.group(2)),
                    "total_games": int(m_end.group(3)),
                    "reason": m_end.group(4),
                    "plies": int(m_end.group(5)),
                    "captures": int(m_end.group(6)),
                    "checks": int(m_end.group(7)),
                    "positions_per_sec": float(m_end.group(9)),
                })
    except Exception as e:
        status["error"] = str(e)

    return status


def load_metrics():
    history = []
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            history.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
    return history


def get_available_checkpoints():
    ckpts = []
    if CHECKPOINTS_DIR.exists():
        for p in CHECKPOINTS_DIR.glob("*.pt"):
            name = p.name
            label = name
            m_iter = re.search(r"iter_(\d+)", name)
            if m_iter:
                label = f"Iteration {m_iter.group(1)}"
            elif name == "model.pt":
                label = "Latest Active Model"
            elif name == "model_best.pt":
                label = "Best Promoted Model"
            elif "legacy" in name or "inverted" in name:
                label = "Trained Iteration 35 Model"
            ckpts.append({
                "filename": name,
                "label": label,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                "mtime": p.stat().st_mtime,
            })
    ckpts.sort(key=lambda x: x["mtime"], reverse=True)
    return ckpts


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>AlphaZero Chess • Live Training & Play</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(20, 26, 40, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-green: #10b981;
      --accent-blue: #38bdf8;
      --accent-purple: #a855f7;
      --accent-orange: #f59e0b;
      --text-main: #f3f4f6;
      --text-dim: #9ca3af;
      --text-faint: #4b5563;
      --sq-light: #e2e8f0;
      --sq-dark: #64748b;
      --sq-highlight: rgba(56, 189, 248, 0.45);
      --sq-selected: rgba(245, 158, 11, 0.55);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      background: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      padding: 14px 12px 40px;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 40%);
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    
    /* Top Nav Tabs */
    .tabs-nav {
      display: flex;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 4px;
      margin-bottom: 18px;
      gap: 4px;
    }
    .tab-btn {
      flex: 1;
      padding: 8px 12px;
      border: none;
      background: transparent;
      color: var(--text-dim);
      font-size: 0.82rem;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    .tab-btn.active {
      background: var(--card-bg);
      color: #ffffff;
      border: 1px solid var(--card-border);
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 {
      font-size: 1.3rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #ffffff;
    }
    .title-group p { font-size: 0.75rem; color: var(--text-dim); margin-top: 2px; }
    .header-right {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 4px;
    }
    .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 9999px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }
    .live-badge.idle { background: rgba(107, 114, 128, 0.15); border-color: rgba(107, 114, 128, 0.3); color: #9ca3af; }
    .pulse-dot {
      width: 7px; height: 7px; border-radius: 50%; background: #10b981;
      box-shadow: 0 0 8px #10b981; animation: pulse 1.6s infinite ease-in-out;
    }
    .live-badge.idle .pulse-dot { background: #6b7280; box-shadow: none; animation: none; }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.4); opacity: 0.5; }
    }
    .heartbeat-text { font-size: 0.65rem; color: var(--text-dim); }

    /* Tab Content Containers */
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* Stats Grid */
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 16px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      backdrop-filter: blur(12px);
    }
    .card-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; }
    .card-val { font-size: 1.45rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
    .card-sub { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; }
    .progress-bar-container { background: rgba(255, 255, 255, 0.06); border-radius: 9999px; height: 5px; margin-top: 6px; overflow: hidden; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, #38bdf8, #10b981); width: 0%; transition: width 0.3s ease; }

    .chart-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px 12px;
      margin-bottom: 16px;
    }
    .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .chart-title { font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: var(--text-main); }
    .chart-legend { font-size: 0.75rem; color: #38bdf8; font-weight: 700; }
    .chart-wrapper { position: relative; height: 170px; width: 100%; }
    canvas#lossCanvas { width: 100%; height: 100%; display: block; }

    .section-title {
      font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim);
      margin: 16px 0 10px; display: flex; justify-content: space-between; align-items: center;
    }
    .games-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 18px; }
    .game-item {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px;
      padding: 9px 12px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;
    }
    .game-badge {
      padding: 3px 7px; border-radius: 6px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
      background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .game-badge.fifty_move { background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(168, 85, 247, 0.3); }

    .log-terminal {
      background: rgba(10, 14, 23, 0.95); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 12px; font-size: 0.7rem; line-height: 1.45; color: #94a3b8; max-height: 140px;
      overflow-y: auto; white-space: pre-wrap; word-break: break-all;
    }

    /* === Play Tab Interactive Chessboard === */
    .play-controls {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .controls-row {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }
    .select-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      flex: 1;
    }
    .select-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-dim);
    }
    select.custom-select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 0.8rem;
      outline: none;
    }
    .btn-action {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-action:active { background: rgba(255, 255, 255, 0.18); }
    .btn-action.primary {
      background: #0284c7;
      border-color: #38bdf8;
    }

    /* Chessboard Container */
    .board-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
    }
    .board-status-bar {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.82rem;
      padding: 0 4px;
    }
    .eval-badge {
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
    }
    .board-outer {
      width: 100%;
      max-width: 360px;
      aspect-ratio: 1 / 1;
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      user-select: none;
      -webkit-user-select: none;
    }
    .sq {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 2.2rem;
      transition: background 0.12s ease;
    }
    .sq.light { background: var(--sq-light); color: #1e293b; }
    .sq.dark { background: var(--sq-dark); color: #0f172a; }
    .sq.selected { background: var(--sq-selected) !important; }
    .sq.highlight::after {
      content: '';
      position: absolute;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--accent-blue);
      box-shadow: 0 0 8px var(--accent-blue);
    }
    .sq.in-check { background: rgba(239, 68, 68, 0.6) !important; }
    .sq.last-from { background: rgba(56, 189, 248, 0.3) !important; }
    .sq.last-to { background: rgba(56, 189, 248, 0.45) !important; }
    
    .piece {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 1;
      filter: drop-shadow(0 2px 3px rgba(0,0,0,0.4));
    }
    .piece.white { color: #f8fafc; text-shadow: 0 0 2px #000; }
    .piece.black { color: #0f172a; text-shadow: 0 0 1px #fff; }

    .moves-history {
      width: 100%;
      background: rgba(10, 14, 23, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 0.72rem;
      max-height: 65px;
      overflow-y: auto;
      color: var(--text-dim);
      line-height: 1.4;
    }

    .footer { text-align: center; font-size: 0.7rem; color: var(--text-faint); margin-top: 24px; }
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>Chess AlphaZero</h1>
      <p>RTX 4070 • Batched MCTS Engine</p>
    </div>
    <div class="header-right">
      <div id="liveBadge" class="live-badge">
        <div class="pulse-dot"></div>
        <span id="liveStatusText">LIVE</span>
      </div>
      <span id="heartbeat" class="heartbeat-text mono">Syncing...</span>
    </div>
  </div>

  <!-- Top Navigation Tabs -->
  <div class="tabs-nav">
    <button id="tabBtnTrain" class="tab-btn active" onclick="switchTab('train')">
      📊 Live Training
    </button>
    <button id="tabBtnPlay" class="tab-btn" onclick="switchTab('play')">
      ♟️ Play Against Model
    </button>
  </div>

  <!-- Tab 1: Live Training Monitoring -->
  <div id="tabPaneTrain" class="tab-pane active">
    <div class="grid">
      <div class="card">
        <div class="card-label">Iteration & Game</div>
        <div class="card-val mono" id="valIteration">--</div>
        <div class="card-sub mono" id="valGameSub">Game --/--</div>
        <div class="progress-bar-container">
          <div id="gameProgress" class="progress-bar"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-label">Speed</div>
        <div class="card-val mono" style="color:#10b981;" id="valSpeed">--</div>
        <div class="card-sub mono" id="valMctsSims">-- sim/s</div>
      </div>

      <div class="card">
        <div class="card-label">Latest Loss</div>
        <div class="card-val mono" style="color:#38bdf8;" id="valLoss">--</div>
        <div class="card-sub mono" id="valLossDelta">best checkpoint</div>
      </div>

      <div class="card">
        <div class="card-label">Replay Buffer</div>
        <div class="card-val mono" style="color:#f59e0b;" id="valReplay">--</div>
        <div class="card-sub mono" id="valCurrentPly">Current Ply: --</div>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-title">Training Loss Trajectory</div>
        <div class="chart-legend mono" id="chartLatest">Latest: --</div>
      </div>
      <div class="chart-wrapper">
        <canvas id="lossCanvas"></canvas>
      </div>
    </div>

    <div class="section-title">
      <span>Recent Games Completed</span>
      <button class="btn-action" onclick="fetchStatus()">Refresh</button>
    </div>
    <div class="games-list" id="gamesList">
      <div class="game-item" style="color:var(--text-dim);">Awaiting completed games...</div>
    </div>

    <div class="section-title">
      <span>Live Console Log</span>
    </div>
    <div class="log-terminal mono" id="consoleLog">Listening for live stream...</div>
  </div>

  <!-- Tab 2: Interactive Play Against Checkpoint -->
  <div id="tabPanePlay" class="tab-pane">
    <div class="play-controls">
      <div class="controls-row">
        <div class="select-group">
          <span class="select-label">Choose Checkpoint</span>
          <select id="checkpointSelect" class="custom-select mono"></select>
        </div>
        <div class="select-group" style="max-width:110px;">
          <span class="select-label">Simulations</span>
          <select id="simsSelect" class="custom-select mono">
            <option value="16">16 (Fast)</option>
            <option value="32" selected>32 (Normal)</option>
            <option value="64">64 (Deep)</option>
          </select>
        </div>
      </div>
      <div class="controls-row" style="margin-top:2px;">
        <button class="btn-action primary" onclick="startNewGame('w')">Play as White</button>
        <button class="btn-action" onclick="startNewGame('b')">Play as Black</button>
        <button class="btn-action" onclick="undoMove()">Undo</button>
      </div>
    </div>

    <div class="board-card">
      <div class="board-status-bar">
        <span id="turnIndicator" style="font-weight:700;">White to move</span>
        <span id="evalBadge" class="eval-badge mono">Eval: 0.00</span>
      </div>

      <div id="chessBoard" class="board-outer"></div>

      <div class="moves-history mono" id="movesHistory">Moves: (New game started)</div>
    </div>
  </div>

  <div class="footer">
    AlphaZero Chess Monitor • Live Cloudflare Tunnel
  </div>

  <script>
    let globalHistory = [];
    let currentTab = 'train';

    function switchTab(tab) {
      currentTab = tab;
      document.getElementById('tabBtnTrain').classList.toggle('active', tab === 'train');
      document.getElementById('tabBtnPlay').classList.toggle('active', tab === 'play');
      document.getElementById('tabPaneTrain').classList.toggle('active', tab === 'train');
      document.getElementById('tabPanePlay').classList.toggle('active', tab === 'play');
      if (tab === 'train') {
        drawChart(globalHistory);
      } else {
        fetchCheckpoints();
        renderBoard();
      }
    }

    // --- Tab 1: Live Status & Chart ---
    function drawChart(history) {
      const canvas = document.getElementById('lossCanvas');
      if (!canvas || !history || history.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const w = rect.width, h = rect.height;
      const padL = 36, padR = 12, padT = 12, padB = 24;
      const plotW = w - padL - padR, plotH = h - padT - padB;

      ctx.clearRect(0, 0, w, h);

      const losses = history.map(item => Number(item.loss));
      const minLoss = Math.floor(Math.min(...losses) * 0.95);
      const maxLoss = Math.ceil(Math.max(...losses) * 1.05);
      const lossRange = Math.max(0.1, maxLoss - minLoss);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.fillStyle = '#64748b';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const gridSteps = 4;
      for (let i = 0; i <= gridSteps; i++) {
        const yVal = minLoss + (lossRange * i) / gridSteps;
        const yPix = padT + plotH - (plotH * i) / gridSteps;
        ctx.beginPath();
        ctx.moveTo(padL, yPix);
        ctx.lineTo(w - padR, yPix);
        ctx.stroke();
        ctx.fillText(yVal.toFixed(1), padL - 6, yPix);
      }

      if (losses.length < 2) return;

      const points = losses.map((loss, idx) => ({
        x: padL + (idx / (losses.length - 1)) * plotW,
        y: padT + plotH - ((loss - minLoss) / lossRange) * plotH
      }));

      const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

      ctx.beginPath();
      ctx.moveTo(points[0].x, padT + plotH);
      points.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.lineTo(points[points.length - 1].x, padT + plotH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.stroke();

      const lastPt = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(lastPt.x, lastPt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status?t=' + Date.now());
        if (!res.ok) return;
        const data = await res.json();
        updateUI(data);
        const d = new Date();
        document.getElementById('heartbeat').textContent = 'Synced ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
      } catch (err) {
        document.getElementById('heartbeat').textContent = 'Reconnecting...';
      }
    }

    function updateUI(data) {
      if (!data) return;
      const live = data.live || {};
      const history = data.history || [];
      globalHistory = history;

      const badge = document.getElementById('liveBadge');
      const badgeText = document.getElementById('liveStatusText');
      if (live.is_active) {
        badge.classList.remove('idle');
        badgeText.textContent = 'LIVE';
      } else {
        badge.classList.add('idle');
        badgeText.textContent = 'IDLE / REPLAY';
      }

      const iter = live.iteration || (history.length ? history[history.length - 1].iteration + 1 : 1);
      document.getElementById('valIteration').textContent = 'Iter ' + iter;

      const gameNum = live.game || 1;
      const totalGames = live.total_games || 8;
      document.getElementById('valGameSub').textContent = 'Game ' + gameNum + '/' + totalGames;
      const pct = Math.min(100, Math.round((gameNum / totalGames) * 100));
      document.getElementById('gameProgress').style.width = pct + '%';

      const speed = live.positions_per_sec || (history.length ? history[history.length - 1].positions_per_second : 0);
      document.getElementById('valSpeed').textContent = (speed ? speed.toFixed(1) : '--') + ' pos/s';
      document.getElementById('valMctsSims').textContent = '~' + Math.round(speed * 32) + ' sim/s';

      let latestLoss = '--';
      if (history.length > 0) {
        latestLoss = Number(history[history.length - 1].loss).toFixed(3);
      }
      document.getElementById('valLoss').textContent = latestLoss;
      document.getElementById('chartLatest').textContent = 'Latest: ' + latestLoss;

      let replaySize = '--';
      if (history.length > 0) {
        replaySize = Number(history[history.length - 1].replay_size).toLocaleString();
      }
      document.getElementById('valReplay').textContent = replaySize;

      const currentPly = live.ply ? ('Current Ply: ' + live.ply) : 'Current Ply: --';
      document.getElementById('valCurrentPly').textContent = currentPly;

      if (currentTab === 'train') drawChart(history);

      const gamesList = document.getElementById('gamesList');
      if (live.recent_games && live.recent_games.length > 0) {
        gamesList.innerHTML = live.recent_games.map(g => `
          <div class="game-item">
            <div>
              <span class="mono" style="font-weight:700;">Iter ${g.iteration} • Game ${g.game}/${g.total_games}</span>
              <div style="font-size:0.72rem; color:var(--text-dim); margin-top:2px;">
                ${g.plies} plies • ${g.captures} caps • ${g.checks} chks • ${Number(g.positions_per_sec).toFixed(1)} pos/s
              </div>
            </div>
            <span class="game-badge ${g.reason}">${g.reason}</span>
          </div>
        `).join('');
      }

      if (live.raw_lines && live.raw_lines.length > 0) {
        const consoleEl = document.getElementById('consoleLog');
        consoleEl.textContent = live.raw_lines.join('\\n');
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }
    }

    // --- Tab 2: Interactive Play Against Model ---
    const UNICODE_PIECES = {
      'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
      'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    };

    let gameState = {
      playerColor: 'w',
      moves: [],
      board: [
        ['r','n','b','q','k','b','n','r'],
        ['p','p','p','p','p','p','p','p'],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['P','P','P','P','P','P','P','P'],
        ['R','N','B','Q','K','B','N','R']
      ],
      selectedSq: null,
      legalDests: [],
      lastMove: null,
      isEngineThinking: false,
      turn: 'w'
    };

    async function fetchCheckpoints() {
      try {
        const res = await fetch('/api/checkpoints');
        if (!res.ok) return;
        const ckpts = await res.json();
        const sel = document.getElementById('checkpointSelect');
        const curr = sel.value;
        sel.innerHTML = ckpts.map(c => `<option value="${c.filename}">${c.label} (${c.size_mb}MB)</option>`).join('');
        if (curr) sel.value = curr;
      } catch (e) {
        console.error(e);
      }
    }

    function startNewGame(color = 'w') {
      gameState.playerColor = color;
      gameState.moves = [];
      gameState.board = [
        ['r','n','b','q','k','b','n','r'],
        ['p','p','p','p','p','p','p','p'],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['','','','','','','',''],
        ['P','P','P','P','P','P','P','P'],
        ['R','N','B','Q','K','B','N','R']
      ];
      gameState.selectedSq = null;
      gameState.legalDests = [];
      gameState.lastMove = null;
      gameState.turn = 'w';
      gameState.isEngineThinking = false;
      document.getElementById('turnIndicator').textContent = 'White to move';
      document.getElementById('evalBadge').textContent = 'Eval: 0.00';
      document.getElementById('movesHistory').textContent = 'Moves: (New game started)';
      renderBoard();

      if (color === 'b') {
        triggerEngineMove();
      }
    }

    function renderBoard() {
      const container = document.getElementById('chessBoard');
      container.innerHTML = '';
      const isFlipped = (gameState.playerColor === 'b');

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const row = isFlipped ? 7 - r : r;
          const col = isFlipped ? 7 - c : c;
          const isDark = (row + col) % 2 === 1;
          const file = String.fromCharCode(97 + col);
          const rank = 8 - row;
          const uciSq = file + rank;

          const sqEl = document.createElement('div');
          sqEl.className = 'sq ' + (isDark ? 'dark' : 'light');
          sqEl.dataset.sq = uciSq;
          sqEl.dataset.row = row;
          sqEl.dataset.col = col;

          if (gameState.selectedSq === uciSq) sqEl.classList.add('selected');
          if (gameState.legalDests.includes(uciSq)) sqEl.classList.add('highlight');
          if (gameState.lastMove && (gameState.lastMove.from === uciSq || gameState.lastMove.to === uciSq)) {
            sqEl.classList.add(gameState.lastMove.from === uciSq ? 'last-from' : 'last-to');
          }

          const p = gameState.board[row][col];
          if (p) {
            const isWhite = p === p.toUpperCase();
            const pieceEl = document.createElement('div');
            pieceEl.className = 'piece ' + (isWhite ? 'white' : 'black');
            pieceEl.textContent = UNICODE_PIECES[p] || p;
            sqEl.appendChild(pieceEl);
          }

          sqEl.addEventListener('click', () => handleSquareClick(uciSq, row, col));
          container.appendChild(sqEl);
        }
      }
    }

    async function handleSquareClick(uciSq, row, col) {
      if (gameState.isEngineThinking) return;
      if (gameState.turn !== gameState.playerColor) return;

      const p = gameState.board[row][col];
      const isOwnPiece = p && (gameState.playerColor === 'w' ? p === p.toUpperCase() : p === p.toLowerCase());

      // If clicking already selected square, deselect
      if (gameState.selectedSq === uciSq) {
        gameState.selectedSq = null;
        gameState.legalDests = [];
        renderBoard();
        return;
      }

      // If clicked own piece, select it and request legal moves
      if (isOwnPiece) {
        gameState.selectedSq = uciSq;
        try {
          const res = await fetch('/api/legal_moves', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ moves: gameState.moves, from_square: uciSq })
          });
          if (res.ok) {
            const data = await res.json();
            gameState.legalDests = data.destinations || [];
          }
        } catch (e) {
          gameState.legalDests = [];
        }
        renderBoard();
        return;
      }

      // If a destination square was clicked for an existing selection
      if (gameState.selectedSq && gameState.legalDests.includes(uciSq)) {
        const fromSq = gameState.selectedSq;
        let uciMove = fromSq + uciSq;
        // Auto-queen promotion
        const fromRow = 8 - parseInt(fromSq[1], 10);
        const fromCol = fromSq.charCodeAt(0) - 97;
        const movingPiece = gameState.board[fromRow][fromCol];
        if (movingPiece && movingPiece.toLowerCase() === 'p' && (uciSq[1] === '8' || uciSq[1] === '1')) {
          uciMove += 'q';
        }

        executeMove(uciMove);
        gameState.selectedSq = null;
        gameState.legalDests = [];
        renderBoard();

        // Trigger engine reply
        triggerEngineMove();
      }
    }

    function executeMove(uci) {
      const fCol = uci.charCodeAt(0) - 97;
      const fRow = 8 - parseInt(uci[1], 10);
      const tCol = uci.charCodeAt(2) - 97;
      const tRow = 8 - parseInt(uci[3], 10);
      const promo = uci.length > 4 ? uci[4] : null;

      let p = gameState.board[fRow][fCol];
      if (promo) {
        p = (gameState.turn === 'w') ? promo.toUpperCase() : promo.toLowerCase();
      }
      gameState.board[tRow][tCol] = p;
      gameState.board[fRow][fCol] = '';

      // Castling king move updates rook
      if (p.toLowerCase() === 'k' && Math.abs(tCol - fCol) === 2) {
        if (tCol === 6) { // Kingside
          gameState.board[fRow][5] = gameState.board[fRow][7];
          gameState.board[fRow][7] = '';
        } else if (tCol === 2) { // Queenside
          gameState.board[fRow][3] = gameState.board[fRow][0];
          gameState.board[fRow][0] = '';
        }
      }

      gameState.moves.push(uci);
      gameState.lastMove = { from: uci.slice(0, 2), to: uci.slice(2, 4) };
      gameState.turn = (gameState.turn === 'w') ? 'b' : 'w';

      document.getElementById('turnIndicator').textContent = (gameState.turn === 'w') ? 'White to move' : 'Black to move';
      document.getElementById('movesHistory').textContent = 'Moves: ' + gameState.moves.join(' ');
    }

    async function triggerEngineMove() {
      gameState.isEngineThinking = true;
      document.getElementById('turnIndicator').textContent = '🧠 Engine is thinking...';
      const ckpt = document.getElementById('checkpointSelect').value || 'model.pt';
      const sims = parseInt(document.getElementById('simsSelect').value, 10) || 32;

      try {
        const res = await fetch('/api/engine_move', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ moves: gameState.moves, checkpoint: ckpt, simulations: sims })
        });
        if (!res.ok) throw new Error('Engine move failed');
        const data = await res.json();
        if (data.move) {
          executeMove(data.move);
          const evalScore = (data.eval !== undefined) ? data.eval.toFixed(2) : '0.00';
          document.getElementById('evalBadge').textContent = 'Eval: ' + (data.eval > 0 ? '+' : '') + evalScore;
          if (data.is_game_over) {
            document.getElementById('turnIndicator').textContent = 'Game Over: ' + (data.reason || 'Draw');
          }
        }
      } catch (err) {
        console.error('Engine error:', err);
        document.getElementById('turnIndicator').textContent = 'Error: could not generate move';
      } finally {
        gameState.isEngineThinking = false;
        if (!document.getElementById('turnIndicator').textContent.includes('Game Over')) {
          document.getElementById('turnIndicator').textContent = (gameState.turn === 'w') ? 'White to move' : 'Black to move';
        }
        renderBoard();
      }
    }

    function undoMove() {
      if (gameState.moves.length === 0 || gameState.isEngineThinking) return;
      // Undo 2 plies if user is playing
      const undoCount = (gameState.moves.length >= 2 && gameState.turn === gameState.playerColor) ? 2 : 1;
      for (let i = 0; i < undoCount; i++) {
        gameState.moves.pop();
      }
      // Rebuild board from move list
      const savedMoves = [...gameState.moves];
      startNewGame(gameState.playerColor);
      savedMoves.forEach(m => executeMove(m));
      renderBoard();
    }

    window.addEventListener('DOMContentLoaded', () => {
      fetchStatus();
      setInterval(fetchStatus, 1500);
      fetchCheckpoints();
      renderBoard();
    });
  </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            latest_log = get_latest_task_log()
            live_status = parse_active_log(latest_log)
            history = load_metrics()
            payload = {
                "live": live_status,
                "history": history,
                "local_ip": get_local_ip(),
                "timestamp": time.time(),
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/checkpoints"):
            ckpts = get_available_checkpoints()
            body = json.dumps(ckpts).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(post_data) if post_data else {}
        except Exception:
            req = {}

        if self.path.startswith("/api/legal_moves"):
            moves = req.get("moves", [])
            from_sq = req.get("from_square", "").lower()
            try:
                state, _ = build_state_from_moves(moves)
                dests = []
                for m in state.legal_moves():
                    uci = move_to_uci(m)
                    if uci.startswith(from_sq):
                        dests.append(uci[2:4])
                res = {"destinations": dests}
            except Exception as e:
                res = {"error": str(e), "destinations": []}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/engine_move"):
            moves = req.get("moves", [])
            ckpt = req.get("checkpoint", "model.pt")
            sims = int(req.get("simulations", 32))
            try:
                state, counts = build_state_from_moves(moves)
                if not state.legal_moves():
                    res = {
                        "move": None,
                        "is_game_over": True,
                        "reason": "checkmate" if state.in_check() else "stalemate",
                    }
                elif is_repetition_draw(counts):
                    res = {
                        "move": None,
                        "is_game_over": True,
                        "reason": "threefold_repetition",
                    }
                elif state.fifty_move_draw():
                    res = {
                        "move": None,
                        "is_game_over": True,
                        "reason": "fifty_move_draw",
                    }
                else:
                    model = get_model(ckpt)

                    def evaluate(child):
                        if child.fifty_move_draw():
                            return [], [], 0.0, True
                        legals = list(child.legal_moves())
                        if not legals:
                            # Child has no legal moves. If child is in check, child was checkmated!
                            # The player who just moved won (+1.0)!
                            val = 1.0 if child.in_check() else 0.0
                            return [], [], val, True
                        frame = child.planes()
                        obs = encode_history(
                            [frame], child.castling_rights(), child.side_to_move()
                        ).unsqueeze(0).to(DEVICE)
                        with torch.inference_mode():
                            log_pol, val = model(obs)
                        priors = []
                        for m in legals:
                            flag = (m >> 12) & 0x0F
                            promo = (
                                ("n", "b", "r", "q")[(flag - 8) % 4]
                                if 8 <= flag <= 15
                                else None
                            )
                            mv = Move(m & 0x3F, (m >> 6) & 0x3F, promo)
                            nn_p = float(log_pol[0, move_to_index(mv)].exp())
                            # Blend with uniform prior to guarantee exploration of all legal moves
                            p = 0.5 * nn_p + 0.5 * (1.0 / len(legals))
                            if flag in {4, 5, 12, 13, 14, 15}:
                                p *= 2.5
                            priors.append(p)
                        tot = sum(priors)
                        priors = [p / tot for p in priors] if tot > 0 else [1.0 / len(legals)] * len(legals)
                        nn_val = float(val[0].item())
                        mat_val = compute_material_value(child)
                        move_val = -(0.5 * nn_val + 0.5 * mat_val)
                        return legals, priors, move_val, False

                    search = alphazero_cpp.MCTS(state)
                    search.run(sims, evaluate)
                    dist = search.visit_distribution()
                    best_m = max(dist, key=lambda x: x[1])[0]
                    chosen_uci = move_to_uci(best_m)

                    next_state = state.apply(best_m)
                    counts[next_state.hash()] = counts.get(next_state.hash(), 0) + 1
                    is_rep = counts[next_state.hash()] >= 3
                    is_over = len(next_state.legal_moves()) == 0 or next_state.fifty_move_draw() or is_rep
                    reason = None
                    if is_over:
                        if next_state.in_check():
                            reason = "checkmate"
                        elif is_rep:
                            reason = "threefold_repetition"
                        elif next_state.fifty_move_draw():
                            reason = "fifty_move_draw"
                        else:
                            reason = "stalemate"

                    eval_num = round(compute_material_value(state) * 5.0, 1)
                    res = {
                        "move": chosen_uci,
                        "eval": f"{'+' if eval_num > 0 else ''}{eval_num}",
                        "is_check": next_state.in_check(),
                        "is_game_over": is_over,
                        "reason": reason,
                    }
            except Exception as e:
                res = {"error": str(e), "move": None}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def run_server(port: int = 8080):
    ip = get_local_ip()
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), DashboardHandler) as httpd:
        print(f"=== AlphaZero Live Training & Play Dashboard ===", flush=True)
        print(f"Local Access:   http://localhost:{port}", flush=True)
        print(f"Phone Access:   http://{ip}:{port}", flush=True)
        print(f"Server is listening on 0.0.0.0:{port}...", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


if __name__ == "__main__":
    run_server(8080)
