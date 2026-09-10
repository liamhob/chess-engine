import glob
import http.server
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import time

WORKSPACE = Path(__file__).resolve().parents[1]
LOGS_DIR = WORKSPACE / "logs"
METRICS_FILE = LOGS_DIR / "training.jsonl"
TASKS_DIR_PATTERN = os.path.expanduser(
    r"~/.gemini/antigravity/brain/*/.system_generated/tasks/task-*.log"
)


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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>AlphaZero Chess • Live Training</title>
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
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      background: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      padding: 16px 14px 40px;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 40%);
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #ffffff;
    }
    .title-group p { font-size: 0.78rem; color: var(--text-dim); margin-top: 2px; }
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
      padding: 5px 10px;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }
    .live-badge.idle {
      background: rgba(107, 114, 128, 0.15);
      border-color: rgba(107, 114, 128, 0.3);
      color: #9ca3af;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 1.6s infinite ease-in-out;
    }
    .live-badge.idle .pulse-dot { background: #6b7280; box-shadow: none; animation: none; }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.4); opacity: 0.5; }
    }
    .heartbeat-text {
      font-size: 0.68rem;
      color: var(--text-dim);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px;
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
    }
    .card-label {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      margin-bottom: 6px;
    }
    .card-val { font-size: 1.55rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
    .card-sub { font-size: 0.75rem; color: var(--text-dim); margin-top: 5px; }
    .progress-bar-container {
      background: rgba(255, 255, 255, 0.06);
      border-radius: 9999px;
      height: 6px;
      margin-top: 8px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
      width: 0%;
      transition: width 0.3s ease;
    }
    .chart-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px 14px;
      margin-bottom: 18px;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .chart-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-main);
    }
    .chart-legend { font-size: 0.75rem; color: #38bdf8; font-weight: 700; }
    .chart-wrapper {
      position: relative;
      height: 180px;
      width: 100%;
    }
    canvas#lossCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .section-title {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      margin: 18px 0 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .games-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; }
    .game-item {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.82rem;
    }
    .game-badge {
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .game-badge.fifty_move {
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border-color: rgba(168, 85, 247, 0.3);
    }
    .game-badge.checkmate {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.4);
    }
    .log-terminal {
      background: rgba(10, 14, 23, 0.95);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      font-size: 0.72rem;
      line-height: 1.45;
      color: #94a3b8;
      max-height: 150px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .footer { text-align: center; font-size: 0.72rem; color: var(--text-faint); margin-top: 24px; }
    .refresh-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      cursor: pointer;
    }
    .refresh-btn:active { background: rgba(255, 255, 255, 0.2); }
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>Chess AlphaZero</h1>
      <p>RTX 4070 • Batched MCTS</p>
    </div>
    <div class="header-right">
      <div id="liveBadge" class="live-badge">
        <div class="pulse-dot"></div>
        <span id="liveStatusText">LIVE</span>
      </div>
      <span id="heartbeat" class="heartbeat-text mono">Syncing...</span>
    </div>
  </div>

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
    <button class="refresh-btn" onclick="fetchStatus()">Refresh</button>
  </div>
  <div class="games-list" id="gamesList">
    <div class="game-item" style="color:var(--text-dim);">Awaiting completed games...</div>
  </div>

  <div class="section-title">
    <span>Live Console Log</span>
  </div>
  <div class="log-terminal mono" id="consoleLog">Listening for live stream...</div>

  <div class="footer">
    AlphaZero Chess Monitor • Live Cloudflare Tunnel
  </div>

  <script>
    let globalHistory = [];

    // Native standalone Canvas line chart (Zero external CDN dependencies, 100% bulletproof)
    function drawChart(history) {
      const canvas = document.getElementById('lossCanvas');
      if (!canvas || !history || history.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const padL = 36;
      const padR = 12;
      const padT = 12;
      const padB = 24;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      ctx.clearRect(0, 0, w, h);

      const losses = history.map(item => Number(item.loss));
      const minLoss = Math.floor(Math.min(...losses) * 0.95);
      const maxLoss = Math.ceil(Math.max(...losses) * 1.05);
      const lossRange = Math.max(0.1, maxLoss - minLoss);

      // Grid lines
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

      // Plot line
      const points = losses.map((loss, idx) => {
        const x = padL + (idx / (losses.length - 1)) * plotW;
        const y = padT + plotH - ((loss - minLoss) / lossRange) * plotH;
        return { x, y };
      });

      // Fill gradient
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

      // Line stroke
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Highlight last point
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
        if (!res.ok) {
          document.getElementById('heartbeat').textContent = 'HTTP ' + res.status;
          return;
        }
        const data = await res.json();
        updateUI(data);
        const d = new Date();
        const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        document.getElementById('heartbeat').textContent = 'Synced ' + timeStr;
      } catch (err) {
        document.getElementById('heartbeat').textContent = 'Reconnecting...';
      }
    }

    function updateUI(data) {
      if (!data) return;
      const live = data.live || {};
      const history = data.history || [];
      globalHistory = history;

      // Badge
      const badge = document.getElementById('liveBadge');
      const badgeText = document.getElementById('liveStatusText');
      if (live.is_active) {
        badge.classList.remove('idle');
        badgeText.textContent = 'LIVE';
      } else {
        badge.classList.add('idle');
        badgeText.textContent = 'IDLE / REPLAY';
      }

      // Iteration & Game
      const iter = live.iteration || (history.length ? history[history.length - 1].iteration + 1 : 1);
      document.getElementById('valIteration').textContent = 'Iter ' + iter;

      const gameNum = live.game || 1;
      const totalGames = live.total_games || 8;
      document.getElementById('valGameSub').textContent = 'Game ' + gameNum + '/' + totalGames;
      const pct = Math.min(100, Math.round((gameNum / totalGames) * 100));
      document.getElementById('gameProgress').style.width = pct + '%';

      // Speed
      const speed = live.positions_per_sec || (history.length ? history[history.length - 1].positions_per_second : 0);
      document.getElementById('valSpeed').textContent = (speed ? speed.toFixed(1) : '--') + ' pos/s';
      document.getElementById('valMctsSims').textContent = '~' + Math.round(speed * 32) + ' sim/s';

      // Loss
      let latestLoss = '--';
      if (history.length > 0) {
        const last = history[history.length - 1];
        latestLoss = Number(last.loss).toFixed(3);
      }
      document.getElementById('valLoss').textContent = latestLoss;
      document.getElementById('chartLatest').textContent = 'Latest: ' + latestLoss;

      // Replay Buffer
      let replaySize = '--';
      if (history.length > 0) {
        replaySize = Number(history[history.length - 1].replay_size).toLocaleString();
      }
      document.getElementById('valReplay').textContent = replaySize;

      // Current Ply
      const currentPly = live.ply ? ('Current Ply: ' + live.ply) : 'Current Ply: --';
      document.getElementById('valCurrentPly').textContent = currentPly;

      // Chart
      try {
        drawChart(history);
      } catch (e) {
        console.error('Chart draw error:', e);
      }

      // Recent Games List
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

      // Live console logs
      if (live.raw_lines && live.raw_lines.length > 0) {
        const consoleEl = document.getElementById('consoleLog');
        consoleEl.textContent = live.raw_lines.join('\\n');
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }
    }

    window.addEventListener('resize', () => drawChart(globalHistory));
    fetchStatus();
    setInterval(fetchStatus, 1500);
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
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def log_message(self, format, *args):
        return


def run_server(port: int = 8080):
    ip = get_local_ip()
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), DashboardHandler) as httpd:
        print(f"=== AlphaZero Live Training Dashboard ===", flush=True)
        print(f"Local Access:   http://localhost:{port}", flush=True)
        print(f"Phone Access:   http://{ip}:{port}", flush=True)
        print(f"Server is listening on 0.0.0.0:{port}...", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


if __name__ == "__main__":
    run_server(8080)
