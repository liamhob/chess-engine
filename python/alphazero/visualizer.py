from __future__ import annotations

import time
from typing import Any


class TrainingVisualizer:
    """Small Tk board viewer updated from the self-play loop."""

    _symbols = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")

    def __init__(self, delay: float = 0.03) -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("AlphaZero self-play")
        self._canvas = tk.Canvas(self._root, width=640, height=680)
        self._canvas.pack()
        self._label = tk.Label(self._root, text="Starting self-play")
        self._label.pack()
        self._delay = max(0.0, delay)
        self._closed = False
        self._root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self) -> None:
        self._closed = True
        self._root.destroy()

    def show(self, state: Any, policy: Any, selected_move: int, ply: int) -> None:
        if self._closed:
            return
        import numpy as np

        planes = np.asarray(state.planes())
        self._canvas.delete("all")
        square = 80
        for rank in range(8):
            for file in range(8):
                x0, y0 = file * square, (7 - rank) * square
                color = "#f0d9b5" if (file + rank) % 2 == 0 else "#b58863"
                self._canvas.create_rectangle(x0, y0, x0 + square, y0 + square, fill=color, outline=color)
                occupied = np.flatnonzero(planes[:12, rank, file])
                if len(occupied):
                    self._canvas.create_text(
                        x0 + square / 2, y0 + square / 2,
                        text=self._symbols[int(occupied[0])], font=("Segoe UI Symbol", 36),
                        fill="#111111" if int(occupied[0]) < 6 else "#ffffff",
                    )
        self._label.configure(
            text=f"ply={ply} side={'black' if state.side_to_move() else 'white'} "
                 f"visits={float(policy.sum()):.2f} selected=0x{selected_move:04x}"
        )
        self._root.update_idletasks()
        self._root.update()
        if self._delay:
            time.sleep(self._delay)
