#!/usr/bin/env python3
"""
Live Fibonacci simulation TUI.

Controls
--------
P   pause / resume
F   speed up   (×2)
S   slow down  (÷2)
R   reset
Q   quit
"""

import math
import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

# ── helpers ──────────────────────────────────────────────────────────────────

BRAILLE  = " ▁▂▃▄▅▆▇█"
HISTORY  = 48


def _log_norm(values: list[int]) -> list[float]:
    logged = [math.log1p(v) for v in values]
    mx = max(logged) if logged else 1.0
    return [v / mx for v in logged]


def make_sparkline(values: list[int], width: int = 46) -> str:
    if not values:
        return "─" * width
    vals = list(values[-width:])
    while len(vals) < width:
        vals.insert(0, 0)
    normed = _log_norm(vals)
    return "".join(BRAILLE[min(8, int(n * 8))] for n in normed)


def make_chart(values: list[int], width: int = 46, height: int = 11) -> str:
    border_top    = "┌" + "─" * width + "┐"
    border_bottom = "└" + "─" * width + "┘"
    if not values:
        empty_row = "│" + " " * width + "│"
        return "\n".join([border_top] + [empty_row] * height + [border_bottom])
    vals = list(values[-width:])
    while len(vals) < width:
        vals.insert(0, 0)
    normed = _log_norm(vals)
    rows = [border_top]
    for row in range(height - 1, -1, -1):
        thresh = row / height
        rows.append("│" + "".join("█" if n > thresh else " " for n in normed) + "│")
    rows.append(border_bottom)
    return "\n".join(rows)


# ── app ──────────────────────────────────────────────────────────────────────

class FibTUI(App):

    TITLE     = "Fibonacci Live"
    SUB_TITLE = "TUI Simulation Demo"

    CSS = """
    Screen { background: #1a1b26; }
    Header { background: #7aa2f7; color: #1a1b26; text-style: bold; }
    Footer { background: #1e2030; }

    #left  { width: 34; padding: 0 1; }
    #right { width: 1fr; padding: 0 1; }

    .card {
        border:        round #414868;
        padding:       1 2;
        margin-bottom: 1;
        background:    #1e2030;
        height:        auto;
    }

    #chart-box {
        border:     round #9ece6a;
        padding:    1 2;
        height:     1fr;
        background: #1e2030;
    }
    """

    BINDINGS = [
        ("q", "quit",         "Quit"),
        ("p", "toggle_pause", "Pause"),
        ("f", "speed_up",     "Faster"),
        ("s", "slow_down",    "Slower"),
        ("r", "reset",        "Reset"),
    ]

    # ── state ────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self._a:     int   = 0
        self._b:     int   = 1
        self._step:  int   = 0
        self._fib:   int   = 0
        self._dt:    float = 0.0
        self._speed: float = 0.5          # seconds per step
        self._paused: bool = False
        self._history: list[int] = []
        self._last_update: float = 0.0

    # ── layout ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Static("", id="card-step",  classes="card")
                yield Static("", id="card-fib",   classes="card")
                yield Static("", id="card-dt",    classes="card")
                yield Static("", id="card-ctrl",  classes="card")
            with Vertical(id="right"):
                yield Static("", id="chart-box")
        yield Footer()

    def on_mount(self) -> None:
        self._last_update = time.perf_counter()
        self.set_interval(1 / 30, self._frame)   # 30 fps render loop
        self._render()

    # ── simulation loop ──────────────────────────────────────────────────────

    def _frame(self) -> None:
        now = time.perf_counter()
        if not self._paused and (now - self._last_update) >= self._speed:
            self._dt          = (now - self._last_update) * 1000
            self._last_update = now
            self._a, self._b  = self._b, self._a + self._b
            self._step       += 1
            self._fib         = self._a
            self._history.append(self._fib)
            if len(self._history) > HISTORY:
                self._history.pop(0)
        self._render()

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self) -> None:
        fib_str = str(self._fib)
        digits  = len(fib_str)
        if digits > 20:
            fib_str = fib_str[:17] + "..."

        status = "[bold red]⏸  PAUSED[/]" if self._paused else "[bold green]▶  RUNNING[/]"
        speed_label = f"{self._speed * 1000:.0f} ms / step"

        self.query_one("#card-step").update(
            f"[bold #7aa2f7]STEP[/]\n\n"
            f"[bold white]n = {self._step}[/]"
        )
        self.query_one("#card-fib").update(
            f"[bold #7aa2f7]F(n)[/]\n\n"
            f"[bold #9ece6a]{fib_str}[/]\n"
            f"[dim]{digits} digit{'s' if digits != 1 else ''}[/]"
        )
        self.query_one("#card-dt").update(
            f"[bold #7aa2f7]Δt  (step time)[/]\n\n"
            f"[bold #e0af68]{self._dt:.1f} ms[/]\n"
            f"[dim]target  {speed_label}[/]"
        )
        self.query_one("#card-ctrl").update(
            f"[bold #7aa2f7]STATUS[/]\n\n"
            f"{status}\n\n"
            f"[dim]P pause    R reset\n"
            f"F faster   S slower[/]"
        )

        spark = make_sparkline(self._history, width=46)
        chart = make_chart(self._history, width=46, height=11)
        self.query_one("#chart-box").update(
            f"[bold #9ece6a]Fibonacci Growth[/]  "
            f"[dim](log scale · last {HISTORY} steps)[/]\n\n"
            f"[dim]trend [/][#e0af68]{spark}[/]\n\n"
            f"[#7aa2f7]{chart}[/]"
        )

    # ── actions ──────────────────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if not self._paused:
            self._last_update = time.perf_counter()

    def action_speed_up(self) -> None:
        self._speed = max(0.05, self._speed * 0.5)

    def action_slow_down(self) -> None:
        self._speed = min(3.0, self._speed * 2.0)

    def action_reset(self) -> None:
        self._a, self._b = 0, 1
        self._step       = 0
        self._fib        = 0
        self._dt         = 0.0
        self._history.clear()
        self._last_update = time.perf_counter()


if __name__ == "__main__":
    FibTUI().run()
