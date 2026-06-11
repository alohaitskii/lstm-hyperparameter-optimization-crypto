"""Console formatter for trading signals using rich panels and tables."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.helpers import fmt_pct, fmt_usd
from utils.ticker_utils import get_display_name

console = Console()


def _signal_color_emoji(signal: str) -> tuple[str, str]:
    if signal == "LONG":
        return "green", "🟢"
    if signal == "SHORT":
        return "red", "🔴"
    return "yellow", "⚪"


def _pct(a: float, b: float) -> str:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return "—"
    return fmt_pct((a - b) / b)


def format_signal_box(signal: Mapping[str, Any], pair_name: str = "BTC-USD (Yahoo Finance)") -> Panel:
    """Render the signal as a bordered panel."""
    direction = signal.get("final_signal", "HOLD")
    color, emoji = _signal_color_emoji(direction)

    entry = signal.get("entry_price", float("nan"))
    entry_low = signal.get("entry_low", float("nan"))
    entry_high = signal.get("entry_high", float("nan"))
    tp1 = signal.get("tp1", float("nan"))
    tp2 = signal.get("tp2", float("nan"))
    sl = signal.get("sl", float("nan"))
    conf = signal.get("confidence", 0.0)
    rr = signal.get("risk_reward", float("nan"))
    trend_1h = signal.get("trend_1h", "UNKNOWN")
    timestamp = signal.get("timestamp", "")

    body = Text()
    body.append(f"Pair        : {pair_name}\n")
    body.append(f"Timeframe   : 15m  │  1h Trend: {trend_1h}\n")
    body.append("Signal      : ")
    body.append(f"{emoji} {direction}\n", style=f"bold {color}")
    body.append(f"Confidence  : {conf * 100:.1f}%\n")

    if direction in ("LONG", "SHORT") and np.isfinite(entry):
        body.append(
            f"Entry Zone  : {fmt_usd(entry_low)} – {fmt_usd(entry_high)}\n"
        )
        body.append(f"TP1         : {fmt_usd(tp1)}  ({_pct(tp1, entry)})\n")
        body.append(f"TP2         : {fmt_usd(tp2)}  ({_pct(tp2, entry)})\n")
        body.append(f"Stop Loss   : {fmt_usd(sl)}  ({_pct(sl, entry)})\n")
        body.append(
            f"Risk/Reward : 1:{rr:.2f}\n" if np.isfinite(rr) else "Risk/Reward : —\n"
        )
    else:
        body.append("Entry Zone  : — (HOLD: tunggu sinyal yang valid)\n")

    body.append("\nSIGNAL BASIS:\n", style="bold")
    confirmations = signal.get("confirmations", []) or []
    if not confirmations:
        body.append("  (tidak ada konfirmasi teknikal — sinyal HOLD)\n")
    else:
        for r in confirmations:
            body.append(f"  ✅ {r}\n")

    body.append(f"\nGenerated   : {timestamp}\n")
    body.append(
        "⚠️  BUKAN REKOMENDASI FINANSIAL. GUNAKAN DENGAN BIJAK.\n",
        style="bold yellow",
    )

    return Panel(
        body,
        title="[bold cyan]CRYPTO LSTM SIGNAL GENERATOR[/bold cyan]",
        border_style=color,
        padding=(1, 2),
    )


def print_signal(signal: Mapping[str, Any], pair_name: str = "BTC-USD (Yahoo Finance)") -> None:
    console.print(format_signal_box(signal, pair_name))


# --------------------------------------------------------------------------- #
# Multi-pair scan table
# --------------------------------------------------------------------------- #
# Keyword → short tag mapping for the Basis column (keeps generator unchanged)
_BASIS_TAGS = [
    ("RSI", "RSI"),
    ("MACD", "MACD"),
    ("Bollinger", "BB"),
    ("Stochastic", "Stoch"),
    ("Fear & Greed", "FGI"),
]


def _basis_tags(confirmations: Sequence[str] | None) -> str:
    """Compress full confirmation sentences into 'RSI + MACD + FGI' style tags."""
    confirmations = confirmations or []
    tags = [short for key, short in _BASIS_TAGS if any(key in c for c in confirmations)]
    return " + ".join(tags) if tags else "—"


def _fmt_price(p: float | None) -> str:
    """Adaptive decimals so micro-cap prices (SHIB ~0.000025) stay readable."""
    if p is None or not np.isfinite(p):
        return "—"
    if p >= 1:
        return f"${p:,.2f}"
    if p >= 0.01:
        return f"${p:.4f}"
    return f"${p:.8f}".rstrip("0")


def _pct1(a: float | None, b: float | None) -> str:
    """Signed percent of a relative to b, 1 decimal place."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return "—"
    return fmt_pct((a - b) / b, 1)


def format_scan_table(
    results: Sequence[Mapping[str, Any]],
    generated_at: str = "",
    show_all: bool = False,
) -> Table:
    """Ranked multi-pair signal table.

    `results` must already be sorted (LONG desc-confidence, SHORT, HOLD).
    HOLD rows are excluded here when show_all=False — print_scan_table prints
    them collapsed below the table instead.
    """
    n_long = sum(1 for r in results if r.get("final_signal") == "LONG")
    n_short = sum(1 for r in results if r.get("final_signal") == "SHORT")
    n_hold = sum(1 for r in results if r.get("final_signal") == "HOLD")

    title = (
        f"[bold cyan]CRYPTO SCAN — {generated_at}[/bold cyan]\n"
        f"[white]{len(results)} pair discan │ "
        f"[green]{n_long} LONG[/green] │ [red]{n_short} SHORT[/red] │ "
        f"[yellow]{n_hold} HOLD[/yellow][/white]"
    )
    table = Table(
        title=title,
        caption="⚠️  BUKAN REKOMENDASI FINANSIAL. GUNAKAN DENGAN BIJAK.",
        caption_style="bold yellow",
        header_style="bold",
        show_lines=False,
    )
    table.add_column("#", justify="right")
    table.add_column("Ticker")
    table.add_column("Signal")
    table.add_column("Confidence", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("TP1", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("R/R", justify="right")
    table.add_column("Basis")

    rank = 0
    for r in results:
        sig = r.get("final_signal", "HOLD")
        if sig == "HOLD" and not show_all:
            continue
        rank += 1
        color, emoji = _signal_color_emoji(sig)
        conf = r.get("confidence", float("nan"))
        conf_s = f"{conf * 100:.1f}%" if np.isfinite(conf) else "—"

        if sig in ("LONG", "SHORT"):
            entry = r.get("entry_price", float("nan"))
            rr = r.get("risk_reward", float("nan"))
            entry_s = _fmt_price(entry)
            tp1_s = _pct1(r.get("tp1"), entry)
            sl_s = _pct1(r.get("sl"), entry)
            rr_s = f"1:{rr:.1f}" if np.isfinite(rr) else "—"
            basis_s = _basis_tags(r.get("confirmations"))
        else:
            entry_s = tp1_s = sl_s = rr_s = basis_s = "—"

        table.add_row(
            str(rank),
            r.get("ticker", "?"),
            f"[{color}]{emoji} {sig}[/{color}]",
            conf_s,
            entry_s,
            tp1_s,
            sl_s,
            rr_s,
            basis_s,
        )

    if rank == 0:
        table.add_row("—", "—", "⚪ (semua HOLD)", "—", "—", "—", "—", "—", "—")
    return table


def print_scan_table(
    results: Sequence[Mapping[str, Any]],
    generated_at: str = "",
    show_all: bool = False,
) -> None:
    """Print the ranked table; HOLD pairs collapsed into one line unless show_all."""
    console.print(format_scan_table(results, generated_at, show_all))
    if not show_all:
        holds = [r for r in results if r.get("final_signal") == "HOLD"]
        if holds:
            names = ", ".join(get_display_name(r.get("ticker", "?")) for r in holds)
            console.print(f"[yellow]⚪ HOLD ({len(holds)})[/yellow]: {names}")
