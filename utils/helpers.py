"""Reusable helper utilities: config loading, ATR-based TP/SL, safe numeric ops."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config from disk."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Division that returns `default` when denominator is zero/NaN."""
    if den is None or den == 0 or np.isnan(den):
        return default
    return float(num) / float(den)


def pct_change(new: float, old: float) -> float:
    """Percent change as a float (e.g., 0.012 = +1.2%)."""
    return safe_div(new - old, old, default=0.0)


def compute_tp_sl(
    entry_price: float,
    atr: float,
    direction: str,
    cfg: Mapping[str, Any],
) -> dict[str, float]:
    """Compute TP1, TP2, SL based on ATR multipliers.

    direction: "LONG" or "SHORT"
    """
    tp1_mult = cfg.get("atr_tp1_multiplier", 1.5)
    tp2_mult = cfg.get("atr_tp2_multiplier", 2.5)
    sl_mult = cfg.get("atr_sl_multiplier", 1.0)

    if direction == "LONG":
        tp1 = entry_price + tp1_mult * atr
        tp2 = entry_price + tp2_mult * atr
        sl = entry_price - sl_mult * atr
    elif direction == "SHORT":
        tp1 = entry_price - tp1_mult * atr
        tp2 = entry_price - tp2_mult * atr
        sl = entry_price + sl_mult * atr
    else:
        return {"tp1": entry_price, "tp2": entry_price, "sl": entry_price, "rr": 0.0}

    risk = abs(entry_price - sl)
    reward = abs(tp1 - entry_price)
    rr = safe_div(reward, risk, default=0.0)

    return {
        "tp1": float(tp1),
        "tp2": float(tp2),
        "sl": float(sl),
        "rr": float(rr),
    }


def fmt_pct(x: float, places: int = 2) -> str:
    """Format a fraction as a signed percent string (e.g. +1.23%)."""
    return f"{x * 100:+.{places}f}%"


def fmt_usd(x: float) -> str:
    """Format a number as USDT with thousands separators."""
    return f"${x:,.2f}"


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


def set_global_seed(seed: int, include_tf: bool = True) -> None:
    """Seed random, numpy, and (lazily) TensorFlow for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    if include_tf:
        # Lazy import so lightweight callers don't pay the TF import cost
        import tensorflow as tf

        tf.random.set_seed(seed)
