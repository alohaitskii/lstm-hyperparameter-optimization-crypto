"""Ticker utilities: validation, safe filenames, and per-ticker model paths.

Deliberately free of TensorFlow imports so path checks (is_trained,
filter_trained_tickers) stay lightweight — model/lstm_model.py re-exports
get_model_dir/is_trained for spec compliance.
"""
from __future__ import annotations

from pathlib import Path

from utils.helpers import project_root


def ticker_to_safe_name(ticker: str) -> str:
    """'BTC-USD' → 'BTC_USD' (safe for filenames such as CSV logs)."""
    return ticker.replace("-", "_").replace("/", "_").replace(":", "_")


def validate_ticker(ticker: str, supported: list[str]) -> bool:
    """Check if ticker is in the supported list."""
    return ticker in supported


def get_display_name(ticker: str) -> str:
    """'BTC-USD' → 'BTC' (for compact table display)."""
    return ticker.split("-")[0]


def get_model_dir(ticker: str) -> Path:
    """Per-ticker artifact directory: 'BTC-USD' → '<root>/model/saved/BTC-USD/'.

    The dash is kept (valid in directory names) so the folder is identical to
    the Yahoo ticker; only '/' and ':' are sanitized.
    """
    safe = ticker.replace("/", "_").replace(":", "_")
    return project_root() / "model" / "saved" / safe


def is_trained(ticker: str) -> bool:
    """True when a saved model exists for this ticker."""
    return (get_model_dir(ticker) / "model.keras").exists()


def filter_trained_tickers(tickers: list[str]) -> list[str]:
    """Return only tickers that have saved models (order preserved)."""
    return [t for t in tickers if is_trained(t)]
