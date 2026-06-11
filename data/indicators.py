"""Native technical indicator implementations (pure pandas/numpy).

Replaces pandas-ta, whose published build (0.3.14b0) is incompatible with
numpy >= 2.0 (`from numpy import NaN` was removed upstream).

Formulas follow the standard definitions:
  - RSI and ATR use Wilder's smoothing (RMA: ewm with alpha=1/length)
  - MACD = EMA(fast) - EMA(slow); signal = EMA(MACD, signal); hist = MACD - signal
  - Bollinger: mid = SMA(length); bands at mid +/- std_mult * rolling std
  - Stochastic: fast %K over `k` periods, smoothed by `smooth_k`, %D = SMA(d) of %K

All functions return Series/tuples aligned to the input index, with NaN
during the warm-up window (min_periods enforced).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (span-based, standard definition)."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (Running Moving Average) used by RSI and ATR."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (pure uptrend in window), RSI saturates at 100
    out = out.where(avg_loss != 0.0, 100.0)
    # Keep warm-up NaN
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    close: pd.Series, length: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper, mid, lower, pct, width) where:
      pct   = (close - lower) / (upper - lower)   — %B position inside bands
      width = (upper - lower) / mid               — relative bandwidth
    """
    mid = sma(close, length)
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    band_range = (upper - lower).replace(0.0, np.nan)
    pct = (close - lower) / band_range
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return upper, mid, lower, pct, width


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Average True Range with Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr, length)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    smooth_k: int = 3,
    d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator. Returns (%K smoothed, %D)."""
    lowest_low = low.rolling(window=k, min_periods=k).min()
    highest_high = high.rolling(window=k, min_periods=k).max()
    rng = (highest_high - lowest_low).replace(0.0, np.nan)
    k_raw = 100.0 * (close - lowest_low) / rng
    k_smooth = k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d_line = k_smooth.rolling(window=d, min_periods=d).mean()
    return k_smooth, d_line
