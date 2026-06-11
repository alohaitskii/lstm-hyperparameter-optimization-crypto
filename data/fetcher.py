"""Data fetchers: Yahoo Finance OHLCV + Fear & Greed Index.

Why Yahoo Finance?
  - Binance FAPI is geo-blocked in Indonesia; yfinance is freely accessible
    and is the data source used by most Indonesian academic papers on
    LSTM crypto prediction (ticker `BTC-USD`).
  - Prices are the Yahoo/CoinMarketCap spot index, NOT a perpetual contract.
    For decision support the difference is small (<0.05% basis on BTC), but
    derivatives-only data (open interest, funding, L/S ratio) is unavailable.

Yahoo intraday history limits (enforced by Yahoo, handled here):
  interval   max lookback
  1m         7 days
  2m/5m/15m/30m/90m   60 days
  1h (60m)   730 days
  1d/1wk     unlimited

Caching: Parquet files in `data/cache/`. Refetches when cache older than TTL.
A failed fetch returns an empty DataFrame and logs a warning rather than
aborting the pipeline.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests
import yfinance as yf

from utils.helpers import ensure_dir, project_root
from utils.logger import get_logger

log = get_logger()

FGI_URL = "https://api.alternative.me/fng/"

# Default per-request timeout for plain HTTP calls
HTTP_TIMEOUT = 15

# Yahoo interval notation per our config timeframes
YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
}

# Minutes per timeframe (crypto trades 24/7, so candle math is exact)
TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "1d": 1440,
    "1wk": 10080,
}

# Yahoo's maximum lookback window in days per interval (None = unlimited)
YF_MAX_LOOKBACK_DAYS = {
    "1m": 7,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "1h": 729,
    "1d": None,
    "1wk": None,
}


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str | Path, name: str) -> Path:
    root = project_root()
    p = (root / cache_dir) if not Path(cache_dir).is_absolute() else Path(cache_dir)
    ensure_dir(p)
    return p / f"{name}.parquet"


def _cache_is_fresh(path: Path, ttl_minutes: int) -> bool:
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s < ttl_minutes * 60


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Failed to persist cache {path.name}: {exc}")


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# HTTP helper with exponential backoff (used for FGI)
# --------------------------------------------------------------------------- #
def _request_json(
    url: str, params: dict[str, Any] | None = None, retries: int = 3
) -> Any:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning(
                f"HTTP error on {url} (attempt {attempt + 1}/{retries}): {exc}"
            )
            time.sleep(delay)
            delay *= 2
    log.error(f"All retries exhausted for {url}: {last_err}")
    return None


# --------------------------------------------------------------------------- #
# OHLCV via Yahoo Finance
# --------------------------------------------------------------------------- #
def _normalize_yf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to lowercase OHLCV with a UTC tz-aware index."""
    df = raw.copy()

    # yfinance >= 0.2 returns MultiIndex columns (Price, Ticker) even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    # Daily bars come back tz-naive; intraday comes back tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    return df


def fetch_ohlcv(
    ticker: str,
    timeframe: str,
    limit: int,
    cache_dir: str | Path,
    cache_ttl_minutes: int = 15,
    delay_seconds: float = 0.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Yahoo Finance.

    Returns DataFrame indexed by timestamp (UTC) with columns:
        open, high, low, close, volume

    Rate limiting: after a LIVE network fetch (not a cache hit) the function
    sleeps `delay_seconds` so multi-ticker loops (scan / train_all) do not
    hammer Yahoo. Failed downloads are retried `retries` times with growing
    backoff (5s, 10s, ...) before returning an empty frame.

    Note: Yahoo crypto volume is denominated in quote currency (USD). It is
    only consumed relatively (vol_ratio) and then MinMax-scaled, so the unit
    does not affect the model.
    """
    if timeframe not in YF_INTERVAL:
        raise ValueError(
            f"Timeframe '{timeframe}' not supported by Yahoo Finance. "
            f"Pilih salah satu: {sorted(YF_INTERVAL)}"
        )

    safe_name = f"ohlcv_yf_{ticker.replace('-', '_')}_{timeframe}_{limit}"
    cache_file = _cache_path(cache_dir, safe_name)

    if _cache_is_fresh(cache_file, cache_ttl_minutes):
        log.info(f"Using cached OHLCV ({timeframe}) — {cache_file.name}")
        return _load_parquet(cache_file)

    # Work out how many days of history we need, then clamp to Yahoo's cap
    minutes = TF_MINUTES[timeframe]
    days_needed = math.ceil(limit * minutes / 1440) + 2  # +2 days safety buffer
    cap = YF_MAX_LOOKBACK_DAYS.get(timeframe)
    if cap is not None and days_needed > cap:
        log.warning(
            f"Yahoo membatasi riwayat {timeframe} ke {cap} hari — "
            f"permintaan {limit} candle (~{days_needed} hari) akan terpotong."
        )
        days_needed = cap

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_needed)

    log.info(
        f"Fetching OHLCV (Yahoo): ticker={ticker} tf={timeframe} "
        f"window={days_needed}d (target {limit} candle)"
    )

    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                interval=YF_INTERVAL[timeframe],
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                wait = (attempt + 1) * 5
                log.warning(
                    f"{ticker} fetch attempt {attempt + 1}/{retries} gagal, "
                    f"retry dalam {wait}s: {exc}"
                )
                time.sleep(wait)
            else:
                log.error(f"{ticker}: semua retry gagal: {exc}")

    # Rate-limit courtesy delay — applies only after a live network attempt
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if raw is None or raw.empty:
        log.error(f"No OHLCV rows fetched from Yahoo Finance for {ticker}.")
        return pd.DataFrame()

    df = _normalize_yf_frame(raw)
    df = df.tail(limit)

    _save_parquet(df, cache_file)
    log.info(f"OHLCV ({timeframe}) cached: {len(df)} rows")
    return df


# --------------------------------------------------------------------------- #
# Fear & Greed Index (alternative.me — accessible from Indonesia)
# --------------------------------------------------------------------------- #
def fetch_fear_greed_index(
    limit: int = 30,
    cache_dir: str | Path = "data/cache",
    cache_ttl_minutes: int = 60,
) -> pd.DataFrame:
    """Fetch Fear & Greed Index from alternative.me. Index = UTC date."""
    cache_file = _cache_path(cache_dir, f"fgi_{limit}")
    if _cache_is_fresh(cache_file, cache_ttl_minutes):
        log.info(f"Using cached FGI — {cache_file.name}")
        return _load_parquet(cache_file)

    log.info("Fetching Fear & Greed Index")
    data = _request_json(FGI_URL, {"limit": limit})
    if not data or "data" not in data:
        log.warning("FGI fetch failed — returning empty frame")
        return pd.DataFrame(columns=["fgi_value", "fgi_classification"])

    rows = data["data"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True
    ).dt.normalize()
    df["fgi_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fgi_classification"] = df["value_classification"].astype(str)
    df = df.set_index("date")[["fgi_value", "fgi_classification"]].sort_index()

    _save_parquet(df, cache_file)
    return df


# --------------------------------------------------------------------------- #
# Combined fetcher
# --------------------------------------------------------------------------- #
def fetch_all(cfg: Mapping[str, Any], ticker: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch every data source needed by the preprocessor in one call.

    `ticker=None` falls back to the first entry of data.tickers (or legacy
    data.ticker) — preserves the old single-pair behavior.
    """
    d = cfg["data"]
    if ticker is None:
        tickers = d.get("tickers") or []
        ticker = tickers[0] if tickers else d.get("ticker", "BTC-USD")

    delay = float(d.get("fetch_delay_seconds", 0.0))

    primary = fetch_ohlcv(
        ticker, d["primary_tf"], d["candle_limit"], d["cache_dir"],
        d["cache_ttl_minutes"], delay_seconds=delay,
    )
    secondary = fetch_ohlcv(
        ticker, d["secondary_tf"], d["candle_limit"], d["cache_dir"],
        d["cache_ttl_minutes"], delay_seconds=delay,
    )
    # FGI history should cover the whole primary window (daily index, 60d window → 90 entries)
    # Shared across all tickers — the 60-minute cache means scan/train_all hit it once.
    fgi = fetch_fear_greed_index(90, d["cache_dir"], cache_ttl_minutes=60)

    return {
        "primary": primary,
        "secondary": secondary,
        "fgi": fgi,
    }


if __name__ == "__main__":  # pragma: no cover
    from utils.helpers import load_config

    cfg = load_config()
    bundle = fetch_all(cfg)
    for k, v in bundle.items():
        print(f"{k}: {len(v)} rows")
