"""Feature engineering, sequence creation, and anti-leakage controls.

Pipeline overview:
  1. build_features()     — 25-feature matrix (OHLCV + technicals + 1h confirm + sentiment)
  2. add_target()         — binary direction label (LOOKAHEAD_N candles ahead)
  3. apply_anti_leakage() — shift features by 1, drop NaN rows
  4. create_sequences()   — slide window of length SEQUENCE_LENGTH
  5. split_chronological()— chronological train/val/test split (no shuffle)

Technical indicators are computed natively (data/indicators.py) — no pandas-ta.
All operations preserve UTC timestamps. No future data leaks into features.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data import indicators as ind
from utils.helpers import ensure_dir, project_root
from utils.logger import get_logger

log = get_logger()


# Feature order is part of the contract — the model is trained with this layout.
FEATURE_COLUMNS: list[str] = [
    # OHLCV (5)
    "open", "high", "low", "close", "volume",
    # Technicals (15)
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct", "bb_width",
    "atr_14", "ema_9", "ema_21", "ema_50",
    "stoch_k", "stoch_d", "vol_ratio",
    # 1h confirmation (3)
    "rsi_1h", "ema_21_1h", "macd_hist_1h",
    # Sentiment (2)
    "fgi_normalized", "fgi_classification_encoded",
]

# Canonical FGI classification labels (so the encoder is reproducible)
FGI_CLASSES = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]


# --------------------------------------------------------------------------- #
# Indicator computation
# --------------------------------------------------------------------------- #
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for the primary (15m) OHLCV frame."""
    out = df.copy()

    out["rsi_14"] = ind.rsi(out["close"], length=14)

    macd_line, macd_signal, macd_hist = ind.macd(out["close"], fast=12, slow=26, signal=9)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower, bb_pct, bb_width = ind.bollinger(out["close"], length=20, std_mult=2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_pct"] = bb_pct
    out["bb_width"] = bb_width

    out["ema_9"] = ind.ema(out["close"], length=9)
    out["ema_21"] = ind.ema(out["close"], length=21)
    out["ema_50"] = ind.ema(out["close"], length=50)

    out["atr_14"] = ind.atr(out["high"], out["low"], out["close"], length=14)

    stoch_k, stoch_d = ind.stochastic(out["high"], out["low"], out["close"], k=14, smooth_k=3, d=3)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    out["vol_sma_20"] = ind.sma(out["volume"], length=20)
    out["vol_ratio"] = out["volume"] / out["vol_sma_20"].replace(0.0, np.nan)

    return out


def _add_1h_features(primary_15m: pd.DataFrame, secondary_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute 1h trend indicators and broadcast (forward-fill) to the 15m index."""
    if secondary_1h.empty:
        log.warning("Secondary (1h) frame empty — confirmation features filled with NaN")
        out = primary_15m.copy()
        out["rsi_1h"] = np.nan
        out["ema_21_1h"] = np.nan
        out["macd_hist_1h"] = np.nan
        return out

    sec = secondary_1h.copy()
    sec["rsi_1h"] = ind.rsi(sec["close"], length=14)
    sec["ema_21_1h"] = ind.ema(sec["close"], length=21)
    _, _, macd_hist_1h = ind.macd(sec["close"], fast=12, slow=26, signal=9)
    sec["macd_hist_1h"] = macd_hist_1h

    sec = sec[["rsi_1h", "ema_21_1h", "macd_hist_1h"]]
    merged = primary_15m.join(sec.reindex(primary_15m.index, method="ffill"))
    return merged


# --------------------------------------------------------------------------- #
# Sentiment merging
# --------------------------------------------------------------------------- #
def _merge_fgi(
    df: pd.DataFrame, fgi: pd.DataFrame, encoder: LabelEncoder | None = None
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Broadcast daily FGI value+classification across the intraday index."""
    out = df.copy()
    enc = encoder or _fit_fgi_encoder()

    if fgi.empty:
        out["fgi_normalized"] = np.nan
        out["fgi_classification_encoded"] = np.nan
        return out, enc

    fgi_local = fgi.copy()
    fgi_local["fgi_normalized"] = fgi_local["fgi_value"] / 100.0

    # safe_transform: unknown labels → -1
    classes_set = set(enc.classes_.tolist())
    fgi_local["fgi_classification_encoded"] = fgi_local["fgi_classification"].apply(
        lambda v: enc.transform([v])[0] if v in classes_set else -1
    )

    fgi_features = fgi_local[["fgi_normalized", "fgi_classification_encoded"]]

    # Map each candle to its UTC date for the join
    out_idx_date = out.index.normalize()
    fgi_aligned = fgi_features.reindex(out_idx_date, method="ffill")
    fgi_aligned.index = out.index
    out = out.join(fgi_aligned)
    return out, enc


def _fit_fgi_encoder() -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(FGI_CLASSES)
    return enc


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_features(
    bundle: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    fgi_encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Build the full feature matrix from a fetched data bundle."""
    primary = bundle["primary"].copy()
    if primary.empty:
        raise ValueError("Primary OHLCV is empty — cannot build features.")

    df = _add_indicators(primary)
    df = _add_1h_features(df, bundle["secondary"])
    df, encoder = _merge_fgi(df, bundle["fgi"], encoder=fgi_encoder)

    # Guarantee every model column exists (missing sources → NaN, logged)
    needed = list(dict.fromkeys(FEATURE_COLUMNS))
    missing = [c for c in needed if c not in df.columns]
    for col in missing:
        df[col] = np.nan
        log.warning(f"Feature column missing → filled NaN: {col}")

    log.info(f"Feature matrix built — shape={df.shape}, model columns={len(needed)}")
    return df, encoder


def add_target(
    df: pd.DataFrame,
    lookahead_n: int,
    move_threshold: float,
) -> pd.DataFrame:
    """Binary direction label: 1 if close.shift(-N) > close*(1+threshold) else 0."""
    out = df.copy()
    future_close = out["close"].shift(-lookahead_n)
    # Build as float so the unknowable tail rows can hold NaN
    # (pandas >= 3.0 raises on assigning NaN into an int column)
    target = (future_close > out["close"] * (1 + move_threshold)).astype(float)
    target[future_close.isna()] = np.nan
    out["target"] = target
    return out


def apply_anti_leakage(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Shift every feature by 1, then drop rows with NaN target or features.

    This is the canonical guard against look-ahead bias: at evaluation time t,
    only data observable at t-1 is used to predict y_t.
    """
    feats = [c for c in df.columns if c != target_col]
    out = df.copy()
    out[feats] = out[feats].shift(1)

    # Drop rows where target is NaN (caused by lookahead)
    out = out.dropna(subset=[target_col])
    # Drop rows where any feature is still NaN (indicator warm-up rows)
    out = out.dropna(subset=feats)
    out[target_col] = out[target_col].astype(int)

    log.info(f"Anti-leakage applied. Rows after dropna: {len(out)}")
    return out


def create_sequences(
    X: np.ndarray, y: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length `sequence_length` over rows.

    For each window ending at row i, target is y[i].
    Output shape: (n_samples, sequence_length, n_features), (n_samples,)
    """
    if len(X) <= sequence_length:
        return np.empty((0, sequence_length, X.shape[1])), np.empty((0,))

    xs, ys = [], []
    for i in range(sequence_length, len(X)):
        xs.append(X[i - sequence_length : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_chronological(
    n: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[slice, slice, slice]:
    """Index slices for train/val/test that respect time order."""
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)
    return train, val, test


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Fit a MinMaxScaler on training rows only (anti-leakage)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def save_scaler(scaler: MinMaxScaler, path: str | Path) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    ensure_dir(p.parent)
    with open(p, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Scaler saved → {p}")


def load_scaler(path: str | Path) -> MinMaxScaler:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "rb") as f:
        return pickle.load(f)


def save_fgi_encoder(encoder: LabelEncoder, path: str | Path) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    ensure_dir(p.parent)
    with open(p, "wb") as f:
        pickle.dump(encoder, f)
    log.info(f"FGI encoder saved → {p}")


def load_fgi_encoder(path: str | Path) -> LabelEncoder:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Full pipeline helpers
# --------------------------------------------------------------------------- #
def prepare_for_training(
    bundle: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """End-to-end build → label → de-leak → return raw 2D matrix + targets + index."""
    feats_cfg = cfg["features"]

    feat_df, encoder = build_features(bundle, cfg)
    labeled = add_target(feat_df, feats_cfg["lookahead_n"], feats_cfg["move_threshold"])
    clean = apply_anti_leakage(labeled[FEATURE_COLUMNS + ["target"]])

    X_2d = clean[FEATURE_COLUMNS].values.astype(np.float32)
    y_1d = clean["target"].values.astype(np.float32)
    index = clean.index

    return {
        "X": X_2d,
        "y": y_1d,
        "index": index,
        "feature_names": list(FEATURE_COLUMNS),
        "fgi_encoder": encoder,
        "raw_df": clean,
    }


def prepare_for_prediction(
    bundle: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    scaler: MinMaxScaler,
    fgi_encoder: LabelEncoder,
) -> dict[str, Any]:
    """Build features for inference. Returns the last sequence + the most recent feature row.

    Note: for prediction we do NOT shift features by 1 — we want the latest
    observable state of the world up to and including the most recent closed candle.
    """
    feats_cfg = cfg["features"]
    seq_len = feats_cfg["sequence_length"]

    feat_df, _ = build_features(bundle, cfg, fgi_encoder=fgi_encoder)
    feat_only = feat_df[FEATURE_COLUMNS].copy()
    feat_only = feat_only.dropna()

    if len(feat_only) < seq_len:
        raise ValueError(
            f"Not enough clean rows ({len(feat_only)}) to build sequence of length {seq_len}"
        )

    latest_window_2d = feat_only.tail(seq_len).values.astype(np.float32)
    latest_window_scaled = scaler.transform(latest_window_2d).astype(np.float32)
    X_seq = latest_window_scaled.reshape(1, seq_len, len(FEATURE_COLUMNS))

    latest_features_row = feat_df.iloc[-1].to_dict()
    latest_timestamp = feat_df.index[-1]

    return {
        "X_seq": X_seq,
        "latest_features": latest_features_row,
        "latest_timestamp": latest_timestamp,
        "raw_df": feat_df,
    }
