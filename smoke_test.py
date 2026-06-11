"""Offline smoke test: indicators, preprocessor pipeline, and model build.

Runs without network access using synthetic OHLCV data. Used to validate the
pipeline against installed library versions (pandas 3.x, numpy 2.x, Keras 3).

Usage: python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same UTF-8 fix as main.py for legacy Windows consoles
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def make_synthetic_bundle(n_15m: int = 1200) -> dict[str, pd.DataFrame]:
    """Random-walk OHLCV at 15m + 1h plus synthetic FGI history."""
    rng = np.random.default_rng(42)
    idx_15m = pd.date_range("2026-04-01", periods=n_15m, freq="15min", tz="UTC")

    log_ret = rng.normal(0, 0.002, n_15m)
    close = 60000 * np.exp(np.cumsum(log_ret))
    spread = np.abs(rng.normal(0, 0.001, n_15m)) * close
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(1e6, 5e7, n_15m)

    primary = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx_15m,
    )
    primary.index.name = "timestamp"

    secondary = (
        primary.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    days = pd.date_range(idx_15m[0].normalize(), idx_15m[-1].normalize(), freq="D", tz="UTC")
    fgi_vals = rng.integers(10, 90, len(days))
    classes = pd.cut(
        fgi_vals,
        bins=[-1, 24, 44, 55, 75, 101],
        labels=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
    ).astype(str)
    fgi = pd.DataFrame({"fgi_value": fgi_vals, "fgi_classification": classes}, index=days)
    fgi.index.name = "date"

    return {"primary": primary, "secondary": secondary, "fgi": fgi}


def main() -> int:
    failures: list[str] = []

    # ---- 1. Indicators on synthetic series -------------------------------- #
    from data import indicators as ind

    bundle = make_synthetic_bundle()
    close = bundle["primary"]["close"]
    high = bundle["primary"]["high"]
    low = bundle["primary"]["low"]

    rsi = ind.rsi(close)
    assert rsi.dropna().between(0, 100).all(), "RSI out of [0,100]"
    macd_l, macd_s, macd_h = ind.macd(close)
    assert np.allclose((macd_l - macd_s).dropna(), macd_h.dropna()), "MACD hist mismatch"
    upper, mid, lower, pct, width = ind.bollinger(close)
    valid = upper.notna()
    assert (upper[valid] >= lower[valid]).all(), "BB upper < lower"
    atr = ind.atr(high, low, close)
    assert (atr.dropna() >= 0).all(), "ATR negative"
    k, d = ind.stochastic(high, low, close)
    assert k.dropna().between(-1e-9, 100 + 1e-9).all(), "Stoch %K out of range"
    print("OK  indicators (RSI/MACD/BB/ATR/Stoch)")

    # ---- 2. Full preprocessor pipeline ------------------------------------ #
    from data.preprocessor import (
        FEATURE_COLUMNS,
        add_target,
        apply_anti_leakage,
        build_features,
        create_sequences,
        fit_scaler,
        prepare_for_training,
    )
    from utils.helpers import load_config

    cfg = load_config()
    prepped = prepare_for_training(bundle, cfg)
    X, y = prepped["X"], prepped["y"]
    assert X.shape[1] == len(FEATURE_COLUMNS) == 25, f"Expected 25 features, got {X.shape[1]}"
    assert not np.isnan(X).any(), "NaN left in feature matrix"
    assert set(np.unique(y)) <= {0.0, 1.0}, "Targets not binary"
    print(f"OK  preprocessor — X={X.shape}, positives={y.mean():.3f}")

    # ---- 3. Anti-leakage spot-check ---------------------------------------- #
    # The close feature at row t must equal the raw close at t-1 (shift(1))
    feat_df, _ = build_features(bundle, cfg)
    labeled = add_target(feat_df, cfg["features"]["lookahead_n"], cfg["features"]["move_threshold"])
    clean = apply_anti_leakage(labeled[FEATURE_COLUMNS + ["target"]])
    t = clean.index[100]
    pos = feat_df.index.get_loc(t)
    raw_prev_close = feat_df["close"].iloc[pos - 1]
    assert np.isclose(clean.loc[t, "close"], raw_prev_close), "Anti-leakage shift broken"
    print("OK  anti-leakage — features at t == raw data at t-1")

    # ---- 4. Sequences + scaler --------------------------------------------- #
    scaler = fit_scaler(X[:500])
    Xs = scaler.transform(X[:700])
    X_seq, y_seq = create_sequences(Xs, y[:700], cfg["features"]["sequence_length"])
    assert X_seq.shape[1:] == (60, 25), f"Bad sequence shape {X_seq.shape}"
    print(f"OK  sequences — {X_seq.shape}")

    # ---- 5. Model build + 1 epoch micro-train (Keras 3) -------------------- #
    from model.lstm_model import build_model

    model = build_model((60, 25), cfg["model"])
    n_params = model.count_params()
    micro_cfg = dict(cfg["model"])
    micro_cfg["epochs"] = 1
    micro_cfg["batch_size"] = 32
    hist = model.fit(
        X_seq[:128], y_seq[:128],
        validation_data=(X_seq[128:192], y_seq[128:192]),
        epochs=1, batch_size=32, verbose=0,
    )
    prob = model.predict(X_seq[:2], verbose=0)
    assert prob.shape == (2, 1) and ((prob >= 0) & (prob <= 1)).all(), "Bad prediction output"
    print(f"OK  model — {n_params:,} params, 1-epoch fit, sigmoid output valid")

    # ---- 6. Signal generation ---------------------------------------------- #
    from signals.generator import predict_signal
    from signals.formatter import print_signal

    latest = feat_df.iloc[-1].to_dict()
    signal = predict_signal(model, X_seq[:1], latest, cfg)
    assert signal["final_signal"] in ("LONG", "SHORT", "HOLD")
    assert 0.0 <= signal["model_probability"] <= 1.0
    print(f"OK  signal — {signal['final_signal']} (p={signal['model_probability']:.3f})")
    print_signal(signal, pair_name="SYNTHETIC-TEST")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
