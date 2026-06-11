"""Signal generation: combine model probability with technical confirmation gates."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from utils.helpers import compute_tp_sl
from utils.logger import get_logger

log = get_logger()


def _direction_from_prob(prob: float, long_thr: float, short_thr: float) -> str:
    if prob >= long_thr:
        return "LONG"
    if prob <= short_thr:
        return "SHORT"
    return "HOLD"


def confirm_signal(direction: str, features: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Apply technical-condition gates.

    Returns (passed, list_of_reason_strings).
    """
    reasons: list[str] = []

    rsi = features.get("rsi_14", float("nan"))
    macd_hist = features.get("macd_hist", float("nan"))
    close = features.get("close", float("nan"))
    bb_lower = features.get("bb_lower", float("nan"))
    bb_upper = features.get("bb_upper", float("nan"))
    stoch_k = features.get("stoch_k", float("nan"))
    fgi_norm = features.get("fgi_normalized", float("nan"))

    if direction == "LONG":
        if np.isfinite(rsi) and rsi < 50:
            reasons.append(f"RSI(14)={rsi:.1f} — momentum mendekati oversold")
        if np.isfinite(macd_hist) and macd_hist > 0:
            reasons.append("MACD histogram positif (momentum naik)")
        if np.isfinite(close) and np.isfinite(bb_lower) and close < bb_lower * 1.005:
            reasons.append("Harga dekat Bollinger Lower Band (potensi rebound)")
        if np.isfinite(stoch_k) and stoch_k < 25:
            reasons.append(f"Stochastic %K={stoch_k:.1f} — area oversold")
        if np.isfinite(fgi_norm) and fgi_norm < 0.35:
            reasons.append(f"Fear & Greed={fgi_norm * 100:.0f} (Fear — sinyal kontrarian bullish)")

    elif direction == "SHORT":
        if np.isfinite(rsi) and rsi > 50:
            reasons.append(f"RSI(14)={rsi:.1f} — momentum mendekati overbought")
        if np.isfinite(macd_hist) and macd_hist < 0:
            reasons.append("MACD histogram negatif (momentum turun)")
        if np.isfinite(close) and np.isfinite(bb_upper) and close > bb_upper * 0.995:
            reasons.append("Harga dekat Bollinger Upper Band (potensi rejection)")
        if np.isfinite(stoch_k) and stoch_k > 75:
            reasons.append(f"Stochastic %K={stoch_k:.1f} — area overbought")
        if np.isfinite(fgi_norm) and fgi_norm > 0.65:
            reasons.append(f"Fear & Greed={fgi_norm * 100:.0f} (Greed — sinyal kontrarian bearish)")

    passed = len(reasons) >= 2  # min_confirmations enforced at call site too
    return passed, reasons


def _trend_label(features: Mapping[str, Any]) -> str:
    """Coarse 1h trend label using EMA21_1h vs current close."""
    close = features.get("close", float("nan"))
    ema21_1h = features.get("ema_21_1h", float("nan"))
    macd_hist_1h = features.get("macd_hist_1h", float("nan"))
    if not (np.isfinite(close) and np.isfinite(ema21_1h)):
        return "UNKNOWN"
    if close > ema21_1h and (not np.isfinite(macd_hist_1h) or macd_hist_1h >= 0):
        return "BULLISH"
    if close < ema21_1h and (not np.isfinite(macd_hist_1h) or macd_hist_1h <= 0):
        return "BEARISH"
    return "NEUTRAL"


def predict_signal(
    model,
    X_seq: np.ndarray,
    latest_features: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Run model + signal-confirmation gates and return a structured signal dict."""
    sig_cfg = cfg["signals"]
    long_thr = sig_cfg.get("long_threshold", 0.62)
    short_thr = sig_cfg.get("short_threshold", 0.38)
    min_conf = sig_cfg.get("min_confirmations", 2)

    prob = float(model.predict(X_seq, verbose=0).flatten()[0])
    direction = _direction_from_prob(prob, long_thr, short_thr)

    entry_price = float(latest_features.get("close", float("nan")))
    atr = float(latest_features.get("atr_14", float("nan")))

    tp_sl: dict[str, float] = {}
    if direction in ("LONG", "SHORT") and np.isfinite(entry_price) and np.isfinite(atr):
        tp_sl = compute_tp_sl(entry_price, atr, direction, sig_cfg)

    confirmed = False
    reasons: list[str] = []
    if direction in ("LONG", "SHORT"):
        confirmed, reasons = confirm_signal(direction, latest_features)
        if len(reasons) < min_conf:
            confirmed = False

    if direction == "LONG":
        confidence = prob
    elif direction == "SHORT":
        confidence = 1.0 - prob
    else:
        confidence = max(prob, 1.0 - prob)

    final_signal = direction if (direction == "HOLD" or confirmed) else "HOLD"

    trend_1h = _trend_label(latest_features)

    # Build narrow entry zone around current close (±0.05% by default)
    entry_low = entry_price * 0.9995
    entry_high = entry_price * 1.0005

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_probability": prob,
        "raw_direction": direction,
        "final_signal": final_signal,
        "confidence": confidence,
        "confirmations": reasons,
        "confirmations_passed": confirmed,
        "entry_price": entry_price,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "tp1": tp_sl.get("tp1", float("nan")),
        "tp2": tp_sl.get("tp2", float("nan")),
        "sl": tp_sl.get("sl", float("nan")),
        "risk_reward": tp_sl.get("rr", float("nan")),
        "atr_14": atr,
        "trend_1h": trend_1h,
    }
