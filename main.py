"""Entry point for the LSTM crypto prediction system (multi-pair).

Data source: Yahoo Finance (yfinance) — freely accessible from Indonesia and
the same source used by most academic papers on LSTM crypto prediction.
Sentiment from alternative.me Fear & Greed Index. Each of the 22 supported
tickers gets its own independently trained model under model/saved/{TICKER}/.

Usage:
    python main.py --mode train                     # train first ticker (BTC-USD)
    python main.py --mode train --ticker ETH-USD    # train one specific ticker
    python main.py --mode predict --ticker SOL-USD  # signal for one ticker
    python main.py --mode scan                      # ranked table, all trained pairs
    python main.py --mode scan --show-all           # include HOLD detail rows
    python main.py --mode backtest --lookback 200   # single-ticker backtest
    python train_all.py                             # train ALL tickers sequentially
"""
from __future__ import annotations
import ssl
import urllib3
import requests as _req

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch HTTPAdapter — semua library (ccxt, requests) ikut bypass SSL
_orig_send = _req.adapters.HTTPAdapter.send
def _patched_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, request, **kwargs)
_req.adapters.HTTPAdapter.send = _patched_send
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows legacy console defaults to cp1252 which cannot encode the rich
# panel's box-drawing characters — force UTF-8 with graceful replacement.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

# Ensure project root on path for absolute imports when run as script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetcher import fetch_all  # noqa: E402
from data.preprocessor import (  # noqa: E402
    FEATURE_COLUMNS,
    create_sequences,
    fit_scaler,
    prepare_for_prediction,
    prepare_for_training,
    split_chronological,
)
from model.lstm_model import (  # noqa: E402
    build_model,
    load_artifacts,
    save_artifacts,
    train_model,
)
from model.validator import (  # noqa: E402
    print_report,
    save_report,
    walk_forward_validate,
)
from signals.formatter import print_scan_table, print_signal  # noqa: E402
from signals.generator import predict_signal  # noqa: E402
from utils.helpers import ensure_dir, load_config, project_root  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402
from utils.ticker_utils import (  # noqa: E402
    filter_trained_tickers,
    get_model_dir,
    ticker_to_safe_name,
    validate_ticker,
)

console = Console()

# Tickers with fewer clean rows than this after preprocessing are skipped
# (train_all logs them as INSUFFICIENT_DATA and continues).
MIN_TRAINING_ROWS = 100


class InsufficientDataError(RuntimeError):
    """Raised when a ticker lacks enough data to train — skip, don't fail."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_ticker(cfg: dict[str, Any], ticker: str | None) -> str:
    """Default to the first config ticker; validate explicit ones."""
    tickers = cfg["data"].get("tickers") or []
    if not tickers and cfg["data"].get("ticker"):  # legacy single-ticker config
        tickers = [cfg["data"]["ticker"]]

    if ticker is None:
        if not tickers:
            raise SystemExit("config.yaml: data.tickers kosong — tidak ada ticker default.")
        return tickers[0]

    if tickers and not validate_ticker(ticker, tickers):
        raise SystemExit(
            f"Ticker '{ticker}' tidak ada di config.yaml (data.tickers).\n"
            f"Didukung: {', '.join(tickers)}"
        )
    return ticker


def _validation_csv_for(cfg: dict[str, Any], ticker: str) -> str:
    """Per-ticker validation report path: logs/validation_report_BTC_USD.csv"""
    base = Path(cfg["logging"].get("validation_csv", "logs/validation_report.csv"))
    return str(base.with_name(f"{base.stem}_{ticker_to_safe_name(ticker)}{base.suffix}"))


def _log_prediction(signal: dict[str, Any], csv_path: Path, pair: str) -> None:
    """Append a single signal record to the predictions CSV (creating header on first write)."""
    ensure_dir(csv_path.parent)
    new_file = not csv_path.exists()
    fields = [
        "timestamp",
        "pair",
        "signal",
        "model_probability",
        "confidence",
        "entry_price",
        "tp1",
        "tp2",
        "sl",
        "risk_reward",
        "trend_1h",
        "confirmations",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(
            {
                "timestamp": signal.get("timestamp"),
                "pair": pair,
                "signal": signal.get("final_signal"),
                "model_probability": f"{signal.get('model_probability', float('nan')):.6f}",
                "confidence": f"{signal.get('confidence', float('nan')):.6f}",
                "entry_price": f"{signal.get('entry_price', float('nan')):.4f}",
                "tp1": f"{signal.get('tp1', float('nan')):.4f}",
                "tp2": f"{signal.get('tp2', float('nan')):.4f}",
                "sl": f"{signal.get('sl', float('nan')):.4f}",
                "risk_reward": f"{signal.get('risk_reward', float('nan')):.4f}",
                "trend_1h": signal.get("trend_1h"),
                "confirmations": " | ".join(signal.get("confirmations", []) or []),
            }
        )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_train(cfg: dict[str, Any], ticker: str | None = None) -> dict[str, Any]:
    """Train one ticker. Returns {'ticker', 'auc_mean', 'f1_mean'} for reporting.

    Raises InsufficientDataError when the ticker has no/too little data so
    train_all.py can classify it as SKIP rather than FAIL.
    """
    log = get_logger()
    ticker = _resolve_ticker(cfg, ticker)
    log.info(f"=== TRAIN MODE — {ticker} ===")

    # 1. Fetch all data sources
    bundle = fetch_all(cfg, ticker=ticker)
    if bundle["primary"].empty:
        raise InsufficientDataError(f"{ticker}: tidak ada data OHLCV dari Yahoo Finance")

    # 2. Build features + targets with anti-leakage
    prepped = prepare_for_training(bundle, cfg)
    X_2d = prepped["X"]
    y_1d = prepped["y"]
    fgi_encoder = prepped["fgi_encoder"]

    if len(X_2d) < MIN_TRAINING_ROWS:
        raise InsufficientDataError(
            f"INSUFFICIENT_DATA: {ticker} hanya {len(X_2d)} baris setelah preprocessing "
            f"(minimum {MIN_TRAINING_ROWS})"
        )

    log.info(f"Feature matrix shape: {X_2d.shape}, target balance: {np.mean(y_1d):.3f}")

    # 3. Walk-forward validation
    report = walk_forward_validate(X_2d, y_1d, cfg)
    auc_mean = f1_mean = float("nan")
    if not report.empty:
        print_report(report)
        save_report(report, _validation_csv_for(cfg, ticker))
        mean_row = report[report["fold"] == "MEAN"]
        if len(mean_row):
            auc_mean = float(mean_row["auc"].iloc[0])
            f1_mean = float(mean_row["f1"].iloc[0])

    # 4. Final model: chronological split (train+val from earlier data, test from latest)
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    seq_len = feat_cfg["sequence_length"]

    train_sl, val_sl, _test_sl = split_chronological(len(X_2d), val_frac=0.15, test_frac=0.15)
    X_train_raw = X_2d[train_sl]
    X_val_raw = X_2d[val_sl]
    y_train_raw = y_1d[train_sl]
    y_val_raw = y_1d[val_sl]

    # Scaler fit on training fold only (anti-leakage)
    scaler = fit_scaler(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, seq_len)
    log.info(f"Final training sequences: {X_train_seq.shape}, val: {X_val_seq.shape}")

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        raise InsufficientDataError(
            f"{ticker}: data tidak cukup setelah sequencing (seq_len={seq_len})"
        )

    model = build_model((seq_len, X_2d.shape[1]), model_cfg)
    train_model(
        model,
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        model_cfg,
        save_path=get_model_dir(ticker) / "model.keras",
    )

    # 5. Persist artifacts to model/saved/{TICKER}/
    save_artifacts(model, scaler, fgi_encoder, ticker)
    console.print(f"\n[bold green]Training {ticker} selesai.[/bold green]")
    return {"ticker": ticker, "auc_mean": auc_mean, "f1_mean": f1_mean}


def run_predict(cfg: dict[str, Any], ticker: str | None = None) -> dict[str, Any] | None:
    log = get_logger()
    ticker = _resolve_ticker(cfg, ticker)
    log.info(f"=== PREDICT MODE — {ticker} ===")

    model, scaler, fgi_encoder = load_artifacts(ticker)
    if model is None:
        console.print(
            f"[red]Model {ticker} belum dilatih.[/red] Jalankan: "
            f"python main.py --mode train --ticker {ticker}"
        )
        return None

    bundle = fetch_all(cfg, ticker=ticker)
    prep = prepare_for_prediction(bundle, cfg, scaler, fgi_encoder)
    X_seq = prep["X_seq"]
    latest_features = prep["latest_features"]

    signal = predict_signal(model, X_seq, latest_features, cfg)
    signal["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print_signal(signal, pair_name=f"{ticker} (Yahoo Finance)")

    csv_path = project_root() / cfg["logging"]["predictions_csv"]
    _log_prediction(signal, csv_path, pair=ticker)
    log.info(f"Prediction logged → {csv_path}")
    return signal


def run_scan(cfg: dict[str, Any], show_all: bool = False) -> list[dict[str, Any]]:
    """Predict every trained ticker and print a ranked signal table."""
    log = get_logger()
    log.info("=== SCAN MODE ===")

    tickers = cfg["data"].get("tickers") or []
    trained = filter_trained_tickers(tickers)
    if not trained:
        console.print(
            "[red]Belum ada model terlatih.[/red] "
            "Jalankan dulu: python train_all.py (atau --mode train per ticker)"
        )
        return []

    log.info(f"Scanning {len(trained)}/{len(tickers)} ticker terlatih...")
    results: list[dict[str, Any]] = []
    for t in trained:
        try:
            model, scaler, encoder = load_artifacts(t)
            bundle = fetch_all(cfg, ticker=t)  # Parquet cache (TTL 15m) reused when fresh
            prep = prepare_for_prediction(bundle, cfg, scaler, encoder)
            sig = predict_signal(model, prep["X_seq"], prep["latest_features"], cfg)
            results.append({"ticker": t, **sig})
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Scan gagal untuk {t}: {exc}")
            continue

    # Rank: LONG first (confidence desc), then SHORT (confidence desc), HOLD last
    sig_order = {"LONG": 0, "SHORT": 1, "HOLD": 2}
    results.sort(
        key=lambda r: (sig_order.get(r.get("final_signal"), 3), -r.get("confidence", 0.0))
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print_scan_table(results, generated_at=generated_at, show_all=show_all)
    _save_scan_log(results, generated_at)
    return results


def _save_scan_log(results: list[dict[str, Any]], generated_at: str) -> None:
    """Persist the full scan to logs/scan_{timestamp}.csv."""
    log = get_logger()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = project_root() / "logs" / f"scan_{ts}.csv"
    ensure_dir(out_path.parent)
    fields = [
        "generated_at", "ticker", "signal", "confidence", "model_probability",
        "entry_price", "tp1", "tp2", "sl", "risk_reward", "trend_1h", "confirmations",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "generated_at": generated_at,
                    "ticker": r.get("ticker"),
                    "signal": r.get("final_signal"),
                    "confidence": f"{r.get('confidence', float('nan')):.6f}",
                    "model_probability": f"{r.get('model_probability', float('nan')):.6f}",
                    "entry_price": f"{r.get('entry_price', float('nan')):.8f}",
                    "tp1": f"{r.get('tp1', float('nan')):.8f}",
                    "tp2": f"{r.get('tp2', float('nan')):.8f}",
                    "sl": f"{r.get('sl', float('nan')):.8f}",
                    "risk_reward": f"{r.get('risk_reward', float('nan')):.4f}",
                    "trend_1h": r.get("trend_1h"),
                    "confirmations": " | ".join(r.get("confirmations", []) or []),
                }
            )
    log.info(f"Scan log saved → {out_path}")


def run_backtest(cfg: dict[str, Any], lookback: int = 200, ticker: str | None = None) -> None:
    """Walk through the last N candles of the held-out set, generating signals
    and comparing them with the realized close N candles forward.
    """
    log = get_logger()
    ticker = _resolve_ticker(cfg, ticker)
    log.info(f"=== BACKTEST MODE — {ticker}, lookback={lookback} ===")

    feat_cfg = cfg["features"]
    seq_len = feat_cfg["sequence_length"]
    lookahead_n = feat_cfg["lookahead_n"]
    move_threshold = feat_cfg["move_threshold"]

    model, scaler, fgi_encoder = load_artifacts(ticker)
    if model is None:
        console.print(
            f"[red]Model {ticker} belum dilatih.[/red] Jalankan: "
            f"python main.py --mode train --ticker {ticker}"
        )
        return

    bundle = fetch_all(cfg, ticker=ticker)
    # Rebuild features (no shift — caller controls slicing) and clean rows
    from data.preprocessor import add_target, apply_anti_leakage, build_features

    feat_df, _ = build_features(bundle, cfg, fgi_encoder=fgi_encoder)
    labeled = add_target(feat_df, lookahead_n, move_threshold)
    clean = apply_anti_leakage(labeled[FEATURE_COLUMNS + ["target"]])

    if len(clean) < seq_len + lookback:
        log.warning(
            f"Insufficient clean rows ({len(clean)}) for backtest of lookback={lookback}. Reducing."
        )
        lookback = max(0, len(clean) - seq_len - 1)

    X_2d = clean[FEATURE_COLUMNS].values.astype(np.float32)
    y_1d = clean["target"].values.astype(np.float32)
    closes = clean["close"].values
    index = clean.index

    scaled = scaler.transform(X_2d)

    rows: list[dict[str, Any]] = []
    correct = 0
    long_signals = 0
    short_signals = 0
    hold_signals = 0

    end_anchor = len(scaled)
    start_anchor = max(seq_len, end_anchor - lookback)

    for i in range(start_anchor, end_anchor):
        window = scaled[i - seq_len : i].reshape(1, seq_len, scaled.shape[1])

        # Build a "latest_features" snapshot from the row at i (post anti-leakage,
        # so this represents observable state at time i)
        snapshot = clean.iloc[i].to_dict()
        signal = predict_signal(model, window, snapshot, cfg)
        prob = signal["model_probability"]

        actual = int(y_1d[i])
        predicted_long = signal["final_signal"] == "LONG"
        predicted_short = signal["final_signal"] == "SHORT"

        # We count "correct" only on issued (non-HOLD) signals
        hit = False
        if predicted_long:
            long_signals += 1
            hit = (actual == 1)
        elif predicted_short:
            short_signals += 1
            hit = (actual == 0)
        else:
            hold_signals += 1

        if hit:
            correct += 1

        rows.append(
            {
                "timestamp": index[i].isoformat(),
                "model_probability": prob,
                "signal": signal["final_signal"],
                "actual_label": actual,
                "hit": hit,
                "close": float(closes[i]),
                "tp1": signal["tp1"],
                "tp2": signal["tp2"],
                "sl": signal["sl"],
            }
        )

    df = pd.DataFrame(rows)
    issued = long_signals + short_signals
    acc = (correct / issued) if issued else float("nan")
    console.print(
        f"\n[bold]Backtest summary — {ticker}[/bold]\n"
        f"  Total candles evaluated : {len(rows)}\n"
        f"  LONG signals            : {long_signals}\n"
        f"  SHORT signals           : {short_signals}\n"
        f"  HOLD signals            : {hold_signals}\n"
        f"  Hit rate on issued      : {acc:.3f}\n"
    )

    safe = ticker_to_safe_name(ticker)
    out_path = (
        project_root() / "logs"
        / f"backtest_{safe}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    log.info(f"Backtest results saved → {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Crypto LSTM signal generator — multi-pair (Yahoo Finance data)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "scan", "backtest"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Single ticker (e.g. ETH-USD). Defaults to first in config if not set.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Scan mode: show HOLD rows in full detail instead of collapsed",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=200,
        help="Backtest lookback in candles (default: 200)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(level=cfg.get("logging", {}).get("log_level", "INFO"))

    if args.mode == "train":
        try:
            run_train(cfg, ticker=args.ticker)
        except InsufficientDataError as exc:
            console.print(f"[yellow]SKIP:[/yellow] {exc}")
    elif args.mode == "predict":
        run_predict(cfg, ticker=args.ticker)
    elif args.mode == "scan":
        run_scan(cfg, show_all=args.show_all)
    elif args.mode == "backtest":
        run_backtest(cfg, lookback=args.lookback, ticker=args.ticker)


if __name__ == "__main__":
    main()
