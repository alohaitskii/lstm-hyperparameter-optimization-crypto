"""Experiment orchestration: GA vs Grid Search vs manual baseline per ticker.

Per (ticker × method):
    1. Search best_hp — GA / budgeted Grid / manual (= current config.yaml
       values, the same hyperparameters that produced model/saved/; the
       baseline is RE-EVALUATED through the identical protocol, not merely
       loaded, so the comparison is apples-to-apples).
    2. Final evaluation of best_hp with the FULL walk_forward_validate
       (final_splits folds, full epochs + early stopping) → wf_auc, wf_f1.
    3. Train the winning model on the chronological split and save it to
       {output_root}/model/optimized/{TICKER}/{method}/ — model/saved/ is
       NEVER touched.
    4. Backtest the winning model (reuses main.run_backtest with preloaded
       artifacts) → hit_rate.
    5. Append the result row to the master CSV IMMEDIATELY (checkpoint) —
       --skip-done resumes at (ticker × method) granularity.

Colab-safety: incremental CSV writes, ga_history written per generation,
clear_session() between heavy phases, output_root can point to a mounted
Google Drive so results survive a dropped session.
"""
from __future__ import annotations

import csv
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.fetcher import fetch_all
from data.preprocessor import (
    create_sequences,
    fit_scaler,
    prepare_for_training,
    split_chronological,
)
from model.lstm_model import build_model, save_artifacts, train_model
from model.validator import walk_forward_validate
from optimization.fitness import build_cfg_for_hp, evaluate
from optimization.genetic_algorithm import run_ga
from optimization.grid_search import run_grid
from optimization.search_space import SearchSpace
from utils.helpers import ensure_dir, project_root
from utils.logger import get_logger
from utils.ticker_utils import ticker_to_safe_name

log = get_logger()

METHODS = ("ga", "grid", "manual")

RESULT_FIELDS = [
    "ticker", "method", "seed", "best_hp_json", "val_auc_search",
    "wf_auc", "wf_f1", "backtest_hit_rate",
    # Hit-rate = benar / sinyal diterbitkan; HOLD tidak masuk penyebut, jadi
    # jumlah sinyal wajib dicatat agar hit-rate antarmetode bisa dibandingkan
    # secara adil (model konservatif otomatis terlihat lebih baik tanpa ini).
    "n_signals_issued", "n_signals_hold", "n_candles_backtest",
    "n_evals", "duration_min",
    "search_epochs", "final_splits", "timestamp",
]


def _blank_if_nan(v: Any) -> Any:
    """CSV-friendly: NaN → string kosong, selain itu nilai apa adanya."""
    try:
        return "" if not np.isfinite(v) else v
    except TypeError:
        return v


# --------------------------------------------------------------------------- #
# Paths & checkpoint IO
# --------------------------------------------------------------------------- #
def resolve_output_root(cfg: Mapping[str, Any], cli_output_root: str | None) -> Path:
    """CLI arg > config optimization.output_root > repo root. Useful on Colab
    to point at a mounted Google Drive so results persist across sessions."""
    raw = cli_output_root or (cfg.get("optimization") or {}).get("output_root")
    return Path(raw) if raw else project_root()


def master_csv_path(output_root: Path) -> Path:
    """Stable (non-timestamped) append-only checkpoint — required so
    --skip-done can find previous results across sessions."""
    return output_root / "logs" / "optimization_results.csv"


def load_done(path: Path) -> set[tuple[str, str]]:
    """(ticker, method) pairs already present in the master CSV."""
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["ticker"], row["method"]))
    return done


def _migrate_csv_header(path: Path) -> None:
    """Rewrite an older CSV in place so it matches the current RESULT_FIELDS.

    Appending new columns to a file written with the previous header would
    shift every value and corrupt the already-valid GA and manual rows. When
    the header differs we back the file up, then rewrite it with the current
    fieldnames — old rows preserved as-is, new columns blank-filled. This
    keeps --skip-done working so finished runs are never repeated.
    """
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8", newline="") as f:
        try:
            header = next(csv.reader(f))
        except StopIteration:
            return  # file kosong — writer akan menulis header baru
    if header == RESULT_FIELDS:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    shutil.copy2(path, backup)

    with open(path, "r", encoding="utf-8", newline="") as f:
        old_rows = list(csv.DictReader(f))

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in old_rows:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})

    log.info(
        f"Header CSV dimigrasi ke skema baru: {path.name} "
        # ASCII saja: migrasi ini jalur kritis data skripsi, pesannya harus
        # tetap tampil walau dipanggil dari entry point tanpa perbaikan UTF-8
        f"({len(old_rows)} baris lama dipertahankan, backup -> {backup.name})"
    )


def append_result(path: Path, row: dict[str, Any]) -> None:
    """Append one result row immediately (checkpoint granularity)."""
    ensure_dir(path.parent)
    _migrate_csv_header(path)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _ga_history_writer(output_root: Path, ticker: str, ts: str):
    """Returns an on_generation callback that appends rows incrementally."""
    path = output_root / "logs" / f"ga_history_{ticker_to_safe_name(ticker)}_{ts}.csv"
    ensure_dir(path.parent)

    def on_generation(row: dict[str, Any]) -> None:
        new_file = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["generation", "best_fitness", "mean_fitness", "n_evals", "best_hp_json"])
            w.writerow(
                [
                    row["generation"],
                    f"{row['best_fitness']:.6f}",
                    f"{row['mean_fitness']:.6f}",
                    row["n_evals"],
                    json.dumps(row["best_hp"]),
                ]
            )

    return on_generation


def _grid_history_writer(output_root: Path, ticker: str, ts: str):
    """Returns an on_eval callback that appends grid rows incrementally.

    `best_so_far` is tracked in the closure so the convergence curve can be
    plotted straight from the CSV without post-processing — the thesis needs
    fitness-vs-evaluations for BOTH methods, not only the GA.
    """
    path = output_root / "logs" / f"grid_history_{ticker_to_safe_name(ticker)}_{ts}.csv"
    ensure_dir(path.parent)
    best_so_far = -float("inf")

    def on_eval(row: dict[str, Any]) -> None:
        nonlocal best_so_far
        best_so_far = max(best_so_far, float(row["fitness"]))
        new_file = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["eval", "fitness", "best_so_far", "hp_json"])
            w.writerow(
                [
                    row["eval"],
                    f"{row['fitness']:.6f}",
                    f"{best_so_far:.6f}",
                    json.dumps(row["hp"]),
                ]
            )

    return on_eval


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def manual_hp_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Baseline manual = the hyperparameters currently in config.yaml
    (the ones that produced the model/saved/ artifacts)."""
    units = cfg["model"].get("lstm_units", [128, 64])
    return {
        "sequence_length": int(cfg["features"]["sequence_length"]),
        "lstm_units_1": int(units[0]),
        "lstm_units_2": int(units[1]),
        "dropout_rate": float(cfg["model"].get("dropout_rate", 0.2)),
        "learning_rate": float(cfg["model"].get("learning_rate", 0.001)),
        "batch_size": int(cfg["model"].get("batch_size", 32)),
    }


def prepare_ticker_data(cfg: Mapping[str, Any], ticker: str) -> dict[str, Any]:
    """fetch_all + prepare_for_training → {X, y, fgi_encoder}. Raises
    ValueError when the ticker has no usable data."""
    bundle = fetch_all(cfg, ticker=ticker)
    if bundle["primary"].empty:
        raise ValueError(f"{ticker}: tidak ada data OHLCV")
    prepped = prepare_for_training(bundle, cfg)
    return prepped


def final_evaluate_and_save(
    ticker: str,
    method: str,
    best_hp: Mapping[str, Any],
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    fgi_encoder,
    cfg: Mapping[str, Any],
    output_root: Path,
    seed: int,
    lookback: int = 200,
) -> dict[str, float]:
    """Full walk-forward on best_hp, train + save winner, backtest hit-rate."""
    import tensorflow as tf

    from main import run_backtest  # deferred: main applies its own patches

    opt_cfg = cfg.get("optimization") or {}
    final_splits = int(opt_cfg.get("final_splits", 3))
    final_epochs = opt_cfg.get("final_epochs")  # None → full model.epochs

    eval_cfg = build_cfg_for_hp(cfg, best_hp, epochs=final_epochs, n_splits=final_splits)
    seq_len = int(best_hp["sequence_length"])

    # --- 1. Reported metrics: full walk-forward validation ----------------- #
    log.info(f"[{ticker}/{method}] Walk-forward final ({final_splits} fold)...")
    report = walk_forward_validate(X_2d, y_1d, eval_cfg)
    wf_auc = wf_f1 = float("nan")
    if not report.empty:
        mean_row = report[report["fold"] == "MEAN"]
        if len(mean_row):
            wf_auc = float(mean_row["auc"].iloc[0])
            wf_f1 = float(mean_row["f1"].iloc[0])

    # --- 2. Train the winning model (chronological split, like run_train) -- #
    log.info(f"[{ticker}/{method}] Training model pemenang...")
    train_sl, val_sl, _ = split_chronological(len(X_2d))
    scaler = fit_scaler(X_2d[train_sl])
    X_train_seq, y_train_seq = create_sequences(
        scaler.transform(X_2d[train_sl]), y_1d[train_sl], seq_len
    )
    X_val_seq, y_val_seq = create_sequences(
        scaler.transform(X_2d[val_sl]), y_1d[val_sl], seq_len
    )
    model = build_model((seq_len, X_2d.shape[1]), eval_cfg["model"])
    train_model(
        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        eval_cfg["model"], save_path=None, verbose=0,
    )

    # --- 3. Save winner to model/optimized/ (baseline untouched) ----------- #
    out_dir = output_root / "model" / "optimized" / ticker / method
    save_artifacts(model, scaler, fgi_encoder, ticker, out_dir=out_dir)
    with open(out_dir / "hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(
            {"hp": dict(best_hp), "wf_auc": wf_auc, "wf_f1": wf_f1, "seed": seed},
            f, indent=2,
        )

    # --- 4. Backtest hit-rate (reuse main.run_backtest, preloaded model) --- #
    log.info(f"[{ticker}/{method}] Backtest hit-rate (lookback={lookback})...")
    bt = run_backtest(
        eval_cfg, lookback=lookback, ticker=ticker,
        preloaded=(model, scaler, fgi_encoder), tag=method,
    )
    hit_rate = float(bt["hit_rate"]) if bt else float("nan")
    n_issued = int(bt["long"] + bt["short"]) if bt else float("nan")
    n_hold = int(bt["hold"]) if bt else float("nan")
    n_candles = int(bt["evaluated"]) if bt else float("nan")

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "wf_auc": wf_auc,
        "wf_f1": wf_f1,
        "hit_rate": hit_rate,
        "n_signals_issued": n_issued,
        "n_signals_hold": n_hold,
        "n_candles_backtest": n_candles,
    }


# --------------------------------------------------------------------------- #
# Per (ticker × method)
# --------------------------------------------------------------------------- #
def run_method(
    ticker: str,
    method: str,
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    fgi_encoder,
    cfg: Mapping[str, Any],
    space: SearchSpace,
    seed: int,
    output_root: Path,
    budget: int | None,
    lookback: int,
    ts: str,
) -> dict[str, Any]:
    """Search + final evaluation for one combination. Returns the CSV row."""
    t0 = time.time()
    opt_cfg = cfg.get("optimization") or {}

    if method == "ga":
        res = run_ga(
            X_2d, y_1d, cfg, space, seed=seed,
            on_generation=_ga_history_writer(output_root, ticker, ts),
            budget=budget,
        )
        best_hp, val_auc, n_evals = res["best_hp"], res["best_fitness"], res["n_evals"]
    elif method == "grid":
        res = run_grid(
            X_2d, y_1d, cfg, space, seed=seed, budget=budget,
            on_eval=_grid_history_writer(output_root, ticker, ts),
        )
        best_hp, val_auc, n_evals = res["best_hp"], res["best_fitness"], res["n_evals"]
    elif method == "manual":
        best_hp = manual_hp_from_cfg(cfg)
        r = evaluate(best_hp, X_2d, y_1d, cfg, seed=seed)
        val_auc = float(r["auc"]) if np.isfinite(r["auc"]) else 0.0
        n_evals = 1
    else:
        raise ValueError(f"Metode tidak dikenal: {method}")

    log.info(f"[{ticker}/{method}] best_hp={best_hp} val_auc={val_auc:.4f} n_evals={n_evals}")

    final = final_evaluate_and_save(
        ticker, method, best_hp, X_2d, y_1d, fgi_encoder,
        cfg, output_root, seed, lookback,
    )

    return {
        "ticker": ticker,
        "method": method,
        "seed": seed,
        "best_hp_json": json.dumps(best_hp),
        "val_auc_search": f"{val_auc:.6f}",
        "wf_auc": f"{final['wf_auc']:.6f}" if np.isfinite(final["wf_auc"]) else "",
        "wf_f1": f"{final['wf_f1']:.6f}" if np.isfinite(final["wf_f1"]) else "",
        "backtest_hit_rate": f"{final['hit_rate']:.6f}" if np.isfinite(final["hit_rate"]) else "",
        "n_signals_issued": _blank_if_nan(final["n_signals_issued"]),
        "n_signals_hold": _blank_if_nan(final["n_signals_hold"]),
        "n_candles_backtest": _blank_if_nan(final["n_candles_backtest"]),
        "n_evals": n_evals,
        "duration_min": round((time.time() - t0) / 60.0, 2),
        "search_epochs": int(opt_cfg.get("search_epochs", 20)),
        "final_splits": int(opt_cfg.get("final_splits", 3)),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Pilot timing
# --------------------------------------------------------------------------- #
def run_pilot(
    cfg: Mapping[str, Any],
    space: SearchSpace,
    ticker: str,
    seed: int,
    n_timing_evals: int = 3,
) -> dict[str, Any]:
    """Time a few real fitness evaluations to estimate full-experiment cost."""
    log.info(f"PILOT: {n_timing_evals} evaluasi pada {ticker} untuk estimasi waktu")
    prepped = prepare_ticker_data(cfg, ticker)
    X_2d, y_1d = prepped["X"], prepped["y"]

    rng = np.random.default_rng(seed)
    durations = []
    for i in range(n_timing_evals):
        hp = space.decode(space.random_individual(rng))
        r = evaluate(hp, X_2d, y_1d, cfg, seed=seed)
        durations.append(r["duration"])
        log.info(
            f"PILOT eval {i + 1}/{n_timing_evals}: auc={r['auc']:.4f} "
            f"durasi={r['duration']:.1f}s hp={hp}"
        )

    ga_cfg = (cfg.get("optimization") or {}).get("ga") or {}
    grid_cfg = (cfg.get("optimization") or {}).get("grid") or {}
    pop = int(ga_cfg.get("population_size", 10))
    gens = int(ga_cfg.get("generations", 6))
    elit = int(ga_cfg.get("elitism", 2))
    ga_evals = pop + gens * (pop - elit)          # upper bound (cache reduces it)
    grid_evals = int(grid_cfg.get("budget", 50))
    n_tickers = len((cfg.get("optimization") or {}).get("tickers", []))

    mean_s = float(np.mean(durations))
    search_s = mean_s * (ga_evals + grid_evals + 1) * n_tickers
    return {
        "mean_eval_seconds": mean_s,
        "ga_evals_est": ga_evals,
        "grid_evals": grid_evals,
        "n_tickers": n_tickers,
        "search_hours_est": search_s / 3600.0,
    }
