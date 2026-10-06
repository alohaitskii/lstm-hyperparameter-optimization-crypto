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

from data.fetcher import assert_snapshot_bundle, fetch_all
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
    "wf_auc", "wf_f1", "wf_precision", "wf_recall", "backtest_hit_rate",
    # Hit-rate = benar / sinyal diterbitkan; HOLD tidak masuk penyebut, jadi
    # jumlah sinyal wajib dicatat agar hit-rate antarmetode bisa dibandingkan
    # secara adil (model konservatif otomatis terlihat lebih baik tanpa ini).
    "n_signals_issued", "n_signals_hold", "n_candles_backtest",
    # duration_min = pencarian + evaluasi akhir; search_duration_min = pencarian
    # saja (pembanding efisiensi GA vs Grid yang adil)
    "n_evals", "duration_min", "search_duration_min",
    "search_epochs", "final_splits", "timestamp",
]


# One row per (ticker, method, seed, fold) of every final evaluation
FOLD_FIELDS = [
    "ticker", "method", "seed", "fold",
    "accuracy", "precision", "recall", "f1", "auc",
    "n_train_seq", "n_val_seq", "pct_label1_train", "pct_label1_val",
    "timestamp",
]


def folds_csv_path(output_root: Path) -> Path:
    """Stable append-only file with the per-fold walk-forward metrics."""
    return output_root / "logs" / "optimization_folds.csv"


def append_fold_rows(
    path: Path, ticker: str, method: str, seed: int,
    folds: list[dict[str, Any]], timestamp: str,
) -> None:
    """Append the fold rows of one final evaluation (checkpoint granularity)."""
    def f6(v: Any) -> str:
        v = float(v)
        return f"{v:.6f}" if np.isfinite(v) else ""

    ensure_dir(path.parent)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FOLD_FIELDS)
        if new_file:
            w.writeheader()
        for r in folds:
            w.writerow({
                "ticker": ticker, "method": method, "seed": seed,
                "fold": int(r["fold"]),
                **{k: f6(r[k]) for k in ("accuracy", "precision", "recall", "f1", "auc")},
                "n_train_seq": int(r["n_train_seq"]),
                "n_val_seq": int(r["n_val_seq"]),
                "pct_label1_train": f"{100.0 * float(r['pos_rate_train']):.4f}",
                "pct_label1_val": f"{100.0 * float(r['pos_rate_val']):.4f}",
                "timestamp": timestamp,
            })


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


# Per-individual GA trace and per-tournament log (Langkah 2a)
GA_TRACE_FIELDS = [
    "generation", "slot", "rank", "origin", "chromosome", "hp_json", "fitness",
    "from_cache", "eval_count_kumulatif", "elite_from_rank",
    "parent1_chromosome", "parent2_chromosome", "crossover_done",
    "crossover_mask", "mutated_genes", "waktu_detik",
]
GA_TOURNAMENT_FIELDS = ["generation", "offspring_slot", "parent_ke", "peserta", "pemenang"]


def _csv_appender(path: Path, fields: list[str]):
    """Callback that appends one dict row immediately (header on first row)."""
    ensure_dir(path.parent)

    def write(row: dict[str, Any]) -> None:
        new_file = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            w.writerow(row)

    return write


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
                w.writerow(["eval", "fitness", "best_so_far", "hp_json", "waktu_detik"])
            w.writerow(
                [
                    row["eval"],
                    f"{row['fitness']:.6f}",
                    f"{best_so_far:.6f}",
                    json.dumps(row["hp"]),
                    row.get("waktu_detik", ""),
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
    assert_snapshot_bundle(cfg, ticker, bundle, "persiapan data (pencarian, walk-forward)")
    prepped = prepare_for_training(bundle, cfg)
    # Kept for the backtest, so no fetch_all happens again in the middle of a run
    prepped["bundle"] = bundle
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
    persist_artifacts: bool = True,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Full walk-forward on best_hp, train + save winner, backtest hit-rate.

    persist_artifacts=False trains and backtests as usual but does NOT write
    model.keras / scaler.pkl / fgi_encoder.pkl. Used when repeating the same
    combination on extra seeds: the thesis needs the spread of metrics across
    seeds, not a copy of every model, so the first seed's artifacts stay in
    place and storage does not balloon.
    """
    import tensorflow as tf

    from main import run_backtest  # deferred: main applies its own patches

    # Fail fast (before minutes of training): in cache-only mode the backtest
    # must reuse the already-loaded bundle instead of calling fetch_all again.
    if (cfg.get("data") or {}).get("cache_only") and bundle is None:
        raise RuntimeError(
            "cache-only: final_evaluate_and_save membutuhkan bundle yang sudah dimuat "
            "(backtest tidak boleh memanggil fetch_all di tengah run)"
        )

    opt_cfg = cfg.get("optimization") or {}
    final_splits = int(opt_cfg.get("final_splits", 3))
    final_epochs = opt_cfg.get("final_epochs")  # None → full model.epochs

    eval_cfg = build_cfg_for_hp(cfg, best_hp, epochs=final_epochs, n_splits=final_splits)
    seq_len = int(best_hp["sequence_length"])

    # --- 1. Reported metrics: full walk-forward validation ----------------- #
    log.info(f"[{ticker}/{method}] Walk-forward final ({final_splits} fold)...")
    report = walk_forward_validate(X_2d, y_1d, eval_cfg)
    # validator.py menghitung accuracy/precision/recall/f1/auc per fold dan
    # merangkumnya di baris MEAN — ambil keempat yang dilaporkan di skripsi.
    wf_auc = wf_f1 = wf_precision = wf_recall = float("nan")
    if not report.empty:
        mean_row = report[report["fold"] == "MEAN"]
        if len(mean_row):
            wf_auc = float(mean_row["auc"].iloc[0])
            wf_f1 = float(mean_row["f1"].iloc[0])
            wf_precision = float(mean_row["precision"].iloc[0])
            wf_recall = float(mean_row["recall"].iloc[0])
    fold_rows: list[dict[str, Any]] = (
        report[~report["fold"].isin(["MEAN", "STD"])].to_dict("records")
        if not report.empty else []
    )

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
    if persist_artifacts:
        out_dir = output_root / "model" / "optimized" / ticker / method
        save_artifacts(model, scaler, fgi_encoder, ticker, out_dir=out_dir)
        with open(out_dir / "hyperparams.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hp": dict(best_hp),
                    "wf_auc": wf_auc,
                    "wf_f1": wf_f1,
                    "wf_precision": wf_precision,
                    "wf_recall": wf_recall,
                    "seed": seed,
                },
                f, indent=2,
            )
    else:
        log.info(
            f"[{ticker}/{method}] seed {seed}: artefak model dilewati "
            f"(hanya metrik yang dicatat)"
        )

    # --- 4. Backtest hit-rate (reuse main.run_backtest, preloaded model) --- #
    log.info(f"[{ticker}/{method}] Backtest hit-rate (lookback={lookback})...")
    if bundle is not None:
        assert_snapshot_bundle(cfg, ticker, bundle, "backtest")
    bt = run_backtest(
        eval_cfg, lookback=lookback, ticker=ticker,
        preloaded=(model, scaler, fgi_encoder), tag=method, bundle=bundle,
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
        "wf_precision": wf_precision,
        "wf_recall": wf_recall,
        "folds": fold_rows,
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
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Search + final evaluation for one combination. Returns the CSV row."""
    t0 = time.time()
    opt_cfg = cfg.get("optimization") or {}

    if method == "ga":
        safe = ticker_to_safe_name(ticker)
        logs = output_root / "logs"
        res = run_ga(
            X_2d, y_1d, cfg, space, seed=seed,
            on_generation=_ga_history_writer(output_root, ticker, ts),
            budget=budget,
            on_individual=_csv_appender(logs / f"ga_trace_{safe}_{ts}.csv", GA_TRACE_FIELDS),
            on_tournament=_csv_appender(
                logs / f"ga_tournament_{safe}_{ts}.csv", GA_TOURNAMENT_FIELDS
            ),
        )
        best_hp, val_auc, n_evals = res["best_hp"], res["best_fitness"], res["n_evals"]
        search_s = float(res["duration"])
    elif method == "grid":
        res = run_grid(
            X_2d, y_1d, cfg, space, seed=seed, budget=budget,
            on_eval=_grid_history_writer(output_root, ticker, ts),
        )
        best_hp, val_auc, n_evals = res["best_hp"], res["best_fitness"], res["n_evals"]
        search_s = float(res["duration"])
    elif method == "manual":
        best_hp = manual_hp_from_cfg(cfg)
        r = evaluate(best_hp, X_2d, y_1d, cfg, seed=seed)
        search_s = float(r["duration"])
        val_auc = float(r["auc"]) if np.isfinite(r["auc"]) else 0.0
        n_evals = 1
    else:
        raise ValueError(f"Metode tidak dikenal: {method}")

    log.info(f"[{ticker}/{method}] best_hp={best_hp} val_auc={val_auc:.4f} n_evals={n_evals}")

    final = final_evaluate_and_save(
        ticker, method, best_hp, X_2d, y_1d, fgi_encoder,
        cfg, output_root, seed, lookback, bundle=bundle,
    )

    return {
        "ticker": ticker,
        "method": method,
        "seed": seed,
        "best_hp_json": json.dumps(best_hp),
        "val_auc_search": f"{val_auc:.6f}",
        "wf_auc": f"{final['wf_auc']:.6f}" if np.isfinite(final["wf_auc"]) else "",
        "wf_f1": f"{final['wf_f1']:.6f}" if np.isfinite(final["wf_f1"]) else "",
        "wf_precision": f"{final['wf_precision']:.6f}" if np.isfinite(final["wf_precision"]) else "",
        "wf_recall": f"{final['wf_recall']:.6f}" if np.isfinite(final["wf_recall"]) else "",
        "backtest_hit_rate": f"{final['hit_rate']:.6f}" if np.isfinite(final["hit_rate"]) else "",
        "n_signals_issued": _blank_if_nan(final["n_signals_issued"]),
        "n_signals_hold": _blank_if_nan(final["n_signals_hold"]),
        "n_candles_backtest": _blank_if_nan(final["n_candles_backtest"]),
        "n_evals": n_evals,
        "duration_min": round((time.time() - t0) / 60.0, 2),
        "search_duration_min": round(search_s / 60.0, 2),
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
