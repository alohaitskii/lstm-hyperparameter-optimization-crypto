"""Walk-forward validation using sklearn's TimeSeriesSplit.

For each fold:
  - Scaler is fit on the training fold ONLY
  - Sequences are built AFTER scaling
  - A fresh model is trained per fold
  - Metrics (accuracy, precision, recall, F1, AUC) recorded per fold
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from data.preprocessor import create_sequences
from model.lstm_model import build_model, train_model
from utils.helpers import ensure_dir, project_root
from utils.logger import get_logger

log = get_logger()
console = Console()


def _metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Standard binary classification metrics."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    # roc_auc requires both classes in y_true
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
    }


def walk_forward_validate(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Run TimeSeriesSplit walk-forward validation.

    Args:
        X_2d: shape (n_rows, n_features) — flat (pre-sequence) feature matrix
        y_1d: shape (n_rows,) — labels aligned to X_2d
        cfg: full config dict (uses cfg['model'] and cfg['features'])

    Returns a DataFrame with per-fold and aggregate metrics.
    """
    model_cfg = cfg["model"]
    feat_cfg = cfg["features"]
    seq_len = feat_cfg["sequence_length"]
    n_splits = model_cfg.get("n_splits_cv", 5)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records: list[dict[str, Any]] = []

    n_features = X_2d.shape[1]

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_2d), start=1):
        log.info(f"=== Fold {fold_idx}/{n_splits} ===")
        log.info(f"Train rows: {len(train_idx)}, Val rows: {len(val_idx)}")

        X_train_raw = X_2d[train_idx]
        X_val_raw = X_2d[val_idx]
        y_train_raw = y_1d[train_idx]
        y_val_raw = y_1d[val_idx]

        # Fit scaler on training fold only
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # Build sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, seq_len)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            log.warning(f"Fold {fold_idx}: not enough rows for sequence_length={seq_len}; skipping")
            continue

        log.info(f"Fold {fold_idx} sequences — train: {X_train_seq.shape}, val: {X_val_seq.shape}")

        # Fresh model per fold
        model = build_model((seq_len, n_features), model_cfg)
        train_model(
            model,
            X_train_seq,
            y_train_seq,
            X_val_seq,
            y_val_seq,
            model_cfg,
            save_path=None,
        )

        y_val_prob = model.predict(X_val_seq, verbose=0).flatten()
        m = _metrics(y_val_seq, y_val_prob)
        m["fold"] = fold_idx
        m["train_rows"] = len(train_idx)
        m["val_rows"] = len(val_idx)
        fold_records.append(m)

    if not fold_records:
        log.error("No folds completed — walk-forward validation produced no metrics.")
        return pd.DataFrame()

    df = pd.DataFrame(fold_records)
    # Aggregate row
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    agg = {c: [df[c].mean(), df[c].std()] for c in metric_cols}
    summary = pd.DataFrame(
        {
            "fold": ["MEAN", "STD"],
            **{c: agg[c] for c in metric_cols},
        }
    )
    out = pd.concat([df[["fold", *metric_cols]], summary], ignore_index=True)
    return out


def print_report(report: pd.DataFrame) -> None:
    """Render the walk-forward report as a rich table."""
    if report.empty:
        console.print("[red]No fold results to display.[/red]")
        return

    table = Table(title="Walk-Forward Validation Report", show_lines=True)
    for col in ["fold", "accuracy", "precision", "recall", "f1", "auc"]:
        table.add_column(col.upper())

    for _, row in report.iterrows():
        cells = [str(row["fold"])]
        for c in ["accuracy", "precision", "recall", "f1", "auc"]:
            v = row[c]
            cells.append(f"{v:.4f}" if pd.notna(v) else "—")
        table.add_row(*cells)

    console.print(table)


def save_report(report: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    ensure_dir(p.parent)
    report.to_csv(p, index=False)
    log.info(f"Validation report saved → {p}")
