"""Fitness evaluation for hyperparameter search (GA & Grid share this).

Protocol (anti-leakage preserved, identical for every candidate):
    split_chronological  → single chronological train/val split (fast)
    fit_scaler(train)    → scaler fit on TRAIN ONLY, then transform both
    create_sequences     → sequences built AFTER scaling, seq_len from hp
    build_model + train_model with search_epochs (early stopping active)
    AUC + F1 on validation predictions

Fitness during search = validation AUC on this single split. The final
(reported) metrics come from the full walk_forward_validate elsewhere —
searching on one split and reporting on walk-forward prevents overfitting
the search to a single validation window.

Memory hygiene: clear_session() after every evaluation so long searches on
CPU/Colab do not accumulate graph state.
"""
from __future__ import annotations

import copy
import gc
import time
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from data.preprocessor import create_sequences, fit_scaler, split_chronological
from model.lstm_model import build_model, train_model
from utils.helpers import set_global_seed
from utils.logger import get_logger

log = get_logger()


def build_cfg_for_hp(
    cfg: Mapping[str, Any],
    hp: Mapping[str, Any],
    epochs: int | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    """Deep-copy cfg with hyperparameters applied (original never mutated).

    lookahead_n / move_threshold are intentionally untouched — task
    definition, not hyperparameters.
    """
    c = copy.deepcopy(dict(cfg))
    c["features"]["sequence_length"] = int(hp["sequence_length"])
    m = c["model"]
    m["lstm_units"] = [int(hp["lstm_units_1"]), int(hp["lstm_units_2"])]
    m["dropout_rate"] = float(hp["dropout_rate"])
    m["learning_rate"] = float(hp["learning_rate"])
    m["batch_size"] = int(hp["batch_size"])
    if epochs is not None:
        m["epochs"] = int(epochs)
    if n_splits is not None:
        m["n_splits_cv"] = int(n_splits)
    return c


def evaluate(
    hp: Mapping[str, Any],
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    cfg: Mapping[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Train once on the chronological split and score the validation window.

    Returns {auc, f1, n_train, n_val, duration}. Degenerate cases (too few
    sequences, or single-class validation set → undefined AUC) return
    auc=NaN so the caller can penalize them.
    """
    import tensorflow as tf

    t0 = time.time()
    set_global_seed(seed)

    seq_len = int(hp["sequence_length"])
    n_features = X_2d.shape[1]

    train_sl, val_sl, _test_sl = split_chronological(len(X_2d))
    X_train_raw, y_train = X_2d[train_sl], y_1d[train_sl]
    X_val_raw, y_val = X_2d[val_sl], y_1d[val_sl]

    # Anti-leakage: scaler fit on train only
    scaler = fit_scaler(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        log.warning(f"Degenerate: seq_len={seq_len} menyisakan 0 sequence — fitness NaN")
        return {
            "auc": float("nan"), "f1": float("nan"),
            "n_train": 0, "n_val": 0, "duration": time.time() - t0,
        }

    search_epochs = int((cfg.get("optimization") or {}).get("search_epochs", 20))
    cfg_hp = build_cfg_for_hp(cfg, hp, epochs=search_epochs)

    model = build_model((seq_len, n_features), cfg_hp["model"])
    train_model(
        model,
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        cfg_hp["model"],
        save_path=None,
        verbose=0,
    )

    probs = model.predict(X_val_seq, verbose=0).flatten()
    try:
        auc = float(roc_auc_score(y_val_seq, probs))
    except ValueError:  # validation window contains a single class
        auc = float("nan")
    f1 = float(f1_score(y_val_seq, (probs >= 0.5).astype(int), zero_division=0))

    # Memory hygiene for long searches
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "auc": auc,
        "f1": f1,
        "n_train": int(len(X_train_seq)),
        "n_val": int(len(X_val_seq)),
        "duration": time.time() - t0,
    }
