"""LSTM model: build, train, save, load.

Architecture matches the spec:
    LSTM(128, return_sequences=True) -> Dropout(0.2) -> BatchNorm
    LSTM(64, return_sequences=False) -> Dropout(0.2) -> BatchNorm
    Dense(32, relu) -> Dropout(0.1) -> Dense(1, sigmoid)

Multi-pair storage: each ticker owns model/saved/{TICKER}/ containing
model.keras + scaler.pkl + fgi_encoder.pkl (save_artifacts/load_artifacts).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Suppress TF info logs *before* importing
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (  # noqa: E402
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential, load_model  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

from utils.helpers import ensure_dir, project_root
from utils.logger import get_logger
# Re-exported for spec compliance — defined in ticker_utils so path checks
# don't require importing TensorFlow.
from utils.ticker_utils import get_model_dir, is_trained  # noqa: F401

log = get_logger()


def build_model(input_shape: tuple[int, int], cfg: Mapping[str, Any]) -> tf.keras.Model:
    """Build and compile the LSTM binary classifier."""
    lstm_units = cfg.get("lstm_units", [128, 64])
    dense_units = cfg.get("dense_units", 32)
    dropout = cfg.get("dropout_rate", 0.2)
    lr = cfg.get("learning_rate", 0.001)

    model = Sequential(
        [
            # Keras 3 style: explicit Input layer instead of input_shape kwarg
            Input(shape=input_shape),
            LSTM(lstm_units[0], return_sequences=True),
            Dropout(dropout),
            BatchNormalization(),
            LSTM(lstm_units[1], return_sequences=False),
            Dropout(dropout),
            BatchNormalization(),
            Dense(dense_units, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = project_root() / path
    return path


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Mapping[str, Any],
    save_path: str | Path | None = None,
    verbose: int = 2,
) -> tf.keras.callbacks.History:
    """Fit the model with the standard callback stack.

    verbose=0 silences per-epoch output — used by hyperparameter search
    where dozens of short trainings would flood the console.
    """
    cb_verbose = 1 if verbose else 0
    callbacks: list = []
    callbacks.append(
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=cfg.get("patience_early_stop", 10),
            restore_best_weights=True,
            verbose=cb_verbose,
        )
    )
    callbacks.append(
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=cb_verbose)
    )

    if save_path is not None:
        sp = _resolve_path(save_path)
        ensure_dir(sp.parent)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(sp),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=0,
            )
        )

    # Balanced class weights: with a 0.3% move threshold the positive class is
    # usually the minority — without weighting the model collapses to "always 0".
    class_weight = None
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    if len(classes) == 2 and counts.min() > 0:
        total = counts.sum()
        class_weight = {
            int(c): float(total / (2.0 * cnt)) for c, cnt in zip(classes, counts)
        }
        log.info(f"Class weights (balanced): {class_weight}")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.get("epochs", 100),
        batch_size=cfg.get("batch_size", 32),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )
    return history


def save(model: tf.keras.Model, path: str | Path) -> None:
    sp = _resolve_path(path)
    ensure_dir(sp.parent)
    model.save(sp)
    log.info(f"Model saved → {sp}")


def load(path: str | Path) -> tf.keras.Model:
    sp = _resolve_path(path)
    if not sp.exists():
        raise FileNotFoundError(f"Model not found at {sp}")
    log.info(f"Loading model from {sp}")
    return load_model(sp)


# --------------------------------------------------------------------------- #
# Per-ticker artifact storage (multi-pair)
# --------------------------------------------------------------------------- #
def save_artifacts(
    model: tf.keras.Model, scaler, encoder, ticker: str,
    out_dir: str | Path | None = None,
) -> None:
    """Persist model + scaler + FGI encoder.

    Default target is model/saved/{TICKER}/ (baseline). Pass `out_dir` to
    write elsewhere (e.g. model/optimized/{TICKER}/{method}/) WITHOUT
    touching the baseline artifacts.
    """
    d = Path(out_dir) if out_dir is not None else get_model_dir(ticker)
    ensure_dir(d)
    model.save(d / "model.keras")
    with open(d / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(d / "fgi_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    log.info(f"Artifacts saved → {d}")


def load_artifacts(ticker: str):
    """Load (model, scaler, encoder) for a ticker, or (None, None, None) if untrained."""
    d = get_model_dir(ticker)
    if not (d / "model.keras").exists():
        return None, None, None
    model = load_model(d / "model.keras")
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(d / "fgi_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    log.info(f"Artifacts loaded ← {d}")
    return model, scaler, encoder
