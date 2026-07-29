"""Budgeted Grid Search over the same search space as the GA.

The full Cartesian product (4,800 combinations) is unrealistic to train, so
the grid is subsampled deterministically to `budget` evaluations via evenly
spaced positions in the lexicographic enumeration (see
SearchSpace.iter_grid_budget) — deterministic, reproducible, and covering
the whole space rather than only its first corner.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from optimization.fitness import evaluate
from optimization.search_space import SearchSpace
from utils.logger import get_logger

log = get_logger()


def run_grid(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    cfg: Mapping[str, Any],
    space: SearchSpace,
    seed: int = 42,
    budget: int | None = None,
    on_eval: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the (budgeted) grid. Returns {best_hp, best_fitness, history,
    n_evals, duration}."""
    if budget is None:
        budget = int(((cfg.get("optimization") or {}).get("grid") or {}).get("budget", 50))

    total = space.size()
    log.info(f"Grid start: ruang penuh={total} kombinasi, budget={budget}, seed={seed}")

    best_chrom: list[int] | None = None
    best_fit = -np.inf
    history: list[dict[str, Any]] = []
    n_evals = 0
    t0 = time.time()

    for chrom in space.iter_grid_budget(budget):
        hp = space.decode(chrom)
        res = evaluate(hp, X_2d, y_1d, cfg, seed=seed)
        n_evals += 1
        auc = res["auc"]
        fit = float(auc) if np.isfinite(auc) else 0.0

        row = {"eval": n_evals, "fitness": fit, "hp": hp}
        history.append(row)
        if on_eval is not None:
            on_eval(row)

        if fit > best_fit:
            best_fit = fit
            best_chrom = list(chrom)
            log.info(f"Grid eval {n_evals}/{budget}: BARU TERBAIK auc={fit:.4f} hp={hp}")
        else:
            log.info(f"Grid eval {n_evals}/{budget}: auc={fit:.4f}")

    if best_chrom is None:  # budget 0 or empty space — should not happen
        raise RuntimeError("Grid search tidak mengevaluasi kombinasi apa pun")

    return {
        "best_hp": space.decode(best_chrom),
        "best_fitness": float(best_fit),
        "history": history,
        "n_evals": n_evals,
        "duration": time.time() - t0,
    }
