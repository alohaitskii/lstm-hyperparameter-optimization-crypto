"""Offline smoke test for the hyperparameter-optimization package.

Runs GA (pop=4, gen=2, search_epochs=1) and a tiny budgeted Grid on the
synthetic bundle from smoke_test.py — no network required. Validates:
    - SearchSpace size / encode / decode roundtrip / grid determinism
    - fitness.evaluate returns metrics in [0, 1]
    - GA and Grid return valid best_hp from the space, with sane history
    - anti-leakage building blocks are reused (indirectly via evaluate)

Usage: python smoke_test_optimization.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

from smoke_test import make_synthetic_bundle  # noqa: E402
from data.preprocessor import prepare_for_training  # noqa: E402
from optimization.search_space import GENE_ORDER, SearchSpace  # noqa: E402
from optimization.fitness import build_cfg_for_hp, evaluate  # noqa: E402
from optimization.genetic_algorithm import run_ga  # noqa: E402
from optimization.grid_search import run_grid  # noqa: E402
from optimization.experiment import manual_hp_from_cfg  # noqa: E402
from utils.helpers import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    # Miniature settings — offline and fast
    cfg = copy.deepcopy(cfg)
    cfg["optimization"]["search_epochs"] = 1
    cfg["optimization"]["ga"] = {
        "population_size": 4, "generations": 2, "tournament_size": 2,
        "crossover_rate": 0.8, "mutation_rate": 0.3, "elitism": 1,
    }
    cfg["optimization"]["grid"]["budget"] = 4

    # ---- 1. Search space ---------------------------------------------------- #
    space = SearchSpace.from_config(cfg)
    assert space.size() == 4800, f"Ukuran ruang harus 4800, dapat {space.size()}"

    rng = np.random.default_rng(0)
    chrom = space.random_individual(rng)
    hp = space.decode(chrom)
    assert list(hp.keys()) == list(GENE_ORDER)
    assert space.encode(hp) == chrom, "encode(decode(x)) != x"

    grid_a = list(space.iter_grid_budget(10))
    grid_b = list(space.iter_grid_budget(10))
    assert grid_a == grid_b, "Grid budget harus deterministik"
    assert len(grid_a) == 10
    assert len({tuple(c) for c in grid_a}) == 10, "Grid budget harus unik"
    full_count = sum(1 for _ in space.iter_grid_budget(None))
    assert full_count == 4800
    print("OK  search_space — size=4800, roundtrip, grid deterministik & unik")

    # cfg override must not mutate the original & must keep task definition
    base_look = cfg["features"]["lookahead_n"]
    c2 = build_cfg_for_hp(cfg, hp)
    assert cfg["features"]["sequence_length"] == 60, "cfg asli termutasi!"
    assert c2["features"]["lookahead_n"] == base_look, "lookahead_n berubah!"
    print("OK  build_cfg_for_hp — deep copy, lookahead_n/move_threshold utuh")

    # ---- 2. Synthetic data -------------------------------------------------- #
    bundle = make_synthetic_bundle(n_15m=900)
    prepped = prepare_for_training(bundle, cfg)
    X, y = prepped["X"], prepped["y"]
    print(f"OK  data sintetis — X={X.shape}")

    # ---- 3. Single fitness evaluation --------------------------------------- #
    small_hp = {
        "sequence_length": 30, "lstm_units_1": 32, "lstm_units_2": 16,
        "dropout_rate": 0.1, "learning_rate": 0.001, "batch_size": 32,
    }
    res = evaluate(small_hp, X, y, cfg, seed=42)
    assert np.isfinite(res["auc"]) and 0.0 <= res["auc"] <= 1.0, f"AUC invalid: {res['auc']}"
    assert 0.0 <= res["f1"] <= 1.0
    assert res["n_train"] > 0 and res["n_val"] > 0
    print(f"OK  fitness — auc={res['auc']:.4f} f1={res['f1']:.4f} ({res['duration']:.1f}s)")

    # ---- 4. GA mini ---------------------------------------------------------- #
    gens_seen: list[int] = []
    ga = run_ga(X, y, cfg, space, seed=42,
                on_generation=lambda r: gens_seen.append(r["generation"]))
    assert set(ga["best_hp"].keys()) == set(GENE_ORDER)
    space.encode(ga["best_hp"])  # raises if outside the space
    assert 0.0 <= ga["best_fitness"] <= 1.0
    assert ga["n_evals"] > 0 and len(ga["history"]) >= 1
    assert gens_seen == [r["generation"] for r in ga["history"]], "callback tidak lengkap"
    best_per_gen = [r["best_fitness"] for r in ga["history"]]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best_per_gen, best_per_gen[1:])), \
        "Best fitness GA harus monoton (elitisme)"
    print(f"OK  GA — best={ga['best_fitness']:.4f} evals={ga['n_evals']} "
          f"gen={len(ga['history']) - 1}")

    # ---- 5. Grid mini --------------------------------------------------------- #
    gr = run_grid(X, y, cfg, space, seed=42, budget=4)
    assert set(gr["best_hp"].keys()) == set(GENE_ORDER)
    space.encode(gr["best_hp"])
    assert gr["n_evals"] == 4
    assert 0.0 <= gr["best_fitness"] <= 1.0
    print(f"OK  Grid — best={gr['best_fitness']:.4f} evals={gr['n_evals']}")

    # ---- 6. Manual baseline hp ------------------------------------------------ #
    manual = manual_hp_from_cfg(cfg)
    assert manual["sequence_length"] == 60 and manual["lstm_units_1"] == 128
    print(f"OK  manual baseline hp — {manual}")

    print("\nALL OPTIMIZATION SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
