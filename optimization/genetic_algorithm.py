"""Genetic Algorithm for LSTM hyperparameter search — implemented from
scratch (no external GA library) so every operator is transparent and
defensible in a thesis defense.

Components:
    - Population init : uniform random chromosomes
    - Fitness         : optimization.fitness.evaluate (validation AUC)
    - Selection       : tournament (size k, with replacement)
    - Crossover       : uniform (per-gene coin flip), rate from config
    - Mutation        : per-gene random-reset to a DIFFERENT index
    - Elitism         : top-N copied unchanged into the next generation
    - Caching         : fitness memoized per chromosome — identical
                        configurations are never trained twice

Degenerate fitness (NaN AUC) is treated as 0.0 in selection so broken
configurations die out naturally (logged).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from optimization.fitness import evaluate
from optimization.search_space import GENE_ORDER, SearchSpace
from utils.logger import get_logger

log = get_logger()


def _tournament(
    population: list[list[int]],
    fitnesses: list[float],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pick k random contestants, return a copy of the fittest."""
    idx = rng.integers(0, len(population), size=min(k, len(population)))
    best = max(idx, key=lambda i: fitnesses[i])
    return list(population[best])


def _uniform_crossover(
    p1: list[int], p2: list[int], rng: np.random.Generator
) -> list[int]:
    """Per-gene coin flip between the two parents."""
    return [p1[g] if rng.random() < 0.5 else p2[g] for g in range(len(p1))]


def _mutate(
    chrom: list[int],
    space: SearchSpace,
    rate: float,
    rng: np.random.Generator,
) -> list[int]:
    """Random-reset mutation: each gene flips to a DIFFERENT valid index."""
    out = list(chrom)
    for g, gene in enumerate(GENE_ORDER):
        if rng.random() < rate:
            n = len(space.choices(gene))
            if n > 1:
                new = int(rng.integers(0, n - 1))
                if new >= out[g]:
                    new += 1  # skip current index → guaranteed change
                out[g] = new
    return out


def run_ga(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    cfg: Mapping[str, Any],
    space: SearchSpace,
    seed: int = 42,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Run the GA. Returns {best_hp, best_fitness, history, n_evals, duration}.

    `on_generation(row)` is called after every generation with the history
    row — the experiment layer uses it to write ga_history CSV incrementally
    (Colab-safe: progress survives a dropped session).

    `budget` caps the number of REAL fitness evaluations (cache misses);
    checked at generation boundaries, so the cap is approximate at
    generation granularity.
    """
    ga_cfg = (cfg.get("optimization") or {}).get("ga") or {}
    pop_size = int(ga_cfg.get("population_size", 10))
    generations = int(ga_cfg.get("generations", 6))
    tournament_size = int(ga_cfg.get("tournament_size", 3))
    crossover_rate = float(ga_cfg.get("crossover_rate", 0.8))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.15))
    elitism = int(ga_cfg.get("elitism", 2))

    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, ...], float] = {}
    n_evals = 0
    t0 = time.time()

    def fitness_of(chrom: list[int]) -> float:
        nonlocal n_evals
        key = tuple(chrom)
        if key not in cache:
            res = evaluate(space.decode(chrom), X_2d, y_1d, cfg, seed=seed)
            n_evals += 1
            auc = res["auc"]
            if not np.isfinite(auc):
                log.warning(f"GA: fitness degenerate utk {space.decode(chrom)} → 0.0")
                auc = 0.0
            cache[key] = float(auc)
        return cache[key]

    log.info(
        f"GA start: pop={pop_size}, gen={generations}, tournament={tournament_size}, "
        f"cx={crossover_rate}, mut={mutation_rate}, elit={elitism}, seed={seed}"
    )

    population = [space.random_individual(rng) for _ in range(pop_size)]
    history: list[dict[str, Any]] = []

    for gen in range(generations + 1):  # gen 0 = initial population
        fitnesses = [fitness_of(c) for c in population]

        order = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        row = {
            "generation": gen,
            "best_fitness": fitnesses[0],
            "mean_fitness": float(np.mean(fitnesses)),
            "best_hp": space.decode(population[0]),
            "n_evals": n_evals,
        }
        history.append(row)
        log.info(
            f"GA gen {gen}/{generations}: best={row['best_fitness']:.4f} "
            f"mean={row['mean_fitness']:.4f} evals={n_evals}"
        )
        if on_generation is not None:
            on_generation(row)

        if gen == generations:
            break
        if budget is not None and n_evals >= budget:
            log.info(f"GA berhenti: budget evaluasi tercapai ({n_evals}/{budget})")
            break

        # Breed the next generation
        next_pop = [list(c) for c in population[:elitism]]
        while len(next_pop) < pop_size:
            p1 = _tournament(population, fitnesses, tournament_size, rng)
            p2 = _tournament(population, fitnesses, tournament_size, rng)
            child = _uniform_crossover(p1, p2, rng) if rng.random() < crossover_rate else p1
            child = _mutate(child, space, mutation_rate, rng)
            next_pop.append(child)
        population = next_pop

    return {
        "best_hp": space.decode(population[0]),
        "best_fitness": float(history[-1]["best_fitness"]),
        "history": history,
        "n_evals": n_evals,
        "duration": time.time() - t0,
    }
