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

Tracing: optional callbacks receive one row per individual per generation
(origin, parents, crossover mask, mutated genes, fitness, cache hit, time)
and one row per tournament (participants with fitness, winner). Tracing only
READS values the operators already produce: every random draw happens in the
same order and number as without tracing, so a traced run is bit-identical
to an untraced one.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

import numpy as np

from optimization.fitness import evaluate
from optimization.search_space import GENE_ORDER, SearchSpace
from utils.logger import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Operators. The *_detail variants perform EXACTLY the same rng calls as the
# originals and additionally return what happened, for the trace.
# --------------------------------------------------------------------------- #
def _tournament_detail(
    population: list[list[int]],
    fitnesses: list[float],
    k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], int]:
    """Pick k random contestants (with replacement); return (winner copy,
    contestant indices, winner index). Indices refer to the population as
    passed in — i.e. fitness ranks, since the population is sorted."""
    idx = rng.integers(0, len(population), size=min(k, len(population)))
    best = max(idx, key=lambda i: fitnesses[i])
    return list(population[best]), [int(i) for i in idx], int(best)


def _uniform_crossover_detail(
    p1: list[int], p2: list[int], rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Per-gene coin flip; returns (child, mask) with mask[g] = 1 or 2 (parent)."""
    mask = [1 if rng.random() < 0.5 else 2 for _ in range(len(p1))]
    return [p1[g] if m == 1 else p2[g] for g, m in enumerate(mask)], mask


def _mutate_detail(
    chrom: list[int],
    space: SearchSpace,
    rate: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Random-reset mutation: each gene flips to a DIFFERENT valid index.
    Returns (child, [(gene position, old index, new index), ...])."""
    out = list(chrom)
    changes: list[tuple[int, int, int]] = []
    for g, gene in enumerate(GENE_ORDER):
        if rng.random() < rate:
            n = len(space.choices(gene))
            if n > 1:
                new = int(rng.integers(0, n - 1))
                if new >= out[g]:
                    new += 1  # skip current index → guaranteed change
                changes.append((g, out[g], new))
                out[g] = new
    return out, changes


def _tournament(population, fitnesses, k, rng) -> list[int]:
    return _tournament_detail(population, fitnesses, k, rng)[0]


def _uniform_crossover(p1, p2, rng) -> list[int]:
    return _uniform_crossover_detail(p1, p2, rng)[0]


def _mutate(chrom, space, rate, rng) -> list[int]:
    return _mutate_detail(chrom, space, rate, rng)[0]


# --------------------------------------------------------------------------- #
def run_ga(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    cfg: Mapping[str, Any],
    space: SearchSpace,
    seed: int = 42,
    on_generation: Callable[[dict[str, Any]], None] | None = None,
    budget: int | None = None,
    on_individual: Callable[[dict[str, Any]], None] | None = None,
    on_tournament: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the GA. Returns {best_hp, best_fitness, history, n_evals, duration}.

    `on_generation(row)` is called after every generation with the history
    row — the experiment layer uses it to write ga_history CSV incrementally
    (Colab-safe: progress survives a dropped session).

    `on_individual(row)` / `on_tournament(row)` receive the per-individual and
    per-tournament trace (see module docstring); rows of a generation are
    emitted as soon as that generation is evaluated.

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

    def fitness_of(chrom: list[int]) -> tuple[float, bool, float]:
        """(fitness, from_cache, seconds spent). Cache semantics unchanged."""
        nonlocal n_evals
        key = tuple(chrom)
        if key in cache:
            return cache[key], True, 0.0
        t_eval = time.time()
        res = evaluate(space.decode(chrom), X_2d, y_1d, cfg, seed=seed)
        n_evals += 1
        auc = res["auc"]
        if not np.isfinite(auc):
            log.warning(f"GA: fitness degenerate utk {space.decode(chrom)} → 0.0")
            auc = 0.0
        cache[key] = float(auc)
        return cache[key], False, time.time() - t_eval

    log.info(
        f"GA start: pop={pop_size}, gen={generations}, tournament={tournament_size}, "
        f"cx={crossover_rate}, mut={mutation_rate}, elit={elitism}, seed={seed}"
    )

    population = [space.random_individual(rng) for _ in range(pop_size)]
    # Provenance of each slot of the population about to be evaluated
    meta: list[dict[str, Any]] = [{"origin": "init"} for _ in range(pop_size)]
    history: list[dict[str, Any]] = []

    for gen in range(generations + 1):  # gen 0 = initial population
        fitnesses: list[float] = []
        eval_info: list[tuple[bool, int, float]] = []
        for c in population:
            fit, cached, secs = fitness_of(c)
            fitnesses.append(fit)
            eval_info.append((cached, n_evals, secs))

        order = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)

        if on_individual is not None:
            rank_of = {slot: r for r, slot in enumerate(order)}
            for slot, chrom in enumerate(population):
                m = meta[slot]
                cached, cum, secs = eval_info[slot]
                on_individual({
                    "generation": gen,
                    "slot": slot,
                    "rank": rank_of[slot],
                    "origin": m["origin"],
                    "chromosome": json.dumps(chrom),
                    "hp_json": json.dumps(space.decode(chrom)),
                    "fitness": fitnesses[slot],
                    "from_cache": cached,
                    "eval_count_kumulatif": cum,
                    "elite_from_rank": m.get("elite_from_rank", ""),
                    "parent1_chromosome": json.dumps(m["p1"]) if "p1" in m else "",
                    "parent2_chromosome": json.dumps(m["p2"]) if "p2" in m else "",
                    "crossover_done": m.get("cx", ""),
                    "crossover_mask": "".join(map(str, m["mask"])) if m.get("mask") else "",
                    "mutated_genes": "; ".join(
                        f"{GENE_ORDER[g]}: {old}->{new}" for g, old, new in m.get("mut", [])
                    ),
                    "waktu_detik": round(secs, 2),
                })

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

        # Breed the next generation (rng call order identical to the untraced GA:
        # tournament p1, tournament p2, crossover decision, [crossover genes],
        # mutation genes)
        next_pop = [list(c) for c in population[:elitism]]
        next_meta: list[dict[str, Any]] = [
            {"origin": "elite", "elite_from_rank": r} for r in range(len(next_pop))
        ]
        while len(next_pop) < pop_size:
            slot = len(next_pop)
            p1, idx1, w1 = _tournament_detail(population, fitnesses, tournament_size, rng)
            p2, idx2, w2 = _tournament_detail(population, fitnesses, tournament_size, rng)
            do_cx = rng.random() < crossover_rate
            if do_cx:
                child, mask = _uniform_crossover_detail(p1, p2, rng)
            else:
                child, mask = p1, None
            child, changes = _mutate_detail(child, space, mutation_rate, rng)

            if on_tournament is not None:
                for parent_ke, idx, w in ((1, idx1, w1), (2, idx2, w2)):
                    on_tournament({
                        "generation": gen + 1,
                        "offspring_slot": slot,
                        "parent_ke": parent_ke,
                        "peserta": json.dumps([
                            {"rank": i, "chromosome": population[i], "fitness": fitnesses[i]}
                            for i in idx
                        ]),
                        "pemenang": json.dumps(
                            {"rank": w, "chromosome": population[w], "fitness": fitnesses[w]}
                        ),
                    })
            next_pop.append(child)
            next_meta.append({"origin": "offspring", "p1": p1, "p2": p2,
                              "cx": bool(do_cx), "mask": mask, "mut": changes})
        population = next_pop
        meta = next_meta

    return {
        "best_hp": space.decode(population[0]),
        "best_fitness": float(history[-1]["best_fitness"]),
        "history": history,
        "n_evals": n_evals,
        "duration": time.time() - t0,
    }
