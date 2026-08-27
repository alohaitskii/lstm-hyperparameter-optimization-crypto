"""Hyperparameter search space: chromosome encoding and grid enumeration.

Chromosome = list of integer indices, one per gene, in GENE_ORDER. Index i
selects the i-th candidate value from the corresponding list in the search
space. GA and Grid Search share this single definition so both methods search
an IDENTICAL space (methodological requirement for the thesis).

Deliberately excluded from the space: lookahead_n and move_threshold — they
define the prediction TASK, not the model. Changing them would change the
label distribution and make method comparisons invalid.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Iterator, Mapping

import numpy as np

from utils.logger import get_logger

log = get_logger()

# Gene order is a fixed contract — CSV logs and chromosomes depend on it.
GENE_ORDER: tuple[str, ...] = (
    "sequence_length",
    "lstm_units_1",
    "lstm_units_2",
    "dropout_rate",
    "learning_rate",
    "batch_size",
)

# Fallback when config.yaml lacks optimization.search_space
DEFAULT_SPACE: dict[str, list] = {
    "sequence_length": [30, 45, 60, 90, 120],
    "lstm_units_1": [32, 64, 96, 128],
    "lstm_units_2": [16, 32, 48, 64],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4],
    "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
    "batch_size": [16, 32, 64],
}


def _coprime_stride(total: int, budget: int) -> int:
    """Langkah terkecil >= total//budget yang relatif prima terhadap total.

    Menjamin i*stride mod total bersifat injektif untuk i < total, sehingga
    tepat `budget` posisi unik terpilih DAN setiap gen ikut berputar
    (langkah yang habis dibagi ukuran gen terdalam akan membekukan gen itu).
    """
    s = max(1, total // budget)
    while math.gcd(s, total) != 1:
        s += 1
    return s


class SearchSpace:
    """Immutable view over the candidate lists, with encode/decode helpers."""

    def __init__(self, space: Mapping[str, list]) -> None:
        missing = [g for g in GENE_ORDER if g not in space or not space[g]]
        if missing:
            raise ValueError(f"Search space tidak lengkap, gen kosong: {missing}")
        self._space: dict[str, list] = {g: list(space[g]) for g in GENE_ORDER}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SearchSpace":
        raw = (cfg.get("optimization") or {}).get("search_space") or DEFAULT_SPACE
        return cls(raw)

    # ------------------------------------------------------------------ #
    @property
    def n_genes(self) -> int:
        return len(GENE_ORDER)

    def choices(self, gene: str) -> list:
        return self._space[gene]

    def size(self) -> int:
        """Total number of combinations in the full Cartesian product."""
        return math.prod(len(v) for v in self._space.values())

    # ------------------------------------------------------------------ #
    def decode(self, chromosome: list[int]) -> dict[str, Any]:
        """List of indices → hyperparameter dict."""
        if len(chromosome) != self.n_genes:
            raise ValueError(f"Kromosom harus {self.n_genes} gen, dapat {len(chromosome)}")
        return {
            gene: self._space[gene][idx]
            for gene, idx in zip(GENE_ORDER, chromosome)
        }

    def encode(self, hp: Mapping[str, Any]) -> list[int]:
        """Hyperparameter dict → list of indices (ValueError if value unknown)."""
        chrom = []
        for gene in GENE_ORDER:
            values = self._space[gene]
            if hp[gene] not in values:
                raise ValueError(f"{gene}={hp[gene]} tidak ada dalam ruang pencarian {values}")
            chrom.append(values.index(hp[gene]))
        return chrom

    # ------------------------------------------------------------------ #
    def random_individual(self, rng: np.random.Generator) -> list[int]:
        """Uniform random chromosome with valid indices."""
        return [int(rng.integers(0, len(self._space[g]))) for g in GENE_ORDER]

    def iter_grid(self) -> Iterator[list[int]]:
        """Lazily enumerate the full Cartesian product (never materialized)."""
        ranges = [range(len(self._space[g])) for g in GENE_ORDER]
        for combo in itertools.product(*ranges):
            yield list(combo)

    def iter_grid_budget(self, budget: int | None) -> Iterator[list[int]]:
        """Deterministic, space-covering subsample of the grid.

        When budget >= size, yields the full grid. Otherwise it walks the
        lexicographic enumeration in steps of a stride COPRIME to the total
        (see _coprime_stride): the positions (i * stride) % total are all
        distinct, and because the stride shares no factor with the space
        size, EVERY gene keeps rotating through all of its candidate values.

        A naive evenly-spaced stride (total // budget) is wrong here: with
        4800 combinations and budget 50 the step is exactly 96 — a multiple
        of the innermost gene's cardinality (batch_size, 3 values) — which
        freezes that gene at index 0. Grid Search would then explore only a
        third of the space and its comparison against the GA would be unfair.
        """
        total = self.size()
        if budget is None or budget >= total:
            yield from self.iter_grid()
            return

        stride = _coprime_stride(total, budget)
        wanted = {(i * stride) % total for i in range(budget)}
        for pos, chrom in enumerate(self.iter_grid()):
            if pos in wanted:
                yield chrom
