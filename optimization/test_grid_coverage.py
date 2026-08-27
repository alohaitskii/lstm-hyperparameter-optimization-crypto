"""Uji penerimaan: cakupan Grid Search ber-budget terhadap ruang pencarian.

Membuktikan bahwa subsample deterministik yang dipakai Grid Search benar-benar
menelusuri SELURUH ruang pencarian — bukan hanya sebagian gen. Cacat yang
diuji di sini pernah nyata terjadi: dengan langkah total//budget = 96 (kelipatan
3 = jumlah nilai batch_size), gen batch_size membeku di indeks 0 sehingga Grid
Search efektif hanya menelusuri sepertiga ruang dan perbandingannya dengan
Algoritma Genetika menjadi tidak adil.

Pemeriksaan:
    1. Jumlah kombinasi tepat sama dengan budget
    2. Seluruh kombinasi unik
    3. Setiap nilai kandidat pada SETIAP gen muncul minimal sekali
    4. Sebaran frekuensi tiap gen tidak timpang (maks-min <= 40% rata-rata)

Mandiri, tanpa framework uji. Keluar dengan kode 1 bila ada yang gagal.

Usage:
    python optimization/test_grid_coverage.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Konsol Windows lawas memakai cp1252 — paksa UTF-8 agar simbol tabel aman
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

from optimization.search_space import GENE_ORDER, SearchSpace  # noqa: E402
from utils.helpers import load_config  # noqa: E402

# Ambang sebaran: selisih frekuensi tertinggi-terendah terhadap rata-rata
MAX_SPREAD_RATIO = 0.40


def main() -> int:
    cfg = load_config()
    space = SearchSpace.from_config(cfg)
    budget = int(((cfg.get("optimization") or {}).get("grid") or {}).get("budget", 50))
    total = space.size()

    chroms = list(space.iter_grid_budget(budget))
    failures: list[str] = []

    print("UJI CAKUPAN GRID SEARCH BER-BUDGET")
    print("=" * 78)
    print(f"Ruang pencarian penuh : {total} kombinasi")
    print(f"Budget evaluasi       : {budget}")
    print(f"Kombinasi dihasilkan  : {len(chroms)}")
    print()

    # --- 1. jumlah tepat ---------------------------------------------------- #
    if len(chroms) != budget:
        failures.append(f"Jumlah kombinasi {len(chroms)} != budget {budget}")

    # --- 2. keunikan -------------------------------------------------------- #
    n_unique = len({tuple(c) for c in chroms})
    if n_unique != len(chroms):
        failures.append(f"Ada {len(chroms) - n_unique} kombinasi duplikat")

    # --- 3 & 4. cakupan dan sebaran per gen --------------------------------- #
    print(f"{'Gen':18} {'Frekuensi tiap nilai kandidat':<44} {'Status'}")
    print("-" * 78)
    for g, gene in enumerate(GENE_ORDER):
        values = space.choices(gene)
        counts = Counter(c[g] for c in chroms)
        freqs = [counts.get(i, 0) for i in range(len(values))]
        shown = "  ".join(f"{v}:{n}" for v, n in zip(values, freqs))

        missing = [values[i] for i, n in enumerate(freqs) if n == 0]
        avg = sum(freqs) / len(freqs)
        spread = max(freqs) - min(freqs)
        limit = MAX_SPREAD_RATIO * avg

        if missing:
            status = "GAGAL (nilai tak diuji)"
            failures.append(f"{gene}: nilai tidak pernah diuji -> {missing}")
        elif spread > limit:
            status = f"GAGAL (spread {spread} > {limit:.2f})"
            failures.append(
                f"{gene}: sebaran timpang, maks-min={spread} melebihi "
                f"{MAX_SPREAD_RATIO:.0%} rata-rata ({limit:.2f})"
            )
        else:
            status = f"OK (spread {spread} <= {limit:.2f})"
        print(f"{gene:18} {shown:<44} {status}")

    print()
    if failures:
        print("HASIL: GAGAL")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("HASIL: LULUS — seluruh gen tercakup dan sebarannya seimbang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
