"""Uji pencatatan per individu GA (Langkah 2a) — offline, tanpa pelatihan.

Fitness diganti fungsi deterministik (stub) sehingga uji berjalan dalam detik.
Yang dibuktikan:
 1. Populasi generasi 0 dari kode baru identik dengan populasi yang dibangkitkan
    ulang sebelumnya (seed 42), termasuk individu ke-4 = [3,2,1,3,2,1]
    (seq 90, u1 96, u2 32, drop 0.4, lr 0.001, bs 32) — best_hp generasi 0 BTC
    pada ga_history Juli.
 2. Pencatatan tidak mengubah jalannya GA: run dengan dan tanpa callback
    menghasilkan urutan evaluasi, riwayat, dan keadaan akhir rng yang identik.
 3. Bila git tersedia: GA baru identik dengan GA sebelum pencatatan
    (commit 123930e) — urutan evaluasi, riwayat, dan keadaan akhir rng.
 4. Isi jejak konsisten: anak = crossover(p1, p2, mask) lalu mutasi tercatat;
    pemenang turnamen = peserta dengan fitness tertinggi; elit = peringkat atas
    generasi sebelumnya; hitungan evaluasi kumulatif = jumlah non-cache.

Usage: python test_ga_trace.py   (kode keluar 1 bila gagal)
"""
from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import numpy as np  # noqa: E402

from utils.logger import get_logger  # noqa: E402
get_logger().setLevel(logging.WARNING)

import optimization.genetic_algorithm as ga_new  # noqa: E402
from optimization.search_space import GENE_ORDER, SearchSpace  # noqa: E402
from utils.helpers import load_config  # noqa: E402

REF_COMMIT = "123930e"   # GA sebelum pencatatan per individu
GEN0_SEED42 = [
    [0, 3, 2, 1, 2, 2], [0, 2, 0, 0, 2, 2], [3, 3, 2, 3, 2, 0], [4, 1, 2, 1, 0, 2],
    [3, 2, 1, 3, 2, 1], [2, 0, 0, 2, 4, 0], [4, 3, 1, 2, 0, 2], [3, 1, 0, 3, 2, 2],
    [3, 3, 3, 0, 1, 1], [2, 0, 2, 0, 3, 2],
]

cfg = load_config("config_snapshot_20260912.yaml")
space = SearchSpace.from_config(cfg)
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'OK  ' if cond else 'GAGAL'} {msg}")
    if not cond:
        fails.append(msg)


def make_stub(calls: list):
    """Deterministic fake fitness with spread and occasional ties."""
    def stub(hp, X, y, c, seed=42):
        chrom = space.encode(hp)
        calls.append(tuple(chrom))
        v = (sum((i + 3) * g * g for i, g in enumerate(chrom)) % 41) / 41.0
        return {"auc": 0.5 + 0.4 * v, "f1": 0.0, "n_train": 0, "n_val": 0, "duration": 0.0}
    return stub


def run(module, seed: int, traced: bool):
    calls: list = []
    module.evaluate = make_stub(calls)
    gens: list = []
    ind_rows: list = []
    tour_rows: list = []
    captured: dict = {}
    orig = np.random.default_rng

    def spy(s):
        g = orig(s)
        captured["rng"] = g
        return g

    module.np.random.default_rng = spy
    try:
        kw = {"on_individual": ind_rows.append, "on_tournament": tour_rows.append} if traced else {}
        res = module.run_ga(None, None, cfg, space, seed=seed, on_generation=gens.append, **kw)
    finally:
        module.np.random.default_rng = orig
    state = captured["rng"].bit_generator.state
    return {"calls": calls, "history": res["history"], "best": res["best_hp"],
            "n_evals": res["n_evals"], "state": state, "ind": ind_rows, "tour": tour_rows}


def same(a, b) -> bool:
    return (a["calls"] == b["calls"] and a["n_evals"] == b["n_evals"] and a["best"] == b["best"]
            and [(h["generation"], h["best_fitness"], h["mean_fitness"], h["best_hp"]) for h in a["history"]]
            == [(h["generation"], h["best_fitness"], h["mean_fitness"], h["best_hp"]) for h in b["history"]]
            and a["state"] == b["state"])


# ---------------------------------------------------------------------------
print("1) Populasi generasi 0 (seed 42)")
r = run(ga_new, 42, traced=True)
gen0 = [json.loads(x["chromosome"]) for x in r["ind"] if x["generation"] == 0]
check(gen0 == GEN0_SEED42, "10 kromosom generasi 0 identik dengan hasil regenerasi (urutan slot)")
check(gen0[4] == [3, 2, 1, 3, 2, 1] and space.decode(gen0[4]) == {
    "sequence_length": 90, "lstm_units_1": 96, "lstm_units_2": 32,
    "dropout_rate": 0.4, "learning_rate": 0.001, "batch_size": 32},
    f"individu ke-4 = {gen0[4]} -> {space.decode(gen0[4])}")
check(all(x["origin"] == "init" for x in r["ind"] if x["generation"] == 0), "origin generasi 0 = init")

print("\n2) Pencatatan tidak mengubah jalannya GA (dengan vs tanpa callback)")
for seed in (42, 7, 2026):
    a, b = run(ga_new, seed, traced=True), run(ga_new, seed, traced=False)
    check(same(a, b), f"seed {seed}: urutan {len(a['calls'])} evaluasi, riwayat, dan keadaan rng identik")

print(f"\n3) Identik dengan GA sebelum pencatatan (commit {REF_COMMIT})")
try:
    src = subprocess.run(["git", "show", f"{REF_COMMIT}:optimization/genetic_algorithm.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as d:
        pth = Path(d) / "ga_lama.py"
        pth.write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("ga_lama", pth)
        ga_old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ga_old)
        for seed in (42, 7, 2026):
            a, b = run(ga_new, seed, traced=True), run(ga_old, seed, traced=False)
            check(same(a, b), f"seed {seed}: kode baru (dengan jejak) == kode lama")
except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    print(f"  LEWAT  git tidak tersedia / commit tidak ditemukan: {exc}")

print("\n4) Konsistensi isi jejak (seed 42)")
ga_cfg = cfg["optimization"]["ga"]
pop, elit = ga_cfg["population_size"], ga_cfg["elitism"]
by_gen: dict[int, list] = {}
for x in r["ind"]:
    by_gen.setdefault(x["generation"], []).append(x)
ok_child = ok_elite = ok_tour = ok_cum = True
cum = 0
for g in sorted(by_gen):
    rows = sorted(by_gen[g], key=lambda x: x["slot"])
    for x in rows:
        if not x["from_cache"]:
            cum += 1
        ok_cum &= x["eval_count_kumulatif"] == cum
        if x["origin"] == "offspring":
            p1, p2 = json.loads(x["parent1_chromosome"]), json.loads(x["parent2_chromosome"])
            child = ([p1[i] if m == "1" else p2[i] for i, m in enumerate(x["crossover_mask"])]
                     if x["crossover_done"] else list(p1))
            for part in filter(None, x["mutated_genes"].split("; ")):
                gene, ch = part.split(": ")
                old, new = map(int, ch.split("->"))
                gi = GENE_ORDER.index(gene)
                ok_child &= child[gi] == old and old != new
                child[gi] = new
            ok_child &= child == json.loads(x["chromosome"])
        if x["origin"] == "elite" and g > 0:
            prev = sorted(by_gen[g - 1], key=lambda y: y["rank"])
            ok_elite &= json.loads(x["chromosome"]) == json.loads(prev[x["elite_from_rank"]]["chromosome"])
for t in r["tour"]:
    peserta, win = json.loads(t["peserta"]), json.loads(t["pemenang"])
    ok_tour &= win["fitness"] == max(p["fitness"] for p in peserta) and win in peserta
    child_row = next(x for x in by_gen[t["generation"]] if x["slot"] == t["offspring_slot"])
    key = "parent1_chromosome" if t["parent_ke"] == 1 else "parent2_chromosome"
    ok_tour &= json.loads(child_row[key]) == win["chromosome"]
n_off = sum(1 for x in r["ind"] if x["origin"] == "offspring")
check(ok_child, f"{n_off} anak: crossover(p1, p2, mask) + mutasi tercatat = kromosom anak")
check(ok_elite, "elit = kromosom pada peringkat yang sama di generasi sebelumnya")
check(ok_tour and len(r["tour"]) == 2 * n_off,
      f"{len(r['tour'])} turnamen: pemenang = fitness tertinggi peserta, = induk tercatat")
check(ok_cum and cum == r["n_evals"], f"evaluasi kumulatif konsisten ({cum} = n_evals)")
check(all(len(by_gen[g]) == pop for g in by_gen), f"{len(by_gen)} generasi x {pop} individu tercatat")

print("\nHASIL:", "LULUS" if not fails else f"GAGAL ({len(fails)})")
sys.exit(0 if not fails else 1)
