#!/usr/bin/env python
"""Uji determinisme (Langkah 2b): hasil tidak boleh bergantung pada posisi
evaluasi di dalam satu proses.

Kombinasi baseline [2,3,3,1,2,1] + seed 42, BTC-USD, data snapshot (cache-only).

  python tools/uji_determinisme.py fitness       # proses 1: fitness pencarian
  python tools/uji_determinisme.py walkforward   # proses 2: evaluasi akhir
  python tools/uji_determinisme.py ringkas       # tabel + kode keluar 1 bila beda

fitness     : [baseline, 5 kromosom lain, baseline] lewat fitness.evaluate
              (yang menyetel seed sendiri). Posisi 1 vs 7: AUC & F1 identik.
walkforward : jalur persis reeval_final —
              posisi 1: set_global_seed(42) -> prepare_ticker_data -> final_evaluate_and_save
              posisi 2-6: lima evaluasi fitness lain (mengubah keadaan proses)
              posisi 7: set_global_seed(42) -> final_evaluate_and_save
              Dibandingkan: metrik walk-forward rata-rata, metrik setiap fold,
              hasil backtest, dan 200 probabilitas backtest per candle.
              fetch_all harus dipanggil tepat sekali (tanpa unduh di tengah run).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

CONFIG = "config_snapshot_20260912.yaml"
TICKER = "BTC-USD"
SEED = 42
BASELINE = [2, 3, 3, 1, 2, 1]
OTHERS = [[0, 3, 2, 1, 2, 2], [3, 3, 2, 3, 2, 0], [2, 0, 0, 2, 4, 0],
          [3, 1, 0, 3, 2, 2], [2, 0, 2, 0, 3, 2]]
OUT = ROOT / "bahan_bab4"
TAG = "uji_determinisme"


def _setup():
    import main as _main  # noqa: F401  (patch SSL milik repo; impor TF)
    from utils.logger import get_logger
    get_logger().setLevel(logging.WARNING)
    from optimization.search_space import SearchSpace
    from utils.helpers import load_config
    cfg = load_config(CONFIG)
    assert cfg["data"]["cache_only"] is True
    return cfg, SearchSpace.from_config(cfg)


def fase_fitness() -> None:
    cfg, space = _setup()
    from optimization.experiment import prepare_ticker_data
    from optimization.fitness import evaluate
    data = prepare_ticker_data(cfg, TICKER)
    rows = []
    for pos, chrom in enumerate([BASELINE, *OTHERS, BASELINE], start=1):
        t0 = time.time()
        r = evaluate(space.decode(chrom), data["X"], data["y"], cfg, seed=SEED)
        rows.append({"posisi": pos, "kromosom": chrom, "auc": r["auc"], "f1": r["f1"],
                     "n_val": r["n_val"], "detik": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    (OUT / "uji_determinisme_fitness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def fase_walkforward() -> None:
    cfg, space = _setup()
    import main as _main
    import optimization.experiment as E
    from optimization.fitness import evaluate
    from utils.helpers import project_root, set_global_seed

    n_fetch = {"n": 0}
    real_fetch = E.fetch_all

    def counting_fetch(*a, **k):
        n_fetch["n"] += 1
        return real_fetch(*a, **k)

    E.fetch_all = counting_fetch
    _main.fetch_all = counting_fetch

    hp = space.decode(BASELINE)
    logs = project_root() / "logs"

    def one_final(data) -> dict:
        before = set(logs.glob(f"backtest_BTC_USD_{TAG}_*.csv"))
        t0 = time.time()
        r = E.final_evaluate_and_save(
            TICKER, TAG, hp, data["X"], data["y"], data["fgi_encoder"], cfg,
            project_root(), SEED, 200, persist_artifacts=False, bundle=data["bundle"],
        )
        secs = time.time() - t0
        new = sorted(set(logs.glob(f"backtest_BTC_USD_{TAG}_*.csv")) - before)
        import pandas as pd
        bt = pd.read_csv(new[-1])
        for p in new:  # artefak uji, bukan data eksperimen
            p.unlink()
        return {
            "wf": {k: r[k] for k in ("wf_auc", "wf_f1", "wf_precision", "wf_recall")},
            "folds": [{k: f[k] for k in ("fold", "accuracy", "precision", "recall", "f1", "auc",
                                         "n_val_seq")} for f in r["folds"]],
            "backtest": {k: r[k] for k in ("hit_rate", "n_signals_issued", "n_signals_hold")},
            "prob": bt["model_probability"].tolist(),
            "detik": round(secs, 1),
        }

    res = {}
    set_global_seed(SEED)
    data = E.prepare_ticker_data(cfg, TICKER)      # posisi 1 memuat data, seperti reeval
    res["posisi_1"] = one_final(data)
    print("posisi 1:", json.dumps(res["posisi_1"]["wf"]), flush=True)
    sela = []
    for pos, chrom in enumerate(OTHERS, start=2):
        t0 = time.time()
        r = evaluate(space.decode(chrom), data["X"], data["y"], cfg, seed=SEED)
        sela.append({"posisi": pos, "kromosom": chrom, "auc": r["auc"],
                     "detik": round(time.time() - t0, 1)})
        print(json.dumps(sela[-1]), flush=True)
    set_global_seed(SEED)
    res["posisi_7"] = one_final(data)
    print("posisi 7:", json.dumps(res["posisi_7"]["wf"]), flush=True)
    res["selingan"] = sela
    res["fetch_all_calls"] = n_fetch["n"]
    (OUT / "uji_determinisme_walkforward.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def ringkas() -> int:
    fit = json.loads((OUT / "uji_determinisme_fitness.json").read_text(encoding="utf-8"))
    wf = json.loads((OUT / "uji_determinisme_walkforward.json").read_text(encoding="utf-8"))
    p1, p7 = fit[0], fit[-1]
    a, b = wf["posisi_1"], wf["posisi_7"]
    import numpy as np
    prob_a, prob_b = np.array(a["prob"]), np.array(b["prob"])

    def same(x, y) -> bool:
        """Exact equality that also treats NaN == NaN (repr of floats is exact)."""
        return json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)

    checks = [
        ("Fitness pencarian: AUC posisi 1 = posisi 7", same(p1["auc"], p7["auc"]),
         f"{p1['auc']!r} vs {p7['auc']!r}"),
        ("Fitness pencarian: F1 posisi 1 = posisi 7", same(p1["f1"], p7["f1"]),
         f"{p1['f1']!r} vs {p7['f1']!r}"),
        ("Walk-forward: AUC/F1/precision/recall rata-rata identik", same(a["wf"], b["wf"]),
         json.dumps(a["wf"]) + " vs " + json.dumps(b["wf"])),
        ("Walk-forward: metrik ketiga fold identik", same(a["folds"], b["folds"]),
         f"{len(a['folds'])} fold x 5 metrik"),
        ("Backtest: hit-rate & jumlah sinyal identik", same(a["backtest"], b["backtest"]),
         json.dumps(a["backtest"])),
        ("Backtest: 200 probabilitas per candle identik", bool(np.array_equal(prob_a, prob_b)),
         f"selisih maks {float(np.max(np.abs(prob_a - prob_b))):.3g}"),
        ("fetch_all dipanggil tepat sekali sepanjang run", wf["fetch_all_calls"] == 1,
         f"{wf['fetch_all_calls']}x"),
    ]
    L = ["# Uji determinisme (Langkah 2b)\n",
         f"Kombinasi baseline `{BASELINE}`, seed {SEED}, {TICKER}, data snapshot (cache-only). "
         "Posisi 1 dan posisi 7 dalam satu proses, diselingi 5 evaluasi lain "
         f"(kromosom `{OTHERS}`). Fitness dan walk-forward diuji di dua proses terpisah; "
         "proses walk-forward mengikuti jalur `reeval_final` persis (seed disetel, posisi 1 "
         "memuat data).\n",
         "| Pemeriksaan | Hasil | Rincian |", "|---|---|---|"]
    for name, ok, det in checks:
        L.append(f"| {name} | {'IDENTIK' if ok else '**BERBEDA**'} | {det} |")
    L += ["", "**Urutan evaluasi — fitness pencarian**\n", "| Posisi | Kromosom | AUC | F1 | Detik |",
          "|---:|---|---:|---:|---:|"]
    for r in fit:
        L.append(f"| {r['posisi']} | `{r['kromosom']}` | {r['auc']:.10f} | {r['f1']:.10f} | {r['detik']} |")
    L += ["", "**Urutan evaluasi — walk-forward**\n", "| Posisi | Evaluasi | AUC | Detik |",
          "|---:|---|---:|---:|",
          f"| 1 | `final_evaluate_and_save` baseline | {a['wf']['wf_auc']:.10f} | {a['detik']} |"]
    for r in wf["selingan"]:
        L.append(f"| {r['posisi']} | fitness `{r['kromosom']}` | {r['auc']:.10f} | {r['detik']} |")
    L.append(f"| 7 | `final_evaluate_and_save` baseline | {b['wf']['wf_auc']:.10f} | {b['detik']} |")
    L += ["", "**Metrik per fold (posisi 1)**\n", "| Fold | Accuracy | Precision | Recall | F1 | AUC | n_val_seq |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for f in a["folds"]:
        L.append(f"| {f['fold']} | {f['accuracy']:.6f} | {f['precision']:.6f} | {f['recall']:.6f} "
                 f"| {f['f1']:.6f} | {f['auc']:.6f} | {int(f['n_val_seq'])} |")
    ok_all = all(ok for _, ok, _ in checks)
    L.append(f"\n**Kesimpulan: {'seluruh hasil identik' if ok_all else 'ADA HASIL YANG BERBEDA'}.**\n")
    (OUT / "uji_determinisme.md").write_text("\n".join(L), encoding="utf-8")
    for name, ok, det in checks:
        print(f"  {'IDENTIK ' if ok else 'BERBEDA '} {name}  ({det[:80]})")
    print("\nHASIL:", "LULUS" if ok_all else "GAGAL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "fitness":
        fase_fitness()
    elif mode == "walkforward":
        fase_walkforward()
    elif mode == "ringkas":
        sys.exit(ringkas())
    else:
        sys.exit(__doc__)
