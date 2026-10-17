#!/usr/bin/env python
"""Uji pengaruh urutan run dalam satu proses pada evaluasi akhir (butir i.3).

BTC-USD Manual seed 42 memberi wf_auc 0,648133 / backtest 13 sinyal di proses
terpisah, tetapi 0,664560 / 142 sinyal pada Langkah 4, saat ia menjadi run ke-7
dalam satu proses reeval. Uji ini memakai jalur kode reeval_final persis
(set_global_seed -> prepare_ticker_data sekali per ticker -> final_evaluate_and_save,
artefak hanya untuk seed pertama), tetapi semua keluaran (model, backtest) ditulis
ke folder sementara. logs/ dan model/ tidak disentuh.

Mode (masing-masing di proses Python baru):
  terpisah  Manual s42 saja di proses baru (acuan).
  urutan    Urutan Langkah 4 untuk BTC dalam SATU proses: GA s42,s43,s44, Grid
            s42,s43,s44, Manual s42; lalu fitness pencarian (Manual, GA terbaik,
            Grid terbaik) di proses yang sama.
  fitness   Fitness pencarian ketiga konfigurasi di proses baru.
  isolasi   Usulan perbaikan: setiap (metode, seed) di subprocess baru, urutan
            GA s42, Manual s42, Grid s44, Manual s42.
  semua     Jalankan keempatnya berurutan, lalu tulis laporan.
  laporan   Tulis bahan_bab4/uji_isolasi_urutan.md dari hasil JSON.

Hasil JSON: bahan_bab4/uji_isolasi/. Perkiraan waktu mode semua: 80-100 menit.

Usage:
    python tools/uji_isolasi_urutan.py semua
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

CONFIG = "config_snapshot_20260912.yaml"
TICKER = "BTC-USD"
# UJI_CEPAT=1: uji kering alat ini (1 epoch), hasil ke folder sementara, bukan bahan skripsi
CEPAT = os.environ.get("UJI_CEPAT") == "1"
if CEPAT:
    OUTDIR = Path(tempfile.gettempdir()) / "uji_isolasi_cepat"
    REPORT = OUTDIR / "uji_isolasi_urutan.md"
else:
    OUTDIR = ROOT / "bahan_bab4" / "uji_isolasi"
    REPORT = ROOT / "bahan_bab4" / "uji_isolasi_urutan.md"
URUTAN_L4 = [("ga", 42), ("ga", 43), ("ga", 44), ("grid", 42), ("grid", 43), ("grid", 44),
             ("manual", 42)]
URUTAN_ISO = [("ga", 42), ("manual", 42), ("grid", 44), ("manual", 42)]
UJI_DET = {"wf_auc": 0.6481331256878202, "n_signals_issued": 13, "n_signals_hold": 187}


def _snapshot_dirs() -> dict:
    """Daftar berkas (ukuran, mtime) di logs/ dan model/ untuk memastikan tidak tersentuh."""
    out = {}
    for d in ("logs", "model"):
        for f in (ROOT / d).rglob("*"):
            if f.is_file() and f.suffix != ".log":  # log run (git-ignored) boleh bertambah
                st = f.stat()
                out[str(f)] = (st.st_size, st.st_mtime_ns)
    return out


def _check_untouched(before: dict) -> None:
    after = _snapshot_dirs()
    diff = sorted(set(before.items()) ^ set(after.items()))
    assert not diff, f"logs/ atau model/ berubah: {diff[:5]}"


# --------------------------------------------------------------------------- #
# Inti: satu evaluasi akhir, jalur reeval_final
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self) -> None:
        import pandas as pd

        import optimization.experiment  # noqa: F401  (impor modul seperti reeval_final)
        import utils.helpers as helpers
        from utils.helpers import load_config

        assert "main" not in sys.modules, "main harus diimpor oleh final_evaluate_and_save"
        self.pd = pd
        self.cfg = load_config(CONFIG)
        assert self.cfg["data"]["cache_only"] is True
        if CEPAT:
            self.cfg["optimization"]["final_epochs"] = 1
            self.cfg["optimization"]["search_epochs"] = 1
        self.tmp = Path(tempfile.mkdtemp(prefix="uji_isolasi_"))
        # run_backtest menulis CSV ke project_root()/logs. main.py diimpor (lazy) di dalam
        # final_evaluate_and_save, persis seperti reeval_final, dan saat itu mengikat
        # project_root dari utils.helpers -> ganti di sana agar backtest masuk folder
        # sementara. Modul data/model sudah mengikat versi asli lebih dulu.
        helpers.project_root = lambda: self.tmp
        self.guard = _snapshot_dirs()
        src = pd.read_csv(ROOT / "logs" / "optimization_results.csv").iloc[:15]
        self.src = src[src.ticker == TICKER].set_index("method")
        self.data = None
        self.pos = 0

    def final(self, method: str, seed: int, persist: bool) -> dict:
        from optimization.experiment import final_evaluate_and_save, prepare_ticker_data
        from utils.helpers import set_global_seed

        self.pos += 1
        hp = json.loads(self.src.loc[method, "best_hp_json"])
        t0 = time.time()
        set_global_seed(seed)  # sama dengan reeval_final
        if self.data is None:
            self.data = prepare_ticker_data(self.cfg, TICKER)
        d = self.data
        before = set((self.tmp / "logs").glob("*.csv")) if (self.tmp / "logs").exists() else set()
        r = final_evaluate_and_save(
            TICKER, method, hp, d["X"], d["y"], d["fgi_encoder"], self.cfg, self.tmp, seed,
            200, persist_artifacts=persist, bundle=d["bundle"],
        )
        new = sorted(set((self.tmp / "logs").glob("*.csv")) - before)
        assert len(new) == 1, new
        bt = self.pd.read_csv(new[0])
        return {
            "posisi": self.pos, "method": method, "seed": seed, "persist": persist,
            "wf_auc": r["wf_auc"], "wf_f1": r["wf_f1"], "wf_precision": r["wf_precision"],
            "wf_recall": r["wf_recall"], "hit_rate": r["hit_rate"],
            "n_signals_issued": r["n_signals_issued"], "n_signals_hold": r["n_signals_hold"],
            "fold_auc": [f["auc"] for f in r["folds"]],
            "prob": bt["model_probability"].round(10).tolist(),
            "detik": round(time.time() - t0, 1),
        }

    def fitness(self, method: str) -> dict:
        from optimization.fitness import evaluate

        if self.data is None:
            from optimization.experiment import prepare_ticker_data
            self.data = prepare_ticker_data(self.cfg, TICKER)
        hp = json.loads(self.src.loc[method, "best_hp_json"])
        r = evaluate(hp, self.data["X"], self.data["y"], self.cfg, seed=42)
        return {"method": method, "auc": float(r["auc"]),
                "val_auc_search": float(self.src.loc[method, "val_auc_search"])}


def _save(name: str, obj) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")


def mode_terpisah() -> None:
    rn = Runner()
    _save("terpisah", {"runs": [rn.final("manual", 42, True)]})
    _check_untouched(rn.guard)


def mode_urutan() -> None:
    rn = Runner()
    seen: set = set()
    runs = []
    for method, seed in URUTAN_L4:
        persist = method not in seen  # artefak hanya seed pertama, seperti reeval_final
        seen.add(method)
        runs.append(rn.final(method, seed, persist))
        print(f"[urutan] {method} s{seed}: wf_auc={runs[-1]['wf_auc']!r}", flush=True)
    fit = [rn.fitness(m) for m in ("manual", "ga", "grid")]
    _save("urutan", {"runs": runs, "fitness_setelah_7_run": fit})
    _check_untouched(rn.guard)


def mode_fitness() -> None:
    rn = Runner()
    _save("fitness", {"fitness_proses_baru": [rn.fitness(m) for m in ("manual", "ga", "grid")]})
    _check_untouched(rn.guard)


def mode_worker(method: str, seed: int, persist: bool, name: str) -> None:
    rn = Runner()
    _save(name, {"runs": [rn.final(method, seed, persist)]})
    _check_untouched(rn.guard)


def _sub(*args: str) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    print(f"\n===== {' '.join(args)} ({datetime.now().strftime('%H:%M:%S')})", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def mode_isolasi() -> None:
    runs = []
    seen: set = set()
    for i, (method, seed) in enumerate(URUTAN_ISO, 1):
        persist = method not in seen
        seen.add(method)
        name = f"isolasi_{i}"
        _sub("worker", method, str(seed), str(int(persist)), name)
        r = json.loads((OUTDIR / f"{name}.json").read_text(encoding="utf-8"))["runs"][0]
        r["posisi"] = i
        runs.append(r)
        (OUTDIR / f"{name}.json").unlink()
    _save("isolasi", {"runs": runs})


# --------------------------------------------------------------------------- #
# Laporan
# --------------------------------------------------------------------------- #
def mode_laporan() -> None:
    import pandas as pd

    def load(n):
        return json.loads((OUTDIR / f"{n}.json").read_text(encoding="utf-8"))

    ter, uru, fit, iso = load("terpisah"), load("urutan"), load("fitness"), load("isolasi")
    ref = ter["runs"][0]
    res = pd.read_csv(ROOT / "logs" / "optimization_results.csv")
    l4 = res.iloc[15:]
    l4 = l4[l4.ticker == TICKER].set_index(["method", "seed"])

    def f6(x):
        return f"{x:.6f}".replace(".", ",")

    def same(a, b):
        return (a["wf_auc"] == b["wf_auc"] and a["fold_auc"] == b["fold_auc"]
                and a["n_signals_issued"] == b["n_signals_issued"]
                and a["n_signals_hold"] == b["n_signals_hold"] and a["prob"] == b["prob"])

    L = []
    P = L.append
    P("# Uji pengaruh urutan run pada evaluasi akhir (butir i.3)\n")
    P(f"Dihasilkan oleh `tools/uji_isolasi_urutan.py` ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC). "
      "Jalur kode = `reeval_final` (set_global_seed → prepare_ticker_data sekali per ticker → "
      "final_evaluate_and_save; artefak hanya seed pertama). Keluaran model dan backtest ditulis "
      "ke folder sementara; `logs/` dan `model/` tidak disentuh. Data: snapshot, cache-only.\n")

    P("## a. Urutan Langkah 4 dalam satu proses\n")
    P("Setiap run dibandingkan dengan baris Langkah 4 di `optimization_results.csv` "
      "(6 desimal, seperti tersimpan) dan, untuk Manual s42, dengan proses terpisah.\n")
    P("| Posisi | Run | wf_auc (uji ini) | wf_auc Langkah 4 | Sama dgn Langkah 4? | Sinyal terbit/HOLD (uji) | Langkah 4 | Detik |")
    P("|---:|---|---:|---:|---|---:|---:|---:|")
    n_l4 = 0
    for r in uru["runs"]:
        row = l4.loc[(r["method"], r["seed"])]
        ok = (f"{r['wf_auc']:.6f}" == f"{row.wf_auc:.6f}"
              and r["n_signals_issued"] == row.n_signals_issued and r["n_signals_hold"] == row.n_signals_hold)
        n_l4 += ok
        P(f"| {r['posisi']} | {r['method']} s{r['seed']} | {f6(r['wf_auc'])} | {f6(row.wf_auc)} | "
          f"{'ya' if ok else '**tidak**'} | {r['n_signals_issued']}/{r['n_signals_hold']} | "
          f"{int(row.n_signals_issued)}/{int(row.n_signals_hold)} | {r['detik']} |")
    m7 = uru["runs"][-1]
    P(f"\n{n_l4} dari {len(uru['runs'])} run mereproduksi Langkah 4.\n")
    P("| Manual s42 | wf_auc | AUC per fold | Sinyal terbit/HOLD |")
    P("|---|---:|---|---:|")
    P(f"| Uji determinisme (Langkah 2b) | {UJI_DET['wf_auc']!r} | – | {UJI_DET['n_signals_issued']}/{UJI_DET['n_signals_hold']} |")
    P(f"| Proses terpisah (uji ini) | {ref['wf_auc']!r} | {', '.join(f6(x) for x in ref['fold_auc'])} | {ref['n_signals_issued']}/{ref['n_signals_hold']} |")
    P(f"| Posisi 7 dalam satu proses (uji ini) | {m7['wf_auc']!r} | {', '.join(f6(x) for x in m7['fold_auc'])} | {m7['n_signals_issued']}/{m7['n_signals_hold']} |")
    P(f"\nProses terpisah = uji determinisme: **{'ya' if ref['wf_auc'] == UJI_DET['wf_auc'] and ref['n_signals_issued'] == UJI_DET['n_signals_issued'] else 'tidak'}**. "
      f"Posisi 7 = proses terpisah: **{'ya' if same(m7, ref) else 'tidak'}**.\n")

    P("## b. Tahap pencarian (fitness)\n")
    P("Fitness = `fitness.evaluate` (seed 42, split 70/15/15, `search_epochs`). Dibandingkan dengan "
      "`val_auc_search` yang tercatat saat pencarian (6 desimal).\n")
    P("| Konfigurasi | val_auc_search (pencarian) | Proses baru | Setelah 7 evaluasi akhir dalam satu proses |")
    P("|---|---:|---:|---:|")
    fb = {x["method"]: x for x in fit["fitness_proses_baru"]}
    fu = {x["method"]: x for x in uru["fitness_setelah_7_run"]}
    ok_fit = True
    for m in ("manual", "ga", "grid"):
        a, b, s = fb[m]["auc"], fu[m]["auc"], fb[m]["val_auc_search"]
        ok_fit &= f"{a:.6f}" == f"{s:.6f}" and a == b
        P(f"| {m} | {f6(s)} | {a!r} | {b!r} |")
    P(f"\nFitness tidak terdampak urutan: **{'ya' if ok_fit else 'tidak'}**.\n")

    P("## c. Usulan isolasi: setiap (metode, seed) di subprocess baru\n")
    P("| Posisi | Run | wf_auc | Sinyal terbit/HOLD | Identik dgn proses terpisah? |")
    P("|---:|---|---:|---:|---|")
    all_iso = True
    for r in iso["runs"]:
        line_ok = same(r, ref) if (r["method"], r["seed"]) == ("manual", 42) else None
        if line_ok is not None:
            all_iso &= line_ok
        P(f"| {r['posisi']} | {r['method']} s{r['seed']} | {r['wf_auc']!r} | "
          f"{r['n_signals_issued']}/{r['n_signals_hold']} | "
          f"{'–' if line_ok is None else ('ya' if line_ok else '**tidak**')} |")
    g42 = next(r for r in uru["runs"] if (r["method"], r["seed"]) == ("ga", 42))
    gi = next(r for r in iso["runs"] if (r["method"], r["seed"]) == ("ga", 42))
    P(f"\nIdentik = wf_auc, AUC tiap fold, jumlah sinyal, dan 200 probabilitas backtest sama persis. "
      f"Manual s42 di posisi 2 dan 4 identik dengan proses terpisah: **{'ya' if all_iso else 'tidak'}**. "
      f"GA s42 (posisi 1 di kedua cara) identik: **{'ya' if same(g42, gi) else 'tidak'}**.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"Ditulis: {REPORT}")


def main() -> int:
    mode = sys.argv[1]
    if mode == "terpisah":
        mode_terpisah()
    elif mode == "urutan":
        mode_urutan()
    elif mode == "fitness":
        mode_fitness()
    elif mode == "isolasi":
        mode_isolasi()
    elif mode == "worker":
        mode_worker(sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5])
    elif mode == "laporan":
        mode_laporan()
    elif mode == "semua":
        for m in ("terpisah", "fitness", "urutan", "isolasi"):
            _sub(m)
        mode_laporan()
    else:
        raise SystemExit(f"mode tidak dikenal: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
