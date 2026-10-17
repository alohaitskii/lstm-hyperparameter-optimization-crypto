#!/usr/bin/env python
"""Uji reeval_final.py dengan isolasi subprocess, di sandbox (logs/ dan model/ tetap).

Sandbox = folder sementara berisi logs/optimization_results.csv (15 baris pencarian
snapshot) + salinan config snapshot dengan optimization.output_root = sandbox.
reeval_final.py dipanggil apa adanya (isolasi default). main.run_backtest selalu
menulis CSV ke <repo>/logs, jadi berkas backtest baru segera dipindah ke sandbox;
di akhir, isi logs/ dan model/ diperiksa sama dengan sebelum uji.

Mode:
  perbaikan          Urutan BTC-USD: GA s42, Manual s42, Grid s44, Manual s42.
                     Manual s42 (posisi 2 dan 4) harus identik dengan acuan
                     proses terpisah (bahan_bab4/uji_isolasi/terpisah.json):
                     wf_auc, AUC per fold, jumlah sinyal, 200 probabilitas.
                     -> bahan_bab4/uji_perbaikan_isolasi.md
  ulang K1 K2 ...    K = TICKER:metode:seed. Jalankan ulang tiap kombinasi di
                     sandbox, bandingkan dengan baris reeval terbaru di logs/.
                     -> bahan_bab4/uji_ulang_reeval.md

Usage:
    python tools/uji_perbaikan_isolasi.py perbaikan
    python tools/uji_perbaikan_isolasi.py ulang ETH-USD:grid:43 SOL-USD:manual:44
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

CONFIG = ROOT / "config_snapshot_20260912.yaml"
LOGS = ROOT / "logs"
ACUAN = ROOT / "bahan_bab4" / "uji_isolasi" / "terpisah.json"
UJI_DET = {"wf_auc": 0.6481331256878202, "n_signals_issued": 13, "n_signals_hold": 187}


def _snapshot_dirs() -> dict:
    out = {}
    for d in ("logs", "model"):
        for f in (ROOT / d).rglob("*"):
            if f.is_file() and f.suffix != ".log":
                st = f.stat()
                out[str(f)] = (st.st_size, st.st_mtime_ns)
    return out


class Sandbox:
    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="sandbox_reeval_"))
        (self.dir / "logs").mkdir()
        (self.dir / "backtest").mkdir()
        res = pd.read_csv(LOGS / "optimization_results.csv", dtype=str, keep_default_na=False)
        res.iloc[:15].to_csv(self.dir / "logs" / "optimization_results.csv", index=False)
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        assert cfg["data"]["cache_only"] is True
        cfg["optimization"]["output_root"] = str(self.dir)
        self.cfg = self.dir / "config_sandbox.yaml"
        self.cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self.guard = _snapshot_dirs()
        self.pos = 0

    def run(self, ticker: str, method: str, seed: int) -> dict:
        self.pos += 1
        before = set(LOGS.glob("backtest_*.csv"))
        print(f"\n===== [{self.pos}] {ticker} {method} s{seed} ({datetime.now():%H:%M:%S})", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "optimization" / "reeval_final.py"), "--config", str(self.cfg),
             "--tickers", ticker, "--methods", method, "--seeds", str(seed)],
            cwd=ROOT, check=True,
        )
        new = sorted(set(LOGS.glob("backtest_*.csv")) - before)
        assert len(new) == 1, new
        dst = self.dir / "backtest" / f"{self.pos}_{new[0].name}"
        shutil.move(str(new[0]), dst)
        rows = pd.read_csv(self.dir / "logs" / "optimization_results.csv").iloc[15:]
        r = rows.iloc[-1]
        assert (r.ticker, r.method, int(r.seed)) == (ticker, method, seed)
        folds = pd.read_csv(self.dir / "logs" / "optimization_folds.csv")
        f = folds[folds.timestamp == r.timestamp].sort_values("fold")
        assert len(f) == 3
        bt = pd.read_csv(dst)
        return {"posisi": self.pos, "ticker": ticker, "method": method, "seed": seed,
                "wf_auc": float(r.wf_auc), "fold_auc": [float(x) for x in f.auc],
                "n_signals_issued": int(r.n_signals_issued), "n_signals_hold": int(r.n_signals_hold),
                "prob": bt.model_probability.tolist(), "row": r.to_dict()}

    def close(self) -> None:
        after = _snapshot_dirs()
        diff = sorted(set(self.guard.items()) ^ set(after.items()))
        assert not diff, f"logs/ atau model/ berubah: {diff[:5]}"


def f6(x: float) -> str:
    return f"{x:.6f}"


def same(a: dict, b: dict) -> dict:
    """Bandingkan pada presisi yang tersimpan di CSV (6 desimal) + probabilitas penuh."""
    return {
        "wf_auc": f6(a["wf_auc"]) == f6(b["wf_auc"]),
        "fold_auc": [f6(x) for x in a["fold_auc"]] == [f6(x) for x in b["fold_auc"]],
        "sinyal": (a["n_signals_issued"], a["n_signals_hold"]) == (b["n_signals_issued"], b["n_signals_hold"]),
        "prob": [round(x, 10) for x in a["prob"]] == [round(x, 10) for x in b["prob"]],
    }


def mode_perbaikan() -> int:
    acuan = json.loads(ACUAN.read_text(encoding="utf-8"))["runs"][0]
    sb = Sandbox()
    runs = [sb.run("BTC-USD", m, s) for m, s in (("ga", 42), ("manual", 42), ("grid", 44), ("manual", 42))]
    sb.close()
    L = ["# Uji perbaikan isolasi `reeval_final.py`\n",
         f"Dihasilkan oleh `tools/uji_perbaikan_isolasi.py perbaikan` "
         f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC). `reeval_final.py` dijalankan apa adanya "
         "(isolasi subprocess, default) di sandbox; `logs/` dan `model/` diperiksa tidak berubah. "
         "Acuan = Manual s42 di proses terpisah (`bahan_bab4/uji_isolasi/terpisah.json`).\n",
         "| Posisi | Run | wf_auc | AUC per fold | Terbit/HOLD | wf_auc | fold | sinyal | 200 prob. |",
         "|---:|---|---:|---|---:|---|---|---|---|"]
    ok_all = True
    for r in runs:
        if (r["method"], r["seed"]) == ("manual", 42):
            c = same(r, acuan)
            ok_all &= all(c.values())
            flags = [("ya" if c[k] else "**tidak**") for k in ("wf_auc", "fold_auc", "sinyal", "prob")]
        else:
            flags = ["–"] * 4
        L.append(f"| {r['posisi']} | {r['method']} s{r['seed']} | {f6(r['wf_auc'])} | "
                 f"{', '.join(f6(x) for x in r['fold_auc'])} | {r['n_signals_issued']}/{r['n_signals_hold']} | "
                 + " | ".join(flags) + " |")
    L.append(f"| acuan | manual s42 (proses terpisah) | {f6(acuan['wf_auc'])} | "
             f"{', '.join(f6(x) for x in acuan['fold_auc'])} | {acuan['n_signals_issued']}/{acuan['n_signals_hold']} | | | | |")
    m2, m4 = runs[1], runs[3]
    det_ok = f6(m2["wf_auc"]) == f6(UJI_DET["wf_auc"]) and m2["n_signals_issued"] == UJI_DET["n_signals_issued"]
    L.append(f"\nKolom identik dibandingkan dengan acuan. Manual s42 posisi 2 = posisi 4: "
             f"**{'ya' if all(same(m2, m4).values()) else 'tidak'}**. Sama dengan uji determinisme "
             f"Langkah 2b (wf_auc {UJI_DET['wf_auc']:.6f}, {UJI_DET['n_signals_issued']}/{UJI_DET['n_signals_hold']}): "
             f"**{'ya' if det_ok else 'tidak'}**.\n")
    L.append(f"**HASIL: {'LULUS' if ok_all else 'GAGAL'}**\n")
    (ROOT / "bahan_bab4" / "uji_perbaikan_isolasi.md").write_text("\n".join(L), encoding="utf-8")
    (ROOT / "bahan_bab4" / "uji_isolasi" / "perbaikan.json").write_text(
        json.dumps({"runs": [{k: v for k, v in r.items() if k != "row"} for r in runs]}, indent=1),
        encoding="utf-8")
    print("HASIL:", "LULUS" if ok_all else "GAGAL")
    return 0 if ok_all else 1


def mode_ulang(keys: list[str]) -> int:
    res = pd.read_csv(LOGS / "optimization_results.csv")
    ree = res.iloc[15:]
    folds = pd.read_csv(LOGS / "optimization_folds.csv")
    targets = []
    for k in keys:
        t, m, s = k.split(":")
        r = ree[(ree.ticker == t) & (ree.method == m) & (ree.seed == int(s))]
        assert len(r) == 1, k
        r = r.iloc[0]
        f = folds[(folds.ticker == t) & (folds.method == m) & (folds.seed == int(s))].sort_values("fold")
        stamp = pd.Timestamp(r.timestamp)
        cand = sorted(LOGS.glob(f"backtest_{t.replace('-', '_')}_{m}_*.csv"))
        bt = min(cand, key=lambda p: abs((pd.Timestamp(p.stem[-15:].replace("_", "T"), tz="UTC") - stamp).total_seconds()))
        targets.append((t, m, int(s), {
            "wf_auc": float(r.wf_auc), "fold_auc": [float(x) for x in f.auc],
            "n_signals_issued": int(r.n_signals_issued), "n_signals_hold": int(r.n_signals_hold),
            "prob": pd.read_csv(bt).model_probability.tolist()}))
    sb = Sandbox()
    L = ["# Uji ulang kombinasi reeval di proses terpisah\n",
         f"Dihasilkan oleh `tools/uji_perbaikan_isolasi.py ulang` ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC). "
         "Setiap kombinasi dijalankan ulang dengan `reeval_final.py` di sandbox lalu dibandingkan dengan "
         "baris reeval di `logs/`.\n",
         "| Kombinasi | wf_auc (logs) | wf_auc (ulang) | wf_auc | fold | sinyal | 200 prob. |",
         "|---|---:|---:|---|---|---|---|"]
    ok_all = True
    for t, m, s, ref in targets:
        r = sb.run(t, m, s)
        c = same(r, ref)
        ok_all &= all(c.values())
        L.append(f"| {t} {m} s{s} | {f6(ref['wf_auc'])} | {f6(r['wf_auc'])} | "
                 + " | ".join("ya" if c[k] else "**tidak**" for k in ("wf_auc", "fold_auc", "sinyal", "prob")) + " |")
    sb.close()
    L.append(f"\n**HASIL: {'LULUS' if ok_all else 'GAGAL'}**\n")
    (ROOT / "bahan_bab4" / "uji_ulang_reeval.md").write_text("\n".join(L), encoding="utf-8")
    print("HASIL:", "LULUS" if ok_all else "GAGAL")
    return 0 if ok_all else 1


def main() -> int:
    if sys.argv[1] == "perbaikan":
        return mode_perbaikan()
    if sys.argv[1] == "ulang":
        return mode_ulang(sys.argv[2:])
    raise SystemExit(f"mode tidak dikenal: {sys.argv[1]}")


if __name__ == "__main__":
    sys.exit(main())
