#!/usr/bin/env python
"""Uji penyebab ketidakdeterministikan evaluasi akhir antarproses (butir i.3).

Manual BTC-USD seed 42 memberi dua hasil berbeda yang masing-masing terulang
persis (0,648133 dan 0,664560), juga di proses baru. Uji ini menjalankan
evaluasi akhir yang sama (jalur reeval_final, keluaran ke folder sementara;
logs/ dan model/ dijaga tidak berubah) berkali-kali, SETIAP run di proses
Python baru, pada tiga kondisi:

  K1  seperti sekarang (kontrol)
  K2  affinity proses dibatasi ke P-core (EfficiencyClass tertinggi), oneDNN aktif
  K3  TF_ENABLE_ONEDNN_OPTS=0 + tf.config.experimental.enable_op_determinism()

Kondisi diterapkan di awal proses worker, SEBELUM TensorFlow diimpor. Putaran
dijalankan bergiliran K1, K2, K3 (8 putaran) agar efek waktu tidak menumpuk
pada satu kondisi. Kondisi lulus = 8/8 identik (wf_auc, AUC per fold, jumlah
sinyal, 200 probabilitas). Untuk tiap kondisi yang lulus, Grid BTC s42
dijalankan 4 kali; harus 4/4 identik.

Hasil: bahan_bab4/uji_penyebab/*.json dan bahan_bab4/uji_penyebab.md

Usage:
    python tools/uji_penyebab.py semua
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "bahan_bab4" / "uji_penyebab"
REPORT = ROOT / "bahan_bab4" / "uji_penyebab.md"
KONDISI = ["K1", "K2", "K3"]
KET = {
    "K1": "seperti sekarang (kontrol)",
    "K2": "affinity P-core saja, oneDNN aktif",
    "K3": "TF_ENABLE_ONEDNN_OPTS=0 + enable_op_determinism()",
}
N_MANUAL, N_GRID = 8, 4


# --------------------------------------------------------------------------- #
# Penerapan kondisi (dipakai juga oleh optimization/reeval_final.py)
# --------------------------------------------------------------------------- #
def p_core_mask() -> int:
    """Mask affinity logical processor dengan EfficiencyClass tertinggi (P-core)."""
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    need = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = (ctypes.c_byte * need.value)()
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        raise OSError(ctypes.get_last_error())
    raw, off, cls = bytes(buf), 0, {}
    while off < need.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        base = off + 8  # SYSTEM_CPU_SET_INFORMATION.CpuSet
        cls[raw[base + 6]] = raw[base + 10]  # LogicalProcessorIndex -> EfficiencyClass
        off += size
    top = max(cls.values())
    return sum(1 << lp for lp, c in cls.items() if c == top)


def terapkan_kondisi_awal(k: str) -> dict:
    """Bagian kondisi yang harus diterapkan SEBELUM TensorFlow diimpor."""
    info = {"kondisi": k}
    if k == "K2":
        mask = p_core_mask()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        h = k32.GetCurrentProcess()
        if not k32.SetProcessAffinityMask(ctypes.c_void_p(h), ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error())
        info["affinity_mask"] = hex(mask)
    elif k == "K3":
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        info["TF_ENABLE_ONEDNN_OPTS"] = "0"
    return info


def terapkan_kondisi_tf(k: str, info: dict) -> None:
    """Bagian kondisi sesudah TensorFlow diimpor."""
    if k == "K3":
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
        info["op_determinism"] = True


# --------------------------------------------------------------------------- #
def mode_worker(k: str, method: str, seed: int, out: str) -> None:
    info = terapkan_kondisi_awal(k)
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT))
    import uji_isolasi_urutan as U  # noqa: E402  (impor ringan; TF diimpor di Runner)

    import tensorflow  # noqa: F401  (impor di sini agar kondisi TF bisa diterapkan)
    terapkan_kondisi_tf(k, info)
    rn = U.Runner()
    r = rn.final(method, seed, True)
    U._check_untouched(rn.guard)
    r.update(info)
    Path(out).write_text(json.dumps(r), encoding="utf-8")


def _sub(k: str, method: str, seed: int, name: str) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{name}.json"
    print(f"\n===== {name}: {k} {method} s{seed} ({datetime.now():%H:%M:%S})", flush=True)
    env = {k_: v for k_, v in os.environ.items() if k_ != "TF_ENABLE_ONEDNN_OPTS"}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", k, method, str(seed), str(out)],
                   cwd=ROOT, check=True, env=env)
    r = json.loads(out.read_text(encoding="utf-8"))
    print(f"===== hasil {name}: wf_auc={r['wf_auc']!r} sinyal={r['n_signals_issued']}/{r['n_signals_hold']}",
          flush=True)
    return r


def kunci(r: dict) -> tuple:
    return (r["wf_auc"], tuple(r["fold_auc"]), r["n_signals_issued"], r["n_signals_hold"], tuple(r["prob"]))


def mode_semua() -> int:
    t0 = time.time()
    hasil = {k: [] for k in KONDISI}
    for i in range(1, N_MANUAL + 1):
        for k in KONDISI:
            hasil[k].append(_sub(k, "manual", 42, f"manual_{k}_{i}"))
    lulus_manual = {k: len({kunci(r) for r in hasil[k]}) == 1 for k in KONDISI}
    grid = {}
    for k in ("K2", "K3"):
        if lulus_manual[k]:
            grid[k] = [_sub(k, "grid", 42, f"grid_{k}_{i}") for i in range(1, N_GRID + 1)]
    lulus_grid = {k: len({kunci(r) for r in v}) == 1 for k, v in grid.items()}
    pilih = next((k for k in ("K2", "K3") if lulus_manual[k] and lulus_grid.get(k)), None)
    ringkas = {"lulus_manual": lulus_manual, "lulus_grid": lulus_grid, "pilih": pilih,
               "menit": round((time.time() - t0) / 60, 1),
               "selesai_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    (OUTDIR / "ringkasan.json").write_text(json.dumps(ringkas, indent=1), encoding="utf-8")
    tulis_laporan()
    print("PILIH:", pilih)
    return 0


def tulis_laporan() -> None:
    import glob

    ring = json.loads((OUTDIR / "ringkasan.json").read_text(encoding="utf-8"))
    runs = [json.loads(Path(f).read_text(encoding="utf-8")) | {"_nama": Path(f).stem}
            for f in sorted(glob.glob(str(OUTDIR / "*_K*_*.json")))]

    def f6(x):
        return f"{x:.6f}".replace(".", ",")

    L = ["# Uji penyebab ketidakdeterministikan evaluasi akhir (butir i.3)\n",
         f"Dihasilkan oleh `tools/uji_penyebab.py` (selesai {ring['selesai_utc']} UTC, {ring['menit']} menit). "
         "Setiap run = evaluasi akhir lengkap (walk-forward 3 fold + model pemenang + backtest) di "
         "proses Python baru, jalur `reeval_final`, keluaran ke folder sementara. Kondisi diterapkan "
         "sebelum TensorFlow diimpor. Putaran Manual bergiliran K1, K2, K3. CPU: Intel Core i5-12450H "
         "(4 P-core dengan hyper-threading = logical 0–7, 4 E-core = logical 8–11).\n",
         "| Kondisi | Keterangan |", "|---|---|"]
    L += [f"| {k} | {KET[k]} |" for k in KONDISI]
    for method, n in (("manual", N_MANUAL), ("grid", N_GRID)):
        L.append(f"\n## {'Manual' if method == 'manual' else 'Grid'} BTC-USD seed 42 ({n} run per kondisi)\n")
        L.append("Hasil berbeda dikelompokkan berdasarkan wf_auc, AUC per fold, jumlah sinyal, dan 200 probabilitas.\n")
        L.append("| Kondisi | Hasil (wf_auc, terbit/HOLD) | Muncul | Run ke- | Lulus |")
        L.append("|---|---|---:|---|---|")
        for k in KONDISI:
            rs = [r for r in runs if r["method"] == method and r["kondisi"] == k]
            if not rs:
                L.append(f"| {k} | tidak dijalankan | – | – | – |")
                continue
            grup: dict = {}
            for r in rs:
                grup.setdefault(kunci(r), []).append(int(r["_nama"].rsplit("_", 1)[1]))
            lulus = len(grup) == 1 and len(rs) == n
            for j, (key, idx) in enumerate(sorted(grup.items(), key=lambda kv: -len(kv[1]))):
                L.append(f"| {k if j == 0 else ''} | {f6(key[0])}, {key[2]}/{key[3]} | {len(idx)}/{len(rs)} | "
                         f"{', '.join(map(str, sorted(idx)))} | {('**ya**' if lulus else 'tidak') if j == 0 else ''} |")
        L.append("")
    det = [r for r in runs if r["method"] == "manual"]
    ref = {0.6481331256878202: "uji determinisme / proses terpisah", 0.6645601584570779: "Langkah 4 (reeval lama)"}
    seen = sorted({r["wf_auc"] for r in det})
    L.append("Pembanding Manual s42 sebelumnya: 0,648133 (uji determinisme, proses terpisah) dan "
             "0,664560 (Langkah 4). Hasil yang muncul pada uji ini: "
             + "; ".join(f"{f6(v)}" + (f" (= {ref[v]})" if v in ref else "") for v in seen) + ".\n")
    L.append(f"**Kondisi terpilih: {ring['pilih'] or 'tidak ada (K2 dan K3 tidak lulus)'}.** Aturan: K2 bila "
             "lulus, selain itu K3 bila lulus.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"Ditulis: {REPORT}")


def main() -> int:
    mode = sys.argv[1]
    if mode == "worker":
        mode_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
        return 0
    if mode == "semua":
        return mode_semua()
    if mode == "laporan":
        tulis_laporan()
        return 0
    raise SystemExit(f"mode tidak dikenal: {mode}")


if __name__ == "__main__":
    sys.exit(main())
