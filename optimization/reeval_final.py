#!/usr/bin/env python
"""reeval_final.py — Evaluasi ulang tahap AKHIR tanpa pencarian ulang.

Mengisi metrik yang hilang (precision, recall, jumlah sinyal backtest) pada
kombinasi (ticker, method) yang sudah punya hasil pencarian. Skrip ini TIDAK
memanggil run_ga maupun run_grid: `best_hp` dibaca apa adanya dari kolom
`best_hp_json` di optimization_results.csv, sehingga tidak ada hyperparameter
yang dicari ulang dan fase pencarian tidak pernah diulang.

Baris sumber yang dianggap sah:
    - search_epochs == 20 (baris bernilai 2 adalah sisa uji --fast)
    - method=grid: hanya baris dengan n_signals_issued terisi, yaitu hasil
      SETELAH perbaikan stride relatif prima; baris grid lama memakai langkah
      yang membekukan gen batch_size sehingga tidak sah
    - bila ada duplikat, dipakai yang timestamp-nya paling baru

Kolom yang DISALIN dari baris sumber (menggambarkan fase pencarian yang tidak
diulang): best_hp_json, val_auc_search, n_evals, duration_min, seed,
search_epochs, final_splits.
Kolom yang DIHITUNG ULANG: wf_auc, wf_f1, wf_precision, wf_recall,
backtest_hit_rate, n_signals_issued, n_signals_hold, n_candles_backtest.

Hasil ditulis sebagai baris BARU (append) — baris lama tidak pernah ditimpa.

CATATAN PENTING
    1. Karena model dilatih ulang, wf_auc/wf_f1 baru tidak akan sama persis
       dengan baris lama (status RNG berbeda). Untuk tabel skripsi pakai baris
       hasil skrip ini secara konsisten — jangan mencampur wf_auc lama dengan
       precision baru.
    2. Artefak di model/optimized/{TICKER}/{method}/ DITIMPA dengan model yang
       dilatih ulang memakai hyperparameter yang sama. model/saved/ tidak
       pernah disentuh.

MULTI-SEED
    --seeds menerima beberapa nilai (mis. --seeds 42 43 44). Tiap kombinasi
    dijalankan sekali per seed dan menghasilkan satu baris CSV per seed; kolom
    `seed` membedakannya. Ini yang dibutuhkan skripsi: sebaran metrik lintas
    seed (mean +/- std), karena variasi RNG pada hyperparameter yang sama
    ternyata jauh lebih besar daripada selisih antarmetode.

    Artefak model (model.keras, scaler.pkl, fgi_encoder.pkl, hyperparams.json)
    hanya disimpan untuk SEED PERTAMA setiap kombinasi. Seed berikutnya tetap
    dilatih dan di-backtest, tetapi hanya metriknya yang dicatat — hemat waktu
    tulis dan penyimpanan, dan artefak seed pertama tidak ikut tertimpa.

KONDISI NUMERIK (--kondisi)
    TensorFlow 2.21 di CPU hibrida (Intel i5-12450H, P-core + E-core) dengan
    oneDNN aktif terbukti memberi dua hasil berbeda untuk konfigurasi dan seed
    yang sama, juga di proses baru (bahan_bab4/uji_isolasi_urutan.md). Kondisi
    yang membuat hasil identik dipilih oleh tools/uji_penyebab.py dan dipasang
    pada setiap worker sebelum TensorFlow diimpor.

ISOLASI PROSES (default)
    Setiap run (ticker x metode x seed) dijalankan di subprocess Python baru
    (`--worker`, dengan kondisi numerik di atas), lalu proses induk menulis
    baris CSV-nya. Setiap run mulai dari keadaan TensorFlow/Keras yang sama,
    sehingga hasilnya hanya bergantung pada konfigurasi, seed, dan kondisi.
    `--tanpa-isolasi` mengembalikan perilaku lama (semua run satu proses, K1).

Usage:
    python optimization/reeval_final.py --tickers BTC-USD --methods manual
    python optimization/reeval_final.py --seeds 42 43 44
    python optimization/reeval_final.py
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Konsol Windows lawas memakai cp1252 — paksa UTF-8
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

# Kondisi numerik worker (lihat KONDISI NUMERIK di docstring) harus dipasang
# SEBELUM TensorFlow diimpor (lewat `import main` di bawah). Induk meneruskannya
# lewat variabel lingkungan REEVAL_KONDISI.
import os  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from uji_penyebab import terapkan_kondisi_awal, terapkan_kondisi_tf  # noqa: E402

_KONDISI = os.environ.get("REEVAL_KONDISI", "")
_KONDISI_INFO = terapkan_kondisi_awal(_KONDISI) if _KONDISI else {}

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

# import main menerapkan patch SSL milik pemilik repo (dibutuhkan saat fetch)
import main as _main  # noqa: E402, F401
from optimization.experiment import (  # noqa: E402
    METHODS,
    _blank_if_nan,
    append_fold_rows,
    append_result,
    final_evaluate_and_save,
    folds_csv_path,
    master_csv_path,
    prepare_ticker_data,
    resolve_output_root,
)
from data.fetcher import CacheMissingError, require_cached_inputs  # noqa: E402
from utils.helpers import load_config, set_global_seed  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402

console = Console()

VALID_SEARCH_EPOCHS = "20"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluasi ulang tahap akhir (tanpa pencarian hyperparameter ulang)"
    )
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Subset ticker (default: optimization.tickers di config)")
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   choices=list(METHODS), help="Metode (default: ga grid manual)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Satu atau beberapa seed, mis. --seeds 42 43 44 "
                        "(default: seed dari baris sumber). Artefak model hanya "
                        "disimpan untuk seed pertama tiap kombinasi.")
    p.add_argument("--lookback", type=int, default=200,
                   help="Lookback backtest (default 200, sama seperti run asli)")
    p.add_argument("--config", default="config_snapshot_20260912.yaml",
                   help="Path config YAML (default: snapshot beku, mode cache-only)")
    p.add_argument("--tanpa-isolasi", action="store_true",
                   help="Jalankan semua run dalam satu proses (perilaku lama). Default: "
                        "setiap run di subprocess baru agar hasil tidak bergantung urutan.")
    p.add_argument("--kondisi", choices=["K1", "K2", "K3"], default=KONDISI_DEFAULT,
                   help="Kondisi numerik worker (default %(default)s): K1 = bawaan, "
                        "K2 = affinity P-core, K3 = oneDNN mati + op determinism. "
                        "Lihat bahan_bab4/uji_penyebab.md.")
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# Kondisi numerik terpilih oleh tools/uji_penyebab.py (bahan_bab4/uji_penyebab.md)
KONDISI_DEFAULT = "K2"

# Kode keluar worker bila cache snapshot tidak lengkap (induk menghentikan semua run)
WORKER_EXIT_CACHE = 3


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v.item() if hasattr(v, "item") else v


def _run_one(cfg, ticker, method, best_hp, seed, persist, lookback, output_root,
             data_cache: dict) -> dict:
    """Satu evaluasi akhir: seed global disetel, data dimuat bila belum, lalu
    final_evaluate_and_save. Dipakai mode isolasi (di worker) dan tanpa isolasi."""
    set_global_seed(seed)  # determinisme per seed
    if ticker not in data_cache:
        data_cache[ticker] = prepare_ticker_data(cfg, ticker)
    data = data_cache[ticker]
    return final_evaluate_and_save(
        ticker, method, best_hp,
        data["X"], data["y"], data["fgi_encoder"],
        cfg, output_root, seed, lookback,
        persist_artifacts=persist, bundle=data["bundle"],
    )


def _worker(args: argparse.Namespace) -> int:
    """Subprocess: jalankan satu run dari berkas job, tulis hasil ke <job>.hasil.json."""
    job_path = Path(args.worker)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    setup_logger(level=cfg.get("logging", {}).get("log_level", "INFO"))
    terapkan_kondisi_tf(_KONDISI, _KONDISI_INFO)
    get_logger().info(f"Kondisi numerik worker: {_KONDISI_INFO or 'K1 (bawaan)'}")
    try:
        final = _run_one(cfg, job["ticker"], job["method"], job["best_hp"], job["seed"],
                         job["persist"], job["lookback"], Path(job["output_root"]), {})
    except CacheMissingError as exc:
        console.print(f"[red]{exc}[/red]")
        return WORKER_EXIT_CACHE
    job_path.with_suffix(".hasil.json").write_text(
        json.dumps(_jsonable(final)), encoding="utf-8")
    return 0


def _run_isolated(args, ticker, method, best_hp, seed, persist, output_root,
                  tmp_dir: Path) -> dict:
    job_path = tmp_dir / f"job_{ticker}_{method}_{seed}.json"
    job_path.write_text(json.dumps({
        "ticker": ticker, "method": method, "best_hp": best_hp, "seed": seed,
        "persist": persist, "lookback": args.lookback, "output_root": str(output_root),
    }), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "TF_ENABLE_ONEDNN_OPTS"}
    env["REEVAL_KONDISI"] = args.kondisi
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--config", args.config,
         "--worker", str(job_path)],
        cwd=ROOT, env=env,
    )
    if proc.returncode == WORKER_EXIT_CACHE:
        raise CacheMissingError(f"{ticker}: cache snapshot tidak lengkap (worker)")
    if proc.returncode != 0:
        raise RuntimeError(f"worker keluar dengan kode {proc.returncode}")
    return json.loads(job_path.with_suffix(".hasil.json").read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    """Float -> string 6 desimal; NaN -> string kosong (CSV-friendly)."""
    return f"{v:.6f}" if np.isfinite(v) else ""


def select_source_rows(
    master: Path, tickers: list[str], methods: list[str]
) -> dict[tuple[str, str], dict]:
    """Satu baris sumber terbaik per (ticker, method) menurut aturan keabsahan."""
    if not master.exists():
        raise SystemExit(f"CSV hasil tidak ditemukan: {master}")

    with open(master, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    chosen: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.get("ticker", ""), r.get("method", ""))
        if key[0] not in tickers or key[1] not in methods:
            continue
        if r.get("search_epochs") != VALID_SEARCH_EPOCHS:
            continue
        if not (r.get("best_hp_json") or "").strip():
            continue
        # Grid sebelum perbaikan stride tidak punya n_signals_issued
        if key[1] == "grid" and not (r.get("n_signals_issued") or "").strip():
            continue
        prev = chosen.get(key)
        if prev is None or r.get("timestamp", "") > prev.get("timestamp", ""):
            chosen[key] = r
    return chosen


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker(args)
    cfg = load_config(args.config)
    setup_logger(level=cfg.get("logging", {}).get("log_level", "INFO"))
    log = get_logger()

    opt_cfg = cfg.get("optimization") or {}
    tickers = list(args.tickers or opt_cfg.get("tickers") or [])
    methods = list(args.methods)
    output_root = resolve_output_root(cfg, None)
    master = master_csv_path(output_root)
    folds_path = folds_csv_path(output_root)

    # Cache-only: semua berkas masukan harus ada SEBELUM pekerjaan dimulai
    try:
        require_cached_inputs(cfg, tickers)
    except CacheMissingError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    sources = select_source_rows(master, tickers, methods)
    combos = [(t, m) for t in tickers for m in methods if (t, m) in sources]
    missing = [(t, m) for t in tickers for m in methods if (t, m) not in sources]

    if missing:
        console.print(
            "[yellow]Tidak ada baris sumber sah untuk:[/yellow] "
            + ", ".join(f"{t} x {m}" for t, m in missing)
        )
    if not combos:
        console.print("[red]Tidak ada kombinasi yang bisa dievaluasi ulang.[/red]")
        return 2

    seeds = list(args.seeds) if args.seeds else None
    total_runs = len(combos) * (len(seeds) if seeds else 1)
    console.print(
        f"[bold]Evaluasi ulang {len(combos)} kombinasi[/bold] "
        f"(best_hp diambil dari CSV — TIDAK ada pencarian ulang)"
    )
    if seeds:
        console.print(
            f"Seed: {seeds} -> {total_runs} run. "
            f"Artefak model hanya disimpan untuk seed {seeds[0]}.\n"
        )
    else:
        console.print("Seed: mengikuti baris sumber\n")
    isolate = not args.tanpa_isolasi
    if not isolate and args.kondisi != "K1":
        raise SystemExit("--tanpa-isolasi hanya untuk kondisi K1 (kondisi dipasang per worker)")
    console.print(f"Kondisi numerik: {args.kondisi}")
    console.print(
        "Isolasi: setiap run di subprocess baru\n" if isolate
        else "[yellow]Tanpa isolasi: semua run dalam satu proses[/yellow]\n"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="reeval_jobs_"))

    results: list[dict] = []
    data_cache: dict[str, dict] = {}
    t_start = time.time()
    run_i = 0
    stop = False

    for ticker, method in combos:
        if stop:
            break
        src = sources[(ticker, method)]
        best_hp = json.loads(src["best_hp_json"])
        combo_seeds = seeds or [int(src.get("seed") or opt_cfg.get("seed", 42))]

        for s_i, seed in enumerate(combo_seeds):
            run_i += 1
            # Artefak hanya untuk seed pertama: skripsi butuh sebaran metrik,
            # bukan salinan model per seed.
            persist = (s_i == 0)
            console.print(
                f"[cyan][{run_i}/{total_runs}] {ticker} x {method} seed={seed}[/cyan]"
                f"{'' if persist else ' (metrik saja)'}  hp={best_hp}"
            )
            t0 = time.time()
            try:
                if isolate:
                    final = _run_isolated(args, ticker, method, best_hp, seed, persist,
                                          output_root, tmp_dir)
                else:
                    final = _run_one(cfg, ticker, method, best_hp, seed, persist,
                                     args.lookback, output_root, data_cache)
            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Dihentikan — baris yang sudah selesai tetap tersimpan.[/yellow]"
                )
                stop = True
                break
            except CacheMissingError:
                raise  # cache hilang = snapshot tak lengkap -> hentikan semua
            except Exception as exc:  # noqa: BLE001
                log.error(f"GAGAL {ticker} x {method} seed={seed}: {exc}")
                continue

            elapsed_min = (time.time() - t0) / 60.0
            row = {
                # --- disalin: fase pencarian TIDAK diulang ---
                "ticker": ticker,
                "method": method,
                "seed": seed,
                "best_hp_json": src["best_hp_json"],
                "val_auc_search": src.get("val_auc_search", ""),
                "n_evals": src.get("n_evals", ""),
                "duration_min": src.get("duration_min", ""),
                "search_duration_min": src.get("search_duration_min", ""),
                "search_epochs": src.get("search_epochs", ""),
                "final_splits": src.get("final_splits", ""),
                # --- dihitung ulang ---
                "wf_auc": _fmt(final["wf_auc"]),
                "wf_f1": _fmt(final["wf_f1"]),
                "wf_precision": _fmt(final["wf_precision"]),
                "wf_recall": _fmt(final["wf_recall"]),
                "backtest_hit_rate": _fmt(final["hit_rate"]),
                "n_signals_issued": _blank_if_nan(final["n_signals_issued"]),
                "n_signals_hold": _blank_if_nan(final["n_signals_hold"]),
                "n_candles_backtest": _blank_if_nan(final["n_candles_backtest"]),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            append_result(master, row)
            append_fold_rows(folds_path, ticker, method, seed,
                             final.get("folds", []), row["timestamp"])
            results.append(row)
            console.print(
                f"    auc={row['wf_auc'][:6] or '-'} f1={row['wf_f1'][:6] or '-'} "
                f"prec={row['wf_precision'][:6] or '-'} rec={row['wf_recall'][:6] or '-'} "
                f"issued={row['n_signals_issued']} hold={row['n_signals_hold']} "
                f"hit={row['backtest_hit_rate'][:6] or '-'}  ({elapsed_min:.1f} menit)\n"
            )

    _print_reports(results)
    console.print(
        f"\n[bold]Selesai:[/bold] {len(results)}/{total_runs} run, "
        f"total {((time.time() - t_start) / 60):.1f} menit"
    )
    console.print(f"Baris baru di-append ke: {master}")
    console.print(f"Metrik per fold di-append ke: {folds_path}")
    return 0


def _agg(values: list[float]) -> str:
    """'mean +/- std (n)' — std dihilangkan bila hanya satu sampel."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f} (n=1)"
    return f"{mean(values):.4f} +/- {stdev(values):.4f} (n={len(values)})"


def _print_reports(results: list[dict]) -> None:
    """Tabel per-run, lalu tabel agregat lintas seed bila seed > 1."""
    if not results:
        return

    table = Table(title="[bold cyan]Hasil Evaluasi Ulang (per run)[/bold cyan]",
                  header_style="bold")
    for col in ("Ticker", "Metode", "Seed", "AUC", "F1", "Precision", "Recall",
                "Issued", "Hold", "Hit-rate"):
        table.add_column(col, justify="left" if col in ("Ticker", "Metode") else "right")
    for r in results:
        table.add_row(
            r["ticker"], r["method"], str(r["seed"]),
            (r["wf_auc"] or "-")[:6], (r["wf_f1"] or "-")[:6],
            (r["wf_precision"] or "-")[:6], (r["wf_recall"] or "-")[:6],
            str(r["n_signals_issued"]), str(r["n_signals_hold"]),
            (r["backtest_hit_rate"] or "-")[:6],
        )
    console.print(table)

    # Agregat lintas seed — inti dari mode multi-seed
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["ticker"], r["method"]), []).append(r)
    if all(len(v) < 2 for v in groups.values()):
        return

    def nums(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if (r[key] or "").strip()]

    agg = Table(
        title="[bold cyan]Agregat lintas seed (untuk tabel skripsi)[/bold cyan]",
        header_style="bold", show_lines=True,
    )
    for col in ("Ticker", "Metode", "AUC wf", "F1 wf", "Precision", "Recall"):
        agg.add_column(col, justify="left" if col in ("Ticker", "Metode") else "right")
    for (ticker, method), rows in groups.items():
        agg.add_row(
            ticker, method,
            _agg(nums(rows, "wf_auc")), _agg(nums(rows, "wf_f1")),
            _agg(nums(rows, "wf_precision")), _agg(nums(rows, "wf_recall")),
        )
    console.print(agg)


if __name__ == "__main__":
    sys.exit(main())
