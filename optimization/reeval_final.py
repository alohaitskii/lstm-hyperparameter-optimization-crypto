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

Usage:
    python optimization/reeval_final.py --tickers BTC-USD --methods manual
    python optimization/reeval_final.py --seeds 42 43 44
    python optimization/reeval_final.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
    return p.parse_args()


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
                set_global_seed(seed)  # determinisme per seed
                if ticker not in data_cache:
                    data_cache[ticker] = prepare_ticker_data(cfg, ticker)
                data = data_cache[ticker]

                final = final_evaluate_and_save(
                    ticker, method, best_hp,
                    data["X"], data["y"], data["fgi_encoder"],
                    cfg, output_root, seed, args.lookback,
                    persist_artifacts=persist, bundle=data["bundle"],
                )
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
