#!/usr/bin/env python
"""run_optimization.py — Eksperimen skripsi: GA vs Grid Search vs baseline manual.

Usage:
    python run_optimization.py --pilot                      # ukur waktu/eval dulu (1 ticker)
    python run_optimization.py --fast --tickers BTC-USD     # uji cepat end-to-end
    python run_optimization.py                              # 5 koin x 3 metode penuh
    python run_optimization.py --skip-done                  # resume (level ticker x metode)
    python run_optimization.py --methods ga grid            # subset metode
    python run_optimization.py --budget 50                  # budget evaluasi (GA & Grid)
    python run_optimization.py --output-root "G:/My Drive/skripsi"  # hasil ke Drive

Checkpoint: setiap (ticker x metode) selesai → baris langsung ditulis ke
logs/optimization_results.csv. Ctrl+C aman; lanjutkan dengan --skip-done.

Colab: mount Google Drive lalu berikan --output-root ke path Drive agar
logs/ dan model/optimized/ tidak hilang saat sesi terputus.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same UTF-8 fix as main.py for legacy Windows consoles
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table  # noqa: E402

# main import applies the user's SSL bypass patch — intentional (fetch works
# on this network). It also pulls TF, so GPU detection below is cheap.
import main as _main  # noqa: E402, F401
from optimization.experiment import (  # noqa: E402
    METHODS,
    RESULT_FIELDS,
    append_result,
    load_done,
    master_csv_path,
    prepare_ticker_data,
    resolve_output_root,
    run_method,
    run_pilot,
)
from optimization.search_space import SearchSpace  # noqa: E402
from utils.helpers import ensure_dir, load_config, set_global_seed  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimasi hyperparameter LSTM: GA vs Grid Search vs manual"
    )
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Subset ticker (default: optimization.tickers di config)")
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   choices=list(METHODS), help="Metode yang dijalankan")
    p.add_argument("--budget", type=int, default=None,
                   help="Budget evaluasi — dipakai Grid, dan membatasi GA (budget-matched)")
    p.add_argument("--pilot", action="store_true",
                   help="Hanya ukur waktu per-evaluasi di 1 ticker + cetak estimasi total")
    p.add_argument("--fast", action="store_true",
                   help="Setting mini utk uji: pop=4 gen=2 grid=4 search_epochs=2 final_splits=2")
    p.add_argument("--final-splits", type=int, default=None,
                   help="Override jumlah fold walk-forward evaluasi akhir")
    p.add_argument("--lookback", type=int, default=200,
                   help="Lookback backtest hit-rate (default 200)")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed (default: optimization.seed di config)")
    p.add_argument("--skip-done", action="store_true",
                   help="Lewati (ticker x metode) yang sudah ada di optimization_results.csv")
    p.add_argument("--output-root", default=None,
                   help="Root output (mis. path Google Drive). Default: repo")
    p.add_argument("--config", default="config.yaml", help="Path config YAML")
    return p.parse_args()


def _apply_fast(cfg: dict) -> None:
    """Shrink everything for a quick end-to-end verification run."""
    opt = cfg["optimization"]
    opt["search_epochs"] = 2
    opt["final_splits"] = 2
    opt["final_epochs"] = 3
    opt["ga"]["population_size"] = 4
    opt["ga"]["generations"] = 2
    opt["ga"]["elitism"] = 1
    opt["grid"]["budget"] = 4
    console.print(
        "[yellow]--fast aktif: pop=4 gen=2 grid=4 search_epochs=2 "
        "final_splits=2 final_epochs=3[/yellow]"
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logger(level=cfg.get("logging", {}).get("log_level", "INFO"))
    log = get_logger()

    if "optimization" not in cfg:
        console.print("[red]config.yaml tidak punya section optimization.[/red]")
        return 2

    if args.fast:
        _apply_fast(cfg)
    if args.final_splits is not None:
        cfg["optimization"]["final_splits"] = args.final_splits

    seed = args.seed if args.seed is not None else int(cfg["optimization"].get("seed", 42))
    set_global_seed(seed)
    log.info(f"Seed global: {seed}")

    # GPU / environment awareness (Colab)
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"GPU terdeteksi: {[g.name for g in gpus] if gpus else 'TIDAK ADA (CPU)'}")

    space = SearchSpace.from_config(cfg)
    log.info(f"Ruang pencarian: {space.size()} kombinasi ({space.n_genes} gen)")

    output_root = resolve_output_root(cfg, args.output_root)
    ensure_dir(output_root / "logs")
    log.info(f"Output root: {output_root}")

    tickers = list(args.tickers or cfg["optimization"].get("tickers") or [])
    registry = cfg["data"].get("tickers") or []
    unknown = [t for t in tickers if registry and t not in registry]
    if unknown:
        console.print(f"[red]Ticker tidak ada di registry data.tickers: {unknown}[/red]")
        return 2
    methods = list(args.methods)

    # ---------------- Pilot mode: timing only ---------------- #
    if args.pilot:
        pilot_ticker = tickers[0] if tickers else "BTC-USD"
        est = run_pilot(cfg, space, pilot_ticker, seed)
        console.print(Panel(
            f"Waktu/eval rata-rata : {est['mean_eval_seconds']:.1f}s\n"
            f"GA  (estimasi maks)  : {est['ga_evals_est']} evaluasi/koin\n"
            f"Grid (budget)        : {est['grid_evals']} evaluasi/koin\n"
            f"Jumlah ticker        : {est['n_tickers']}\n\n"
            f"[bold]Estimasi fase pencarian total ≈ {est['search_hours_est']:.1f} jam[/bold]\n"
            f"(belum termasuk evaluasi akhir walk-forward per metode —\n"
            f" tambahkan kira-kira 30–60%)",
            title="[bold cyan]PILOT — Estimasi Waktu[/bold cyan]", padding=(1, 2),
        ))
        return 0

    # ---------------- Full experiment ---------------- #
    master = master_csv_path(output_root)
    done = load_done(master) if args.skip_done else set()
    combos = [(t, m) for t in tickers for m in methods if (t, m) not in done]
    skipped = [(t, m) for t in tickers for m in methods if (t, m) in done]
    if skipped:
        console.print(f"[dim]--skip-done: {len(skipped)} kombinasi sudah selesai, dilewati[/dim]")
    if not combos:
        console.print("[green]Semua kombinasi sudah selesai.[/green]")
        return 0

    console.print(
        f"[bold]Eksperimen:[/bold] {len(tickers)} ticker × {methods} "
        f"→ {len(combos)} kombinasi berjalan\n"
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rows_this_run: list[dict] = []
    interrupted = False
    t_start = time.time()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    try:
        with progress:
            task = progress.add_task("memulai...", total=len(combos))
            current_ticker = None
            data = None
            for ticker, method in combos:
                progress.update(task, description=f"[cyan]{ticker} × {method}[/cyan]")
                try:
                    if ticker != current_ticker:
                        data = prepare_ticker_data(cfg, ticker)
                        current_ticker = ticker
                    row = run_method(
                        ticker, method, data["X"], data["y"], data["fgi_encoder"],
                        cfg, space, seed, output_root,
                        args.budget, args.lookback, ts,
                    )
                    append_result(master, row)   # checkpoint IMMEDIATELY
                    # per-ticker view (tabel per-koin untuk lampiran skripsi)
                    from utils.ticker_utils import ticker_to_safe_name
                    per_ticker = (
                        output_root / "logs"
                        / f"optimization_{ticker_to_safe_name(ticker)}_{ts}.csv"
                    )
                    append_result(per_ticker, row)
                    rows_this_run.append(row)
                    log.info(f"CHECKPOINT: {ticker} × {method} → {master}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.error(f"GAGAL {ticker} × {method}: {exc}")
                progress.advance(task)
    except KeyboardInterrupt:
        interrupted = True
        console.print(
            "\n[yellow]Dihentikan — progres tersimpan di CSV. "
            "Lanjutkan dengan --skip-done[/yellow]"
        )

    # ---------------- Summary ---------------- #
    all_rows = []
    if master.exists():
        with open(master, "r", encoding="utf-8", newline="") as f:
            all_rows = [r for r in csv.DictReader(f) if r["ticker"] in tickers]

    if all_rows:
        summary_path = output_root / "logs" / f"optimization_summary_{ts}.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerows(all_rows)
        log.info(f"Summary → {summary_path}")

        table = Table(
            title=f"[bold cyan]Hasil Optimasi — seed={seed}[/bold cyan]",
            header_style="bold",
        )
        for col in ("Ticker", "Metode", "AUC search", "AUC wf", "F1 wf", "Hit-rate", "n_evals", "Menit"):
            table.add_column(col, justify="right" if col not in ("Ticker", "Metode") else "left")
        for r in all_rows:
            table.add_row(
                r["ticker"], r["method"],
                r["val_auc_search"][:6] or "—", (r["wf_auc"] or "—")[:6],
                (r["wf_f1"] or "—")[:6], (r["backtest_hit_rate"] or "—")[:6],
                str(r["n_evals"]), str(r["duration_min"]),
            )
        console.print(table)

    console.print(
        f"\n[bold]Selesai run ini:[/bold] {len(rows_this_run)}/{len(combos)} kombinasi, "
        f"total {((time.time() - t_start) / 60):.1f} menit"
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
