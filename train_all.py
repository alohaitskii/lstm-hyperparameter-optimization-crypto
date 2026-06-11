#!/usr/bin/env python
"""train_all.py — Latih model LSTM untuk semua ticker di config.yaml secara berurutan.

Usage:
    python train_all.py                                  # latih semua (22 ticker)
    python train_all.py --skip-trained                   # lewati yang sudah punya model
    python train_all.py --tickers BTC-USD ETH-USD SOL-USD  # subset tertentu
    python train_all.py --from ETH-USD                   # lanjutkan dari ticker tertentu
    python train_all.py --fast                           # epochs=30, folds=3 (uji cepat)

Behaviors:
- Ticker yang gagal fetch / kurang data → SKIP, lanjut ke berikutnya
- Exception saat training → FAIL, lanjut ke berikutnya
- Ctrl+C → simpan laporan progres lalu keluar bersih (resume: --skip-trained)
- Laporan akhir → logs/train_all_report_{timestamp}.csv
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
from rich.text import Text  # noqa: E402

# Importing main also applies the user's SSL bypass patch — intentional.
from main import InsufficientDataError, run_train  # noqa: E402
from utils.helpers import ensure_dir, load_config, project_root  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402
from utils.ticker_utils import is_trained  # noqa: E402

console = Console()


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {int(seconds % 60)}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latih model LSTM untuk semua ticker secara berurutan"
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Subset ticker tertentu (default: semua di config.yaml)",
    )
    p.add_argument(
        "--skip-trained", action="store_true",
        help="Lewati ticker yang sudah punya model tersimpan (resume)",
    )
    p.add_argument(
        "--from", dest="from_ticker", default=None, metavar="TICKER",
        help="Mulai dari ticker tertentu dalam urutan list",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Mode cepat untuk uji: epochs=30, n_splits_cv=3",
    )
    p.add_argument("--config", default="config.yaml", help="Path config YAML")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logger(level=cfg.get("logging", {}).get("log_level", "INFO"))
    log = get_logger()

    all_tickers: list[str] = list(cfg["data"].get("tickers") or [])
    if not all_tickers:
        console.print("[red]config.yaml: data.tickers kosong.[/red]")
        return 2

    # --tickers subset (validated against config registry)
    tickers = list(args.tickers) if args.tickers else list(all_tickers)
    unknown = [t for t in tickers if t not in all_tickers]
    if unknown:
        console.print(
            f"[yellow]Ticker tidak dikenal, dilewati: {', '.join(unknown)}[/yellow]"
        )
        tickers = [t for t in tickers if t in all_tickers]

    # --from: resume from a position in the list
    if args.from_ticker:
        if args.from_ticker not in tickers:
            console.print(f"[red]--from {args.from_ticker} tidak ada dalam daftar.[/red]")
            return 2
        tickers = tickers[tickers.index(args.from_ticker):]

    # --skip-trained: drop tickers that already have artifacts
    if args.skip_trained:
        already = [t for t in tickers if is_trained(t)]
        if already:
            console.print(f"[dim]Sudah terlatih, dilewati: {', '.join(already)}[/dim]")
        tickers = [t for t in tickers if not is_trained(t)]

    if not tickers:
        console.print("[green]Tidak ada ticker yang perlu dilatih.[/green]")
        return 0

    if args.fast:
        cfg["model"]["epochs"] = 30
        cfg["model"]["n_splits_cv"] = 3
        console.print("[yellow]--fast aktif: epochs=30, n_splits_cv=3[/yellow]")

    batch_size = int(cfg["data"].get("fetch_batch_size", 10))
    batch_pause = float(cfg["data"].get("fetch_batch_pause", 12.0))

    console.print(
        f"[bold]Melatih {len(tickers)} ticker:[/bold] {', '.join(tickers)}\n"
    )

    rows: list[dict] = []
    counts = {"TRAINED": 0, "SKIP": 0, "FAIL": 0}
    start = time.time()
    interrupted = False

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
            task = progress.add_task("memulai...", total=len(tickers))
            for i, ticker in enumerate(tickers):
                progress.update(
                    task,
                    description=(
                        f"[cyan]{ticker}[/cyan] "
                        f"(✅{counts['TRAINED']} ⏭{counts['SKIP']} ❌{counts['FAIL']})"
                    ),
                )
                t0 = time.time()
                auc = f1 = float("nan")
                err = ""
                try:
                    metrics = run_train(cfg, ticker=ticker)
                    status = "TRAINED"
                    auc = metrics.get("auc_mean", float("nan"))
                    f1 = metrics.get("f1_mean", float("nan"))
                except InsufficientDataError as exc:
                    status = "SKIP"
                    err = str(exc)
                    log.warning(f"SKIP: {ticker} — {exc}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    status = "FAIL"
                    err = str(exc)
                    log.error(f"FAIL: {ticker} — {exc}")

                counts[status] += 1
                rows.append(
                    {
                        "ticker": ticker,
                        "status": status,
                        "auc_mean": f"{auc:.4f}" if np.isfinite(auc) else "",
                        "f1_mean": f"{f1:.4f}" if np.isfinite(f1) else "",
                        "duration_minutes": round((time.time() - t0) / 60.0, 2),
                        "error_msg": err,
                    }
                )
                progress.advance(task)

                # Batch pause to stay friendly with Yahoo Finance
                if (i + 1) % batch_size == 0 and (i + 1) < len(tickers):
                    log.info(f"Jeda batch {batch_pause}s ...")
                    time.sleep(batch_pause)
    except KeyboardInterrupt:
        interrupted = True
        console.print(
            "\n[yellow]Dihentikan oleh user — menyimpan laporan progres...[/yellow]"
        )

    total_s = time.time() - start

    # --- Report CSV -------------------------------------------------------- #
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = project_root() / "logs" / f"train_all_report_{ts}.csv"
    ensure_dir(report_path.parent)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker", "status", "auc_mean", "f1_mean",
                "duration_minutes", "error_msg",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # --- Summary panel ------------------------------------------------------ #
    skipped = [r["ticker"] for r in rows if r["status"] == "SKIP"]
    failed = [r["ticker"] for r in rows if r["status"] == "FAIL"]

    body = Text()
    body.append(f"Trained    : {counts['TRAINED']}/{len(tickers)}\n", style="green")
    body.append(
        f"Skipped    : {counts['SKIP']}"
        + (f"  ({', '.join(skipped)})" if skipped else "")
        + "\n",
        style="yellow",
    )
    body.append(
        f"Failed     : {counts['FAIL']}"
        + (f"  ({', '.join(failed)})" if failed else "")
        + "\n",
        style="red" if failed else "white",
    )
    body.append(f"Total time : {_fmt_duration(total_s)}\n")
    body.append(f"Report     : {report_path}\n", style="dim")
    if interrupted:
        body.append(
            "\nTERPUTUS — lanjutkan dengan: python train_all.py --skip-trained\n",
            style="bold yellow",
        )

    console.print(Panel(body, title="[bold cyan]Training Summary[/bold cyan]", padding=(1, 2)))
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
