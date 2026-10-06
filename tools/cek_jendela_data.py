#!/usr/bin/env python
"""Sidik jari jendela data: apakah setiap backtest_*.csv dihasilkan dari snapshot?

Jendela data = 5000 candle terakhir pada saat fetch, sehingga candle terakhir
sebuah backtest = raw[-1-lookahead] (baris berlabel kosong terbuang). Sebuah
backtest memakai snapshot JIKA DAN HANYA JIKA candle terakhirnya sama dengan
snapshot[-1-lookahead] dan harga close setiap barisnya (close candle sebelumnya,
akibat penggeseran) cocok dengan snapshot.

Usage:
    python tools/cek_jendela_data.py                       # periksa logs/
    python tools/cek_jendela_data.py --dir logs/archive_pra_snapshot
    python tools/cek_jendela_data.py --semua-harus-cocok   # kode keluar 1 bila ada yang tidak
"""
from __future__ import annotations

import argparse
import re
import sys
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

from utils.helpers import load_config  # noqa: E402

PAT = re.compile(r"backtest_([A-Z]+)_USD_(.+)_(\d{8}_\d{6})\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="logs", help="Folder berisi backtest_*.csv")
    ap.add_argument("--config", default="config_snapshot_20260912.yaml")
    ap.add_argument("--semua-harus-cocok", action="store_true",
                    help="Keluar dengan kode 1 bila ada backtest yang bukan dari snapshot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    snap_dir = ROOT / cfg["data"]["cache_dir"]
    look = int(cfg["features"]["lookahead_n"])
    limit, tf = cfg["data"]["candle_limit"], cfg["data"]["primary_tf"]
    folder = ROOT / args.dir

    snaps: dict[str, pd.DataFrame] = {}
    rows = []
    for p in sorted(folder.glob("backtest_*_USD_*_*.csv")):
        m = PAT.search(p.name)
        if not m:
            continue
        coin, tag, stamp = m.groups()
        sp = snap_dir / f"ohlcv_yf_{coin}_USD_{tf}_{limit}.parquet"
        if not sp.exists():
            continue
        if coin not in snaps:
            snaps[coin] = pd.read_parquet(sp)
        s = snaps[coin]
        expected_last = s.index[-1 - look]
        bt = pd.read_csv(p)
        idx = pd.to_datetime(bt["timestamp"], utc=True)
        in_snap = match = 0
        for T, c in zip(idx, bt["close"]):
            if T in s.index:
                k = s.index.get_loc(T)
                if k >= 1:
                    in_snap += 1
                    match += abs(s["close"].iloc[k - 1] - c) <= 1e-9 * max(1.0, abs(c))
        last = idx.iloc[-1]
        ok = last == expected_last and match == in_snap == len(bt)
        rows.append((datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc),
                     coin, tag, last, (expected_last - last).total_seconds() / 3600,
                     match, in_snap, ok))

    rows.sort()
    print(f"Folder: {folder}   snapshot: {snap_dir.name}   berkas: {len(rows)}\n")
    print(f"{'berkas (UTC)':16} {'koin':5} {'tag':18} {'candle terakhir':17} {'selisih':>9}  "
          f"{'close cocok':>11}  snapshot?")
    print("-" * 96)
    for f_utc, coin, tag, last, lag, match, in_snap, ok in rows:
        print(f"{f_utc:%Y-%m-%d %H:%M} {coin:5} {tag[:18]:18} {last:%Y-%m-%d %H:%M}  "
              f"{lag:+8.2f}j  {match:>4}/{in_snap:<4}    {'YA' if ok else 'tidak'}")
    n_ok = sum(r[-1] for r in rows)
    print(f"\nTepat pada snapshot: {n_ok}/{len(rows)}")
    if args.semua_harus_cocok and n_ok != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
