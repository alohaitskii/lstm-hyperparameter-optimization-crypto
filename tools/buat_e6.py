#!/usr/bin/env python
"""Hasilkan bahan_bab4/E6.md: blok walk-forward, jendela backtest, dan konfirmasi kode.

Ringan: tidak ada pelatihan. Data dibaca lewat config_snapshot_20260912.yaml
(cache-only). Indeks blok diambil dari TimeSeriesSplit yang sama dengan
model/validator.py; jendela backtest dari rumus main.run_backtest dan dicocokkan
dengan 45 berkas backtest reeval di logs/.

Usage:
    python tools/buat_e6.py
"""
from __future__ import annotations

import glob
import re
import subprocess
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from data.preprocessor import _fit_fgi_encoder, split_chronological  # noqa: E402
from optimization.experiment import prepare_ticker_data  # noqa: E402
from utils.helpers import load_config  # noqa: E402

CONFIG = "config_snapshot_20260912.yaml"
OUT = ROOT / "bahan_bab4" / "E6.md"
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
LOOKBACK = 200
M15 = pd.Timedelta(minutes=15)


def fid(x: float, dec: int) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def ts(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def pct(y) -> str:
    return fid(100.0 * float(np.mean(y)), 2) + "%"


def kutip(rel: str, start: str, end: str) -> str:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    i0 = next(i for i, l in enumerate(lines) if re.search(start, l))
    i1 = next(i for i in range(i0, len(lines)) if re.search(end, lines[i]))
    body = "\n".join(lines[i0:i1 + 1])
    return f"`{rel}` baris {i0 + 1}–{i1 + 1}:\n\n```python\n{body}\n```\n"


def main() -> int:
    cfg = load_config(CONFIG)
    assert cfg["data"]["cache_only"] is True
    n_splits = int(cfg["optimization"]["final_splits"])
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "tidak diketahui"

    D = {}
    for t in TICKERS:
        d = prepare_ticker_data(cfg, t)
        idx = d["index"] if "index" in d else None
        D[t] = d
        assert idx is not None and len(idx) == len(d["X"]), t

    L: list[str] = []
    P = L.append
    P("# Bahan E6 — walk-forward, jendela backtest, dan konfirmasi kode\n")
    P(f"Dihasilkan oleh `tools/buat_e6.py` pada {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
      f"dari kode commit `{head}`. Data: snapshot `data/snapshot_20260912/` (cache-only). "
      "Tanpa pelatihan. Waktu dalam UTC. Baris = baris bersih (setelah penggeseran dan "
      "pembuangan pemanasan/label kosong). % label 1 dihitung per baris; % per sekuens "
      "bergantung pada L (lihat `laporan_4.md` tabel c).\n")

    # ------------------------------------------------------------------ a
    P(f"## a. Blok walk-forward (`TimeSeriesSplit(n_splits={n_splits})`)\n")
    P("Indeks 0-based pada baris bersih; rentang inklusif. Blok latih selalu dimulai dari "
      "baris 0 (jendela mengembang); blok uji fold k langsung menyusul blok latihnya.\n")
    P("| Koin | Baris bersih | Fold | Blok | Indeks | Awal | Akhir | Baris | % label 1 |")
    P("|---|---:|---:|---|---|---|---|---:|---:|")
    folds_of = {}
    for t in TICKERS:
        d = D[t]
        idx, y = d["index"], d["y"]
        sp = list(TimeSeriesSplit(n_splits=n_splits).split(d["X"]))
        folds_of[t] = sp
        for k, (a, b) in enumerate(sp, 1):
            assert a[0] == 0 and b[0] == a[-1] + 1 and np.all(np.diff(a) == 1) and np.all(np.diff(b) == 1)
            for name, blk in (("latih", a), ("uji", b)):
                P(f"| {t} | {fid(len(y), 0)} | {k} | {name} | {blk[0]}–{blk[-1]} | {ts(idx[blk[0]])} "
                  f"| {ts(idx[blk[-1]])} | {fid(len(blk), 0)} | {pct(y[blk])} |")
    P("")

    # ------------------------------------------------------------------ b
    P(f"## b. Jendela backtest ({LOOKBACK} baris bersih terakhir)\n")
    P("Rumus `main.run_backtest`: indeks `max(L − 1, n − 200)` … `n − 1`. Karena n jauh lebih "
      "besar dari L, jendela = 200 baris terakhir untuk semua konfigurasi. Timestamp dicocokkan "
      "dengan seluruh 60 berkas backtest di `logs/` (per koin 3 dari tahap pencarian dan 9 "
      "dari reeval final).\n")
    P("| Koin | Indeks | Awal | Akhir | % label 1 | Di bagian uji 70/15/15? | Di blok uji fold 3? |")
    P("|---|---|---|---|---:|---|---|")
    fgi_rows = []
    enc = _fit_fgi_encoder()
    fgi = D["BTC-USD"]["bundle"]["fgi"]
    for t in TICKERS:
        d = D[t]
        idx, y, n = d["index"], d["y"], len(d["y"])
        w = np.arange(n - LOOKBACK, n)
        files = glob.glob(str(ROOT / "logs" / f"backtest_{t.replace('-', '_')}_*.csv"))
        assert len(files) == 12, (t, len(files))
        for f in files:
            bt = pd.read_csv(f)
            bts = pd.to_datetime(bt["timestamp"], utc=True)
            assert len(bt) == LOOKBACK and bts.iloc[0] == idx[w[0]] and bts.iloc[-1] == idx[w[-1]], f
            assert np.array_equal(bt["actual_label"].values, y[w].astype(int)), f
        _, _, te = split_chronological(n)
        in_te = w[0] >= te.start
        b3 = folds_of[t][-1][1]
        in_f3 = w[0] >= b3[0] and w[-1] <= b3[-1]
        P(f"| {t} | {w[0]}–{w[-1]} | {ts(idx[w[0]])} | {ts(idx[w[-1]])} | {pct(y[w])} "
          f"| {'ya' if in_te else 'tidak'} (uji mulai indeks {te.start}) "
          f"| {'ya' if in_f3 else 'tidak'} (indeks {b3[0]}–{b3[-1]}) |")
        # FGI yang dilihat model pada baris t = FGI tanggal UTC candle asal (t - 15 menit)
        src_dates = (idx[w] - M15).normalize()
        X_fgi = d["X"][w][:, -2:]
        for dt in sorted(set(src_dates)):
            fd = fgi.index[fgi.index <= dt][-1]
            v, c = int(fgi.loc[fd, "fgi_value"]), str(fgi.loc[fd, "fgi_classification"])
            m = np.asarray(src_dates == dt)
            assert np.allclose(X_fgi[m, 0], v / 100.0) and np.all(X_fgi[m, 1] == enc.encode(c)), (t, dt)
            fgi_rows.append((t, dt, fd, v, c, enc.encode(c), int(m.sum()),
                             idx[w][m][0], idx[w][m][-1]))
    P("")
    P("**FGI selama jendela backtest.** Nilai yang dilihat model pada baris t adalah FGI tanggal "
      "UTC candle asal (t − 15 menit), diperiksa identik dengan kolom fitur.\n")
    P("| Koin | Tanggal FGI | Nilai mentah | Kategori | Kode | Baris jendela | Baris t pertama – terakhir |")
    P("|---|---|---:|---|---:|---:|---|")
    for t, dt, fd, v, c, code, k, a, b in fgi_rows:
        P(f"| {t} | {fd.strftime('%Y-%m-%d')} | {v} | {c} | {code} | {k} | {ts(a)} – {ts(b)} |")
    P("")

    # ------------------------------------------------------------------ c
    P("## c. Konfirmasi dari kode\n")
    P("**1. Data walk-forward = seluruh baris bersih.** `run_method`/`reeval_final` memanggil "
      "`final_evaluate_and_save` dengan `data[\"X\"]`, `data[\"y\"]` dari `prepare_ticker_data` "
      "(= `prepare_for_training`: seluruh baris bersih, tanpa memotong bagian uji 70/15/15), "
      "lalu `walk_forward_validate(X_2d, y_1d, ...)` membagi seluruh matriks itu:\n")
    P(kutip("optimization/experiment.py", r"report = walk_forward_validate\(X_2d", r"report = walk_forward_validate\(X_2d"))
    P(kutip("model/validator.py", r"tscv = TimeSeriesSplit", r"for fold_idx, \(train_idx, val_idx\)"))
    P("| Koin | Baris bersih = baris walk-forward | Latih fold 1 + uji fold 1–3 |")
    P("|---|---:|---:|")
    for t in TICKERS:
        sp = folds_of[t]
        cover = len(sp[0][0]) + sum(len(b) for _, b in sp)
        P(f"| {t} | {fid(len(D[t]['y']), 0)} | {fid(cover, 0)} |")
    P("")
    P("**2. Early stopping memantau blok uji fold yang sama.** Di setiap fold, `train_model` "
      "menerima blok uji fold itu sebagai `validation_data`. `EarlyStopping` memantau `val_auc` "
      "(patience 10, `restore_best_weights=True`) dan `ReduceLROnPlateau` memantau `val_loss` "
      "pada blok itu, lalu metrik fold (termasuk AUC) dihitung pada blok yang sama:\n")
    P(kutip("model/validator.py", r"model = build_model\(\(seq_len, n_features\)", r"m = _metrics\(y_val_seq, y_val_prob\)"))
    P(kutip("model/lstm_model.py", r"EarlyStopping\(", r"ReduceLROnPlateau\(monitor"))
    P("Konsekuensinya, epoch terbaik dipilih dengan melihat blok yang juga dipakai untuk "
      "melaporkan metrik, sehingga AUC walk-forward cenderung **optimistis** (tidak ada blok "
      "validasi terpisah di dalam fold). Hal yang sama berlaku pada fitness pencarian: "
      "`fitness.evaluate` memakai bagian validasi 15% untuk early stopping sekaligus untuk "
      "AUC fitness. Ketiga metode (GA, Grid, Manual) diperlakukan sama, sehingga perbandingan "
      "antarmetode tetap setara, tetapi nilai absolutnya perlu dibaca sebagai batas atas.\n")
    P("**3. Jendela backtest.** Model backtest adalah model pemenang yang dilatih pada bagian "
      "latih 70% dengan early stopping pada bagian validasi 15% (`final_evaluate_and_save`, "
      "langkah 2). Jendela backtest (200 baris terakhir) berada di bagian uji 15% yang tidak "
      "dipakai pelatihan maupun early stopping model itu, dan sekaligus berada **di dalam blok "
      "uji fold 3** walk-forward (tabel b). Blok uji fold 3 dipakai early stopping model fold 3, "
      "tetapi model fold 3 tidak dipakai untuk backtest.\n")
    P(kutip("optimization/experiment.py", r"train_sl, val_sl, _ = split_chronological", r"eval_cfg\[\"model\"\], save_path=None, verbose=0"))

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Ditulis: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
