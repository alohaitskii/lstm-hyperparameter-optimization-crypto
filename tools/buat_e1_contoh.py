#!/usr/bin/env python
"""Hasilkan bahan_bab4/E1_contoh.md: contoh baris BTC-USD dan bahan pendukung E1.

Ringan: tidak ada pelatihan. Data dibaca lewat config_snapshot_20260912.yaml
(mode cache-only). Semua angka diambil dari fungsi pipeline yang sama dengan
eksperimen (build_features, add_target, apply_anti_leakage, prepare_for_training,
split_chronological, fit_scaler, create_sequences, sequence_window), lalu
diperiksa silang dengan perhitungan manual dari data mentah.

Usage:
    python tools/buat_e1_contoh.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.indicators as ind  # noqa: E402
from data.fetcher import fetch_all, require_cached_inputs  # noqa: E402
from data.preprocessor import (  # noqa: E402
    FEATURE_COLUMNS,
    FGI_ORDINAL,
    _fit_fgi_encoder,
    add_target,
    apply_anti_leakage,
    build_features,
    create_sequences,
    fit_scaler,
    prepare_for_training,
    sequence_window,
    split_chronological,
)
from optimization.search_space import SearchSpace  # noqa: E402
from utils.helpers import load_config  # noqa: E402

CONFIG = "config_snapshot_20260912.yaml"
OUT = ROOT / "bahan_bab4" / "E1_contoh.md"
TICKER = "BTC-USD"
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
WIN_L = 60
M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)


def fid(x: float, dec: int) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fraw(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return fid(v, 0)
    if a >= 1000:
        return fid(v, 2)
    if a >= 1:
        return fid(v, 4)
    return fid(v, 6)


def ts(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def main() -> int:
    cfg = load_config(CONFIG)
    assert cfg["data"]["cache_only"] is True, "harus mode cache-only"
    require_cached_inputs(cfg, [TICKER])
    f = cfg["features"]
    N, thr = int(f["lookahead_n"]), float(f["move_threshold"])
    space = SearchSpace.from_config(cfg)
    L_CAND = [int(v) for v in space.choices("sequence_length")]
    L_MAX = max(L_CAND)

    bundle = fetch_all(cfg, TICKER)
    raw15, raw1h, fgi = bundle["primary"], bundle["secondary"], bundle["fgi"]
    feat_df, _ = build_features(bundle, cfg)
    labeled = add_target(feat_df, N, thr)
    clean = apply_anti_leakage(labeled[FEATURE_COLUMNS + ["target"]])
    prepped = prepare_for_training(bundle, cfg)
    X_all, y_all = prepped["X"], prepped["y"]
    assert len(clean) == len(X_all) and np.array_equal(
        clean[FEATURE_COLUMNS].values.astype(np.float32), X_all), "clean != pipeline"

    # Candle 15m terakhir: bukan baris fitur, hanya close(t+N) bagi baris bersih terakhir
    assert clean.index[-1] == raw15.index[-1 - N]
    assert raw15.index[-1] not in clean.index
    # Bar 1 jam terakhir tutup sesudah baris bersih terakhir diketahui -> tak pernah dipakai
    assert raw1h.index[-1] + H1 > clean.index[-1]

    tr, va, te = split_chronological(len(clean))
    train = clean.iloc[tr]
    scaler = fit_scaler(X_all[tr])

    # 1h indikator per bar (stempel mulai), sama persis dengan _add_1h_features
    sec = raw1h.copy()
    sec["rsi_1h"] = ind.rsi(sec["close"], length=14)
    sec["ema_21_1h"] = ind.ema(sec["close"], length=21)
    sec["macd_hist_1h"] = ind.macd(sec["close"], fast=12, slow=26, signal=9)[2]

    # ---- pemilihan baris ------------------------------------------------- #
    pos = np.arange(len(train))
    ok = pos >= L_MAX - 1
    y_tr = train["target"].values
    i1 = int(pos[ok & (y_tr == 1)][0])
    i0 = int(pos[ok & (y_tr == 0)][0])

    def contoh(i: int) -> dict:
        t = train.index[i]
        k = raw15.index.get_loc(t)
        c_t, c_t4 = float(raw15["close"].iloc[k]), float(raw15["close"].iloc[k + N])
        lab = int(c_t4 > c_t * (1 + thr))
        assert lab == int(train["target"].iloc[i]), "label manual != pipeline"
        r = raw15.index[k - 1]                      # candle asal fitur baris t
        assert r == t - M15, "candle asal bukan t-15 menit"
        assert np.allclose(train.iloc[i][FEATURE_COLUMNS].values.astype(float),
                           feat_df.loc[r, FEATURE_COLUMNS].values.astype(float),
                           rtol=0, atol=0, equal_nan=False), "fitur baris t != fitur candle t-15m"
        # Bar 1 jam: terakhir yang sudah tutup saat baris r diketahui (r + 15m = t)
        closed = sec.index[sec.index + H1 <= r + M15]
        h = closed[-1]
        for c in ("rsi_1h", "ema_21_1h", "macd_hist_1h"):
            assert train.iloc[i][c] == sec.loc[h, c], f"{c} != bar 1 jam {h}"
        nxt = sec.index[sec.index > h][0]
        # FGI: tanggal UTC candle asal (ffill ke tanggal terakhir yang tersedia)
        fd = fgi.index[fgi.index <= r.normalize()][-1]
        fv, fc = int(fgi.loc[fd, "fgi_value"]), str(fgi.loc[fd, "fgi_classification"])
        code = _fit_fgi_encoder().encode(fc)
        assert np.isclose(train.iloc[i]["fgi_normalized"], fv / 100.0)
        assert train.iloc[i]["fgi_classification_encoded"] == code
        # Jendela L = 60 untuk y(t): baris i-59 .. i pada bagian latih
        Xs = scaler.transform(X_all[tr])
        w = sequence_window(Xs, i, WIN_L)
        assert np.array_equal(w[-1], Xs[i]) and np.array_equal(w[0], Xs[i - WIN_L + 1])
        raw_vals = train.iloc[i][FEATURE_COLUMNS].values.astype(np.float64)
        norm = scaler.transform(X_all[tr][i:i + 1])[0]
        return {"i": i, "t": t, "t4": raw15.index[k + N], "c_t": c_t, "c_t4": c_t4,
                "thr": c_t * (1 + thr), "chg": 100.0 * (c_t4 / c_t - 1), "lab": lab,
                "r": r, "raw": raw_vals, "norm": norm, "h": h, "h_close": h + H1,
                "h_next_close": nxt + H1, "fd": fd, "fv": fv, "fc": fc, "code": code,
                "w0": train.index[i - WIN_L + 1], "w1": t}

    e1, e0 = contoh(i1), contoh(i0)

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "tidak diketahui"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L: list[str] = []
    P = L.append
    P("# Bahan E1 — contoh baris dan bahan pendukung (BTC-USD)\n")
    P(f"Dihasilkan oleh `tools/buat_e1_contoh.py` pada {now} dari kode commit `{head}`. "
      f"Data: snapshot `data/snapshot_20260912/` lewat `{CONFIG}` (cache-only, diverifikasi "
      "SHA256 terhadap `MANIFEST.md`). Tanpa pelatihan model. Waktu dalam UTC; angka memakai "
      "titik ribuan dan koma desimal.\n")

    # ------------------------------------------------------------------ 1
    P("## 1. Contoh baris BTC-USD\n")
    P(f"Aturan pemilihan (deterministik): baris **pertama** berlabel 1 dan baris **pertama** "
      f"berlabel 0 pada bagian latih dengan indeks ≥ {L_MAX - 1} (indeks 0-based di dalam "
      f"bagian latih). Dengan begitu kedua baris punya jendela penuh dan menjadi target "
      f"y(t) untuk semua kandidat panjang sekuens ({', '.join(map(str, L_CAND))}). Bagian "
      f"latih = {fid(len(train), 0)} baris ({ts(train.index[0])} – {ts(train.index[-1])}).\n")
    P("| | Contoh berlabel 1 | Contoh berlabel 0 |")
    P("|---|---:|---:|")
    P(f"| Indeks di bagian latih | {e1['i']} | {e0['i']} |")
    P(f"| Indeks di data bersih | {e1['i'] + tr.start} | {e0['i'] + tr.start} |")
    P("")

    P("### 1a. Pembentukan label\n")
    P(f"Label 1 bila close(t+{N}) > close(t) × {fid(1 + thr, 3)}.\n")
    P("| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |")
    P("|---|---:|---:|")
    P(f"| Timestamp t | {ts(e1['t'])} | {ts(e0['t'])} |")
    P(f"| close(t) | {fid(e1['c_t'], 2)} | {fid(e0['c_t'], 2)} |")
    P(f"| Timestamp t+{N} | {ts(e1['t4'])} | {ts(e0['t4'])} |")
    P(f"| close(t+{N}) | {fid(e1['c_t4'], 2)} | {fid(e0['c_t4'], 2)} |")
    P(f"| close(t) × {fid(1 + thr, 3)} | {fid(e1['thr'], 2)} | {fid(e0['thr'], 2)} |")
    P(f"| Perubahan close(t+{N}) terhadap close(t) | {fid(e1['chg'], 4)}% | {fid(e0['chg'], 4)}% |")
    P(f"| close(t+{N}) > close(t) × {fid(1 + thr, 3)}? | {'ya' if e1['lab'] else 'tidak'} "
      f"| {'ya' if e0['lab'] else 'tidak'} |")
    P(f"| **Label** | **{e1['lab']}** | **{e0['lab']}** |")
    P("")

    P("### 1b. Nilai 25 fitur pada baris t\n")
    P(f"Setelah penggeseran satu langkah (`apply_anti_leakage`), baris t berisi fitur candle "
      f"**t − 15 menit**: {ts(e1['r'])} (contoh label 1) dan {ts(e0['r'])} (contoh label 0). "
      "Ke-25 nilai diperiksa identik dengan matriks fitur pada candle itu. Kolom *norm.* = "
      "MinMaxScaler [0, 1] yang di-fit hanya pada bagian latih (lihat bagian 2); model "
      "menerima nilai float32.\n")
    P("| # | Fitur | Candle asal | Label 1: mentah | Label 1: norm. | Candle asal | "
      "Label 0: mentah | Label 0: norm. |")
    P("|---:|---|---|---:|---:|---|---:|---:|")
    for j, c in enumerate(FEATURE_COLUMNS):
        P(f"| {j + 1} | `{c}` | {ts(e1['r'])} | {fraw(e1['raw'][j])} | {fid(e1['norm'][j], 6)} "
          f"| {ts(e0['r'])} | {fraw(e0['raw'][j])} | {fid(e0['norm'][j], 6)} |")
    P("\nTiga fitur 1 jam berasal dari bar 1 jam pada 1c, dan dua fitur FGI dari nilai harian "
      "pada 1d. Keduanya sudah melekat pada baris candle asal sebelum penggeseran.\n")

    P("### 1c. Bar 1 jam yang dipakai baris t\n")
    P("Yahoo menstempel bar 1 jam dengan waktu **mulai** h; bar itu baru diketahui saat tutup "
      "pada h + 1 jam. Candle asal r diketahui saat tutup pada r + 15 menit (= t), sehingga "
      "dipakai bar 1 jam terakhir dengan h + 1 jam ≤ t.\n")
    P("| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |")
    P("|---|---:|---:|")
    P(f"| Candle asal r (diketahui r + 15 m) | {ts(e1['r'])} ({ts(e1['t'])}) | {ts(e0['r'])} ({ts(e0['t'])}) |")
    P(f"| Stempel bar 1 jam h | {ts(e1['h'])} | {ts(e0['h'])} |")
    P(f"| Waktu tutup h + 1 jam | {ts(e1['h_close'])} | {ts(e0['h_close'])} |")
    P(f"| Bar berikutnya tutup pada (belum boleh dipakai) | {ts(e1['h_next_close'])} | {ts(e0['h_next_close'])} |")
    P(f"| `rsi_1h` | {fraw(e1['raw'][FEATURE_COLUMNS.index('rsi_1h')])} | {fraw(e0['raw'][FEATURE_COLUMNS.index('rsi_1h')])} |")
    P(f"| `ema_21_1h` | {fraw(e1['raw'][FEATURE_COLUMNS.index('ema_21_1h')])} | {fraw(e0['raw'][FEATURE_COLUMNS.index('ema_21_1h')])} |")
    P(f"| `macd_hist_1h` | {fraw(e1['raw'][FEATURE_COLUMNS.index('macd_hist_1h')])} | {fraw(e0['raw'][FEATURE_COLUMNS.index('macd_hist_1h')])} |")
    P("\nKetiga nilai diperiksa identik dengan indikator yang dihitung langsung dari bar 1 jam h.\n")

    P("### 1d. Fear & Greed Index\n")
    P("FGI harian dipetakan ke candle berdasarkan tanggal UTC candle asal (nilai tanggal "
      "terakhir yang tersedia). `fgi_normalized` = nilai / 100; kode ordinal: "
      + ", ".join(f"{k} = {v}" for k, v in FGI_ORDINAL.items()) + ", lainnya = −1.\n")
    P("| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |")
    P("|---|---:|---:|")
    P(f"| Tanggal FGI (UTC) | {e1['fd'].strftime('%Y-%m-%d')} | {e0['fd'].strftime('%Y-%m-%d')} |")
    P(f"| Nilai mentah (0–100) | {e1['fv']} | {e0['fv']} |")
    P(f"| Kategori | {e1['fc']} | {e0['fc']} |")
    P(f"| Kode ordinal (`fgi_classification_encoded`) | {e1['code']} | {e0['code']} |")
    P(f"| `fgi_normalized` | {fid(e1['fv'] / 100, 2)} | {fid(e0['fv'] / 100, 2)} |")
    P("")

    P(f"### 1e. Jendela L = {WIN_L} untuk target y(t)\n")
    P(f"`sequence_window(X, i, {WIN_L})` = baris i − {WIN_L - 1} … i pada bagian latih (baris t "
      "ikut). Karena penggeseran, isi fiturnya berasal dari candle 15 menit lebih awal.\n")
    P("| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |")
    P("|---|---:|---:|")
    P(f"| Baris pertama jendela | {ts(e1['w0'])} | {ts(e0['w0'])} |")
    P(f"| Baris terakhir jendela (= t) | {ts(e1['w1'])} | {ts(e0['w1'])} |")
    P(f"| Candle asal fitur: pertama – terakhir | {ts(e1['w0'] - M15)} – {ts(e1['w1'] - M15)} "
      f"| {ts(e0['w0'] - M15)} – {ts(e0['w1'] - M15)} |")
    P(f"| Indeks bagian latih | {e1['i'] - WIN_L + 1} – {e1['i']} | {e0['i'] - WIN_L + 1} – {e0['i']} |")
    P("")

    # ------------------------------------------------------------------ 2
    P("## 2. Minimum dan maksimum bagian latih BTC-USD (dasar MinMaxScaler)\n")
    P(f"`fit_scaler(X_latih)`, X_latih = {fid(len(train), 0)} baris × 25 fitur (float32). "
      "x_norm = (x − min) / (maks − min).\n")
    P("| # | Fitur | Minimum | Maksimum | Maks − min |")
    P("|---:|---|---:|---:|---:|")
    mn, mx = scaler.data_min_.astype(np.float64), scaler.data_max_.astype(np.float64)
    for j, c in enumerate(FEATURE_COLUMNS):
        P(f"| {j + 1} | `{c}` | {fraw(mn[j])} | {fraw(mx[j])} | {fraw(mx[j] - mn[j])} |")
    j = FEATURE_COLUMNS.index("close")
    x = float(X_all[tr][e1["i"], j])
    manual = (x - mn[j]) / (mx[j] - mn[j])
    P(f"\n**Contoh perhitungan** (`close`, contoh berlabel 1, candle asal {ts(e1['r'])}):\n")
    P(f"x_norm = ({fid(x, 4)} − {fid(mn[j], 4)}) / ({fid(mx[j], 4)} − {fid(mn[j], 4)}) "
      f"= {fid(x - mn[j], 4)} / {fid(mx[j] - mn[j], 4)} = **{fid(manual, 6)}**\n")
    P(f"Keluaran `scaler.transform`: {fid(float(e1['norm'][j]), 6)} (selisih "
      f"{abs(manual - float(e1['norm'][j])):.1e}, pembulatan float32).\n")

    # ------------------------------------------------------------------ 3
    P("## 3. Pengambilan data (dari `MANIFEST.md`)\n")
    man = (ROOT / cfg["data"]["cache_dir"] / "MANIFEST.md").read_text(encoding="utf-8")
    m_dibuat = re.search(r"Dibuat:\s*(.+?)\s*$", man, re.M).group(1)
    rows = {}
    for line in man.splitlines():
        p = [x.strip().strip("`") for x in line.strip().strip("|").split("|")]
        if len(p) == 7 and p[0].endswith(".parquet"):
            rows[p[0]] = p
    P("Waktu pengambilan = mtime asli berkas cache saat diunduh dari Yahoo Finance "
      "(MANIFEST mencatatnya dalam WIB, UTC+7). Snapshot sendiri disalin dari cache pada "
      f"{m_dibuat}.\n")
    P("| Koin | Diambil (WIB) | Diambil (UTC) | Candle 15m pertama | Candle 15m terakhir "
      "| Candle 15m terakhir tutup | Bar 1 jam pertama | Bar 1 jam terakhir |")
    P("|---|---|---|---|---|---|---|---|")
    open_last = []
    for t in TICKERS:
        s = t.replace("-", "_")
        a = rows[f"ohlcv_yf_{s}_15m_5000.parquet"]
        b = rows[f"ohlcv_yf_{s}_1h_5000.parquet"]
        assert a[5] == b[5], (t, a[5], b[5])
        got = pd.Timestamp(a[5]) - pd.Timedelta(hours=7)
        last_close = pd.Timestamp(a[4]) + M15
        if got < last_close:
            open_last.append(t)
        P(f"| {t} | {a[5]} | {got.strftime('%Y-%m-%d %H:%M')} | {a[3]} | {a[4]} "
          f"| {last_close.strftime('%Y-%m-%d %H:%M')} | {b[3]} | {b[4]} |")
    g = rows["fgi_90.parquet"]
    utc = (pd.Timestamp(g[5]) - pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")
    P(f"\nFear & Greed Index: {g[2]} nilai harian {g[3]} – {g[4]}, diambil {g[5]} WIB ({utc} UTC).\n")
    P("Stempel candle adalah waktu **mulai** candle. Kecocokan candle terakhir 15m di tabel ini "
      "dengan isi snapshot diperiksa oleh `data/fetcher.py` setiap kali berkas dimuat.\n")
    if open_last:
        P(f"**Catatan.** Pada {len(open_last)} dari {len(TICKERS)} koin ({', '.join(open_last)}), "
          "waktu pengambilan (mtime) lebih awal daripada waktu tutup candle 15m terakhir, "
          "sehingga candle terakhir kemungkinan belum lengkap saat diunduh. Candle itu tidak "
          "pernah menjadi baris fitur (penggeseran membuang baris sesudahnya) dan tidak "
          f"berlabel; satu-satunya pemakaiannya adalah sebagai close(t+{N}) untuk label baris "
          f"bersih terakhir (t = candle terakhir − {N} × 15 menit), yang berada di bagian uji, "
          "jendela backtest, dan fold walk-forward terakhir. Bar 1 jam terakhir juga belum tutup, "
          "tetapi tidak pernah dipakai karena penyelarasan hanya mengambil bar yang sudah tutup.\n")

    # ------------------------------------------------------------------ 4
    P("## 4. Jumlah sekuens BTC-USD per kandidat panjang sekuens\n")
    n_tr, n_va, n_te = tr.stop - tr.start, va.stop - va.start, te.stop - te.start
    P(f"Pembagian kronologis 70/15/15 pada {fid(len(clean), 0)} baris bersih: latih "
      f"{fid(n_tr, 0)}, validasi {fid(n_va, 0)}, uji {fid(n_te, 0)} baris. Sekuens dibentuk di "
      "dalam tiap bagian (`create_sequences`), jumlahnya n − L + 1. Ini skema evaluasi fitness "
      "(pencarian) dan model pemenang.\n")
    P("| L | Sekuens latih | Sekuens validasi | % label 1 latih | % label 1 validasi |")
    P("|---:|---:|---:|---:|---:|")
    for Lc in L_CAND:
        Xa, ya = create_sequences(X_all[tr], y_all[tr], Lc)
        Xb, yb = create_sequences(X_all[va], y_all[va], Lc)
        assert len(Xa) == n_tr - Lc + 1 and len(Xb) == n_va - Lc + 1
        assert Xa.shape[1:] == (Lc, len(FEATURE_COLUMNS))
        P(f"| {Lc} | {fid(len(Xa), 0)} | {fid(len(Xb), 0)} | {fid(100 * ya.mean(), 2)}% "
          f"| {fid(100 * yb.mean(), 2)}% |")
    P("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Ditulis: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
