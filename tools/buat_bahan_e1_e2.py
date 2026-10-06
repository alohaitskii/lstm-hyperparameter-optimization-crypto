#!/usr/bin/env python
"""Hasilkan bahan_bab4/E1_E2.md (bahan subbab 4.1.1 dan 4.1.2) dari snapshot.

Ringan: tidak ada pelatihan. Data dibaca lewat config_snapshot_20260912.yaml
(mode cache-only, tanpa unduhan). Angka B7/B8 diambil dari jalur kode asli:
train_model() dipanggil sungguhan, hanya model.fit yang disadap agar tidak
melatih, sehingga argumen yang tercatat persis sama dengan saat eksperimen.

Usage:
    python tools/buat_bahan_e1_e2.py
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

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
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from data.fetcher import fetch_all, require_cached_inputs  # noqa: E402
from data.preprocessor import (  # noqa: E402
    FEATURE_COLUMNS,
    add_target,
    apply_anti_leakage,
    build_features,
    create_sequences,
    fit_scaler,
    prepare_for_training,
    split_chronological,
)
from utils.helpers import load_config  # noqa: E402

CONFIG = "config_snapshot_20260912.yaml"
OUT = ROOT / "bahan_bab4" / "E1_E2.md"
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
BASELINE_CHROM = [2, 3, 3, 1, 2, 1]


# --------------------------------------------------------------------------- #
# Format angka gaya Indonesia: titik ribuan, koma desimal
# --------------------------------------------------------------------------- #
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


def pct(n: int, d: int) -> str:
    return fid(100.0 * n / d, 1) + "%" if d else "-"


def ts(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Kutipan kode apa adanya (pola dicari saat dijalankan; nomor baris aktual)
# --------------------------------------------------------------------------- #
def kutip(rel: str, start: str, end: str, extra: int = 0) -> str:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    i0 = next(i for i, l in enumerate(lines) if re.search(start, l))
    i1 = next(i for i in range(i0, len(lines)) if re.search(end, lines[i])) + extra
    lang = "yaml" if rel.endswith(".yaml") else "python"
    body = "\n".join(lines[i0:i1 + 1])
    return f"`{rel}` baris {i0 + 1}–{i1 + 1}:\n\n```{lang}\n{body}\n```\n"


# --------------------------------------------------------------------------- #
# A. Data
# --------------------------------------------------------------------------- #
def analisis_koin(cfg: dict, ticker: str) -> dict:
    bundle = fetch_all(cfg, ticker)  # cache-only: membaca snapshot
    raw15 = bundle["primary"]

    # Bukti: data yang dibaca = snapshot persis
    snap = pd.read_parquet(
        ROOT / cfg["data"]["cache_dir"] / f"ohlcv_yf_{ticker.replace('-', '_')}_15m_5000.parquet"
    )
    assert raw15.equals(snap), f"{ticker}: data terbaca != snapshot"

    feat_df, _ = build_features(bundle, cfg)
    zero_sma = int((feat_df["vol_sma_20"] == 0).sum())
    f = cfg["features"]
    labeled = add_target(feat_df, f["lookahead_n"], f["move_threshold"])
    sub = labeled[FEATURE_COLUMNS + ["target"]]

    # Pisahkan alasan pembuangan baris persis seperti apply_anti_leakage
    shifted = sub.copy()
    shifted[FEATURE_COLUMNS] = shifted[FEATURE_COLUMNS].shift(1)
    nan_feat = shifted[FEATURE_COLUMNS].isna().any(axis=1)
    no_label = shifted["target"].isna()
    assert not (nan_feat & no_label).any(), "baris pemanasan & label kosong tumpang tindih"
    first_ok = int(np.argmax(~nan_feat.values))
    lead = first_ok                                  # pemanasan (awal)
    interior = int(nan_feat.iloc[first_ok:].sum())   # NaN di tengah data
    lead_by_feat = {
        c: int(np.argmax(shifted[c].notna().values)) for c in FEATURE_COLUMNS
    }
    mid_nan = shifted[FEATURE_COLUMNS].iloc[first_ok:].isna()
    interior_cause = {
        c: int(v) for c, v in mid_nan[mid_nan.any(axis=1)].sum().items() if v
    }

    clean = apply_anti_leakage(sub)
    prepped = prepare_for_training(bundle, cfg)
    assert len(clean) == len(prepped["X"]), "hitungan baris berbeda dari pipeline"

    # Baris yang dibuang di tengah membuat baris bersih tidak lagi berurutan
    # 15 menit; hitung jendela sekuens yang melompati candle karenanya.
    seq_len = int(f["sequence_length"])
    gap = (clean.index.to_series().diff() > pd.Timedelta(minutes=15)).values
    cg = np.concatenate([[0], np.cumsum(gap)])
    win_gap = sum(1 for i in range(seq_len, len(clean)) if cg[i] - cg[i - seq_len + 1] > 0)

    n = len(clean)
    tr, va, te = split_chronological(n)
    parts = {"Latih": clean.iloc[tr], "Validasi": clean.iloc[va], "Uji": clean.iloc[te]}
    return {
        "ticker": ticker, "bundle": bundle, "raw15": raw15, "clean": clean,
        "prepped": prepped, "n_raw": len(sub), "lead": lead, "interior": interior,
        "n_nolabel": int(no_label.sum()), "lead_by_feat": lead_by_feat,
        "interior_cause": interior_cause,
        "vol_zero": int((raw15["volume"] == 0).sum()),
        "zero_sma": zero_sma,
        "n_gaps": int(gap.sum()), "win_gap": win_gap, "n_win": n - seq_len,
        "slices": (tr, va, te), "parts": parts,
    }


def main() -> int:
    cfg = load_config(CONFIG)
    assert cfg["data"]["cache_only"] is True, "harus mode cache-only"
    require_cached_inputs(cfg, TICKERS)
    f = cfg["features"]
    seq_len = int(f["sequence_length"])
    L: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "tidak diketahui"

    L.append("# Bahan Bab 4 — E1 dan E2 (subbab 4.1.1 dan 4.1.2)\n")
    L.append(
        f"Dihasilkan oleh `tools/buat_bahan_e1_e2.py` pada {now} dari kode pipeline commit "
        f"`{head}`. Sumber data: snapshot "
        "`data/snapshot_20260912/` (lihat `MANIFEST.md`), dibaca dalam mode cache-only "
        f"lewat `{CONFIG}` — tidak ada pengunduhan. Setiap berkas yang terbaca diverifikasi "
        "identik dengan snapshot. Tidak ada pelatihan model dalam pembuatan dokumen ini.\n"
    )
    L.append("Konvensi: waktu dalam UTC; angka memakai titik pemisah ribuan dan koma desimal.\n")

    data = {t: analisis_koin(cfg, t) for t in TICKERS}

    # ------------------------------------------------------------------ A.1
    L.append("## A. Bahan 4.1.1 — Data\n")
    L.append("### A.1 Jumlah baris per tahap\n")
    L.append("| Koin | Mentah 15m | Pemanasan dibuang | Setelah pemanasan | Kosong di tengah dibuang "
             "| Label kosong dibuang | Bersih | Latih (70%) | Validasi (15%) | Uji (15%) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t, d in data.items():
        tr, va, te = d["slices"]
        n = len(d["clean"])
        L.append(
            f"| {t} | {fid(d['n_raw'], 0)} | {d['lead']} | {fid(d['n_raw'] - d['lead'], 0)} "
            f"| {d['interior']} | {d['n_nolabel']} | {fid(n, 0)} | {fid(tr.stop - tr.start, 0)} "
            f"| {fid(va.stop - va.start, 0)} | {fid(te.stop - te.start, 0)} |"
        )
    btc = data["BTC-USD"]
    top = sorted(btc["lead_by_feat"].items(), key=lambda kv: -kv[1])[:3]
    L.append("")
    L.append(
        f"Baris pemanasan adalah baris awal yang masih memuat nilai kosong setelah fitur "
        f"digeser satu langkah. Fitur dengan pemanasan terpanjang (BTC-USD): "
        + ", ".join(f"`{c}` {k} baris" for c, k in top)
        + f". Jadi {top[0][1]} baris pertama dibuang: indikator `{top[0][0]}` membutuhkan "
        f"{top[0][1] - 1} baris riwayat, ditambah 1 baris karena penggeseran. "
        f"Label kosong = {f['lookahead_n']} baris terakhir, karena close(t+{f['lookahead_n']}) "
        f"belum tersedia."
    )
    L.append(
        "\n**Volume nol.** Yahoo Finance melaporkan volume 0 untuk sebagian candle 15m. Bila 20 "
        "candle berturut-turut bervolume 0, SMA-20 volume bernilai 0; pada baris itu `vol_ratio` "
        "didefinisikan 0 (bukan dibiarkan kosong), tanpa imputasi lain. Karena itu tidak ada baris "
        "yang dibuang di tengah data, dan baris bersih tetap berurutan 15 menit tanpa celah. "
        "Indeks waktu data mentah sendiri lengkap.\n"
    )
    L.append("| Koin | Candle volume = 0 | Baris SMA-20 volume = 0 (`vol_ratio` = 0) "
             "| Baris kosong di tengah | Celah pada baris bersih |")
    L.append("|---|---:|---:|---:|---:|")
    for t, d in data.items():
        L.append(
            f"| {t} | {fid(d['vol_zero'], 0)} ({pct(d['vol_zero'], d['n_raw'])}) "
            f"| {d['zero_sma']} | {d['interior']} | {d['n_gaps']} |"
        )
    L.append("")
    L.append(
        f"\nPembagian 70/15/15 memakai `split_chronological` (kronologis, tanpa pengacakan): "
        f"uji = int(0,15·n), validasi = int(0,15·n), latih = sisanya.\n"
    )

    # ------------------------------------------------------------------ A.2
    L.append("### A.2 Rentang waktu tiap bagian\n")
    L.append("| Koin | Bagian | Awal | Akhir | Baris |")
    L.append("|---|---|---|---|---:|")
    for t, d in data.items():
        for name, part in d["parts"].items():
            L.append(f"| {t} | {name} | {ts(part.index[0])} | {ts(part.index[-1])} | {fid(len(part), 0)} |")
    L.append("")

    # ------------------------------------------------------------------ A.3
    L.append("### A.3 Distribusi label per bagian\n")
    L.append(
        f"Label 1 = close(t+{f['lookahead_n']}) > close(t) × "
        f"{fid(1 + f['move_threshold'], 3)} (naik ≥ {fid(100 * f['move_threshold'], 1)}% "
        f"dalam {f['lookahead_n']} candle); label 0 = selainnya.\n"
    )
    L.append("| Koin | Bagian | Kelas 1 | % kelas 1 | Kelas 0 | % kelas 0 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for t, d in data.items():
        for name, part in list(d["parts"].items()) + [("Total bersih", d["clean"])]:
            y = part["target"].values
            n1, n0 = int((y == 1).sum()), int((y == 0).sum())
            L.append(f"| {t} | {name} | {fid(n1, 0)} | {pct(n1, len(y))} | {fid(n0, 0)} | {pct(n0, len(y))} |")
    L.append("")

    # ------------------------------------------------------------------ A.4
    L.append("### A.4 Contoh baris BTC-USD\n")
    clean, raw15 = btc["clean"], btc["raw15"]
    tr, va, te = btc["slices"]
    train = clean.iloc[tr]
    X_all = btc["prepped"]["X"]
    scaler = fit_scaler(X_all[tr])  # skema yang sama dgn fitness & model pemenang
    t1 = train.index[train["target"].values == 1][0]
    t0 = train.index[train["target"].values == 0][0]
    L.append(
        "Aturan pemilihan (deterministik, tidak dipilih-pilih): baris **pertama** berlabel 1 dan "
        "baris **pertama** berlabel 0 pada bagian latih.\n"
    )

    def label_info(t: pd.Timestamp) -> dict:
        k = raw15.index.get_loc(t)
        c_t = float(raw15["close"].iloc[k])
        c_t4 = float(raw15["close"].iloc[k + f["lookahead_n"]])
        thr = c_t * (1 + f["move_threshold"])
        lab = int(c_t4 > thr)
        assert lab == int(clean.loc[t, "target"]), "label manual != label pipeline"
        assert np.isclose(clean.loc[t, "close"], raw15["close"].iloc[k - 1]), "penggeseran tak sesuai"
        return {"t": t, "t4": raw15.index[k + f["lookahead_n"]], "t_feat": raw15.index[k - 1],
                "c_t": c_t, "c_t4": c_t4, "thr": thr, "lab": lab}

    e1, e0 = label_info(t1), label_info(t0)
    L.append("**Pembentukan label**\n")
    L.append("| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |")
    L.append("|---|---:|---:|")
    L.append(f"| Timestamp t | {ts(e1['t'])} | {ts(e0['t'])} |")
    L.append(f"| close(t) | {fid(e1['c_t'], 2)} | {fid(e0['c_t'], 2)} |")
    L.append(f"| Timestamp t+{f['lookahead_n']} | {ts(e1['t4'])} | {ts(e0['t4'])} |")
    L.append(f"| close(t+{f['lookahead_n']}) | {fid(e1['c_t4'], 2)} | {fid(e0['c_t4'], 2)} |")
    L.append(f"| Ambang close(t) × {fid(1 + f['move_threshold'], 3)} | {fid(e1['thr'], 2)} | {fid(e0['thr'], 2)} |")
    L.append(f"| close(t+{f['lookahead_n']}) > ambang? | {'ya' if e1['lab'] else 'tidak'} | {'ya' if e0['lab'] else 'tidak'} |")
    L.append(f"| **Label** | **{e1['lab']}** | **{e0['lab']}** |")
    L.append("")

    r1 = clean.loc[t1, FEATURE_COLUMNS].values.astype(np.float64)
    r0 = clean.loc[t0, FEATURE_COLUMNS].values.astype(np.float64)
    n1 = scaler.transform(r1.reshape(1, -1).astype(np.float32))[0]
    n0 = scaler.transform(r0.reshape(1, -1).astype(np.float32))[0]
    L.append(
        f"**Nilai 25 fitur pada baris t (setelah penggeseran satu langkah)** — nilai mentah "
        f"berasal dari candle t−15 menit ({ts(e1['t_feat'])} dan {ts(e0['t_feat'])}). "
        f"Normalisasi: MinMaxScaler [0, 1] yang di-fit **hanya** pada bagian latih "
        f"({fid(tr.stop - tr.start, 0)} baris), skema yang sama dengan evaluasi fitness dan "
        f"model pemenang.\n"
    )
    L.append("| # | Fitur | Label 1: mentah | Label 1: norm. | Label 0: mentah | Label 0: norm. |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for i, c in enumerate(FEATURE_COLUMNS, 1):
        L.append(f"| {i} | `{c}` | {fraw(r1[i - 1])} | {fid(n1[i - 1], 6)} | {fraw(r0[i - 1])} | {fid(n0[i - 1], 6)} |")
    L.append("")
    from data.preprocessor import FGI_CLASSES, FGI_UNKNOWN_CODE, _fit_fgi_encoder
    enc = _fit_fgi_encoder()
    mapping = {c: enc.encode(c) for c in FGI_CLASSES}
    L.append(
        "**Pengodean `fgi_classification_encoded`.** Pemetaan ordinal eksplisit mengikuti urutan "
        f"sentimen (`FGIOrdinalEncoder`); kelas lain diberi kode {FGI_UNKNOWN_CODE}. Encoder ini "
        "disimpan bersama artefak model (`fgi_encoder.pkl`), sehingga pelatihan, backtest, dan "
        "inferensi memakai pemetaan yang sama:\n"
    )
    L.append("| Kelas | " + " | ".join(FGI_CLASSES) + " |")
    L.append("|---|" + "---:|" * len(FGI_CLASSES))
    L.append("| Kode | " + " | ".join(str(mapping[c]) for c in FGI_CLASSES) + " |")
    L.append(
        "\nSebelumnya kode ini dibuat oleh `LabelEncoder`, yang mengurutkan kelas secara alfabetis "
        "(*Extreme Greed* = 1, *Neutral* = 4); diperbaiki pada commit `4b230a5`.\n"
    )
    L.append(
        "**Penjajaran waktu jendela.** Fitur digeser satu langkah (`apply_anti_leakage`), sehingga "
        f"baris t berisi nilai candle t−1. Jendela untuk y(t) adalah baris t−{seq_len - 1} … t "
        "(`sequence_window`, baris t **ikut**), jadi informasi terbaru yang dilihat model berasal "
        f"dari candle **t−1**, sedangkan label membandingkan close(t+{f['lookahead_n']}) dengan "
        "close(t). Satu fungsi yang sama membentuk jendela pada pencarian, walk-forward, model "
        "pemenang, backtest, dan inferensi. Baris fitur pada tabel di atas adalah baris "
        f"**terakhir** jendela untuk y(t), dan juga muncul di jendela y(t+1) … y(t+{seq_len - 1}).\n"
    )
    L.append(kutip("data/preprocessor.py", r"def sequence_window\(", r"return X\[i - sequence_length \+ 1"))
    L.append(
        "**Penyelarasan fitur 1 jam.** Yahoo menstempel bar dengan waktu **mulai**: bar 1 jam "
        "berstempel h menggabungkan candle 15m h, h+15, h+30, h+45 (diverifikasi pada snapshot: "
        "close bar 1 jam h sama dengan close candle 15m h+45 pada 100% bar) dan baru diketahui saat "
        "tutup pada h+1 jam. Setiap baris 15m r diketahui saat tutup pada r+15 menit, sehingga "
        "hanya boleh memakai bar 1 jam terakhir yang sudah tutup saat itu. Contoh: baris 15m 10:15 "
        "(diketahui 10:30) memakai bar 1 jam 09:00 (tutup 10:00), bukan bar 10:00 yang baru tutup "
        "11:00. Penggabungan lama berdasarkan stempel mulai membocorkan hingga 45 menit data masa "
        "depan ke `rsi_1h`, `ema_21_1h`, dan `macd_hist_1h`; diperbaiki pada commit `613e4bd`.\n"
    )
    L.append(kutip("data/preprocessor.py", r"by_close = sec\.copy\(\)", r"aligned\.index = primary_15m\.index"))
    L.append(
        "Kebenaran penjajaran diuji otomatis oleh `test_penjajaran.py`: semua informasi yang belum "
        "diketahui pada awal candle t (candle 15m ≥ t, bar 1 jam yang belum tutup, FGI bertanggal "
        "> t) diganti nilai acak, lalu jendela untuk y(t) harus identik bit demi bit.\n"
    )

    # ------------------------------------------------------------------ A.5
    L.append(f"### A.5 Bentuk tensor BTC-USD (sequence_length = {seq_len})\n")
    y_all = btc["prepped"]["y"]
    Xtr, ytr = create_sequences(scaler.transform(X_all[tr]), y_all[tr], seq_len)
    Xva, yva = create_sequences(scaler.transform(X_all[va]), y_all[va], seq_len)
    L.append("| Bagian | Baris | X | y | % label 1 (sekuens) |")
    L.append("|---|---:|---|---|---:|")
    L.append(f"| Latih | {fid(tr.stop - tr.start, 0)} | {Xtr.shape} | {ytr.shape} | {pct(int(ytr.sum()), len(ytr))} |")
    L.append(f"| Validasi | {fid(va.stop - va.start, 0)} | {Xva.shape} | {yva.shape} | {pct(int(yva.sum()), len(yva))} |")
    L.append(
        f"\nJumlah sekuens = n − L + 1 = jumlah baris − {seq_len - 1} pada tiap bagian, karena sekuens dibentuk "
        "**di dalam** masing-masing bagian setelah penskalaan (tidak ada jendela yang melintasi "
        "batas latih/validasi). Dimensi: (sampel, langkah waktu, fitur).\n"
    )

    # Suplemen walk-forward
    n_splits = int(cfg["optimization"]["final_splits"])
    L.append(f"**Suplemen — lipatan walk-forward BTC-USD** (`TimeSeriesSplit(n_splits={n_splits})`, "
             "dipakai untuk metrik yang dilaporkan; scaler di-fit ulang per lipatan pada bagian latihnya):\n")
    L.append("| Lipatan | Latih (baris) | Validasi (baris) | X latih | X validasi | Rentang validasi | % label 1 validasi |")
    L.append("|---:|---:|---:|---|---|---|---:|")
    wf = list(TimeSeriesSplit(n_splits=n_splits).split(X_all))
    for k, (a, b) in enumerate(wf, 1):
        # Bentuk diambil dari create_sequences itu sendiri (penskalaan tak memengaruhi bentuk)
        Xa_s, _ = create_sequences(X_all[a], y_all[a], seq_len)
        Xb_s, yb_s = create_sequences(X_all[b], y_all[b], seq_len)
        L.append(
            f"| {k} | {fid(len(a), 0)} | {fid(len(b), 0)} | {Xa_s.shape} "
            f"| {Xb_s.shape} | {ts(clean.index[b[0]])} – {ts(clean.index[b[-1]])} "
            f"| {pct(int(yb_s.sum()), len(yb_s))} |"
        )
    lb = 200
    L.append(
        f"\n**Suplemen — jendela backtest BTC-USD:** {lb} baris bersih terakhir, "
        f"{ts(clean.index[-lb])} – {ts(clean.index[-1])}, seluruhnya di dalam bagian uji "
        f"(uji dimulai {ts(clean.index[te.start])}).\n"
    )

    # ------------------------------------------------------------------ B
    import tensorflow as tf  # noqa: F401  (impor berat hanya untuk bagian B)
    from model.lstm_model import build_model, train_model
    from optimization.experiment import manual_hp_from_cfg
    from optimization.fitness import build_cfg_for_hp
    from optimization.search_space import SearchSpace

    space = SearchSpace.from_config(cfg)
    hp = space.decode(BASELINE_CHROM)
    assert hp == manual_hp_from_cfg(cfg), "kromosom baseline != hyperparameter manual di config"

    L.append("## B. Bahan 4.1.2 — Model\n")
    L.append(f"### B.6 `model.summary()` — konfigurasi baseline {BASELINE_CHROM}\n")
    L.append(
        "Kromosom didekode menjadi: " + ", ".join(f"`{k}={v}`" for k, v in hp.items())
        + ". Nilai ini identik dengan hyperparameter manual di `config.yaml` (diverifikasi).\n"
    )
    cfg_search = build_cfg_for_hp(cfg, hp, epochs=int(cfg["optimization"]["search_epochs"]))
    model = build_model((seq_len, len(FEATURE_COLUMNS)), cfg_search["model"])
    # Rich memakai lebar 80 kolom bila stdout bukan terminal sehingga kolom
    # "Param #" terpotong; COLUMNS memaksa lebar yang cukup.
    import os
    os.environ["COLUMNS"] = "120"
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.summary(line_length=100)
    summary = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).rstrip()
    L.append("```text\n" + summary + "\n```\n")

    # ------------------------------------------------------------------ B.7
    L.append("### B.7 `compile()` dan callbacks\n")
    L.append("Pembangunan dan kompilasi model (sama untuk semua tahap):\n")
    L.append(kutip("model/lstm_model.py", r"model\.compile\(", r"^\s{4}\)\s*$"))
    L.append("Callbacks, bobot kelas, dan `fit()` (fungsi `train_model`, dipakai semua tahap):\n")
    L.append(kutip("model/lstm_model.py", r"cb_verbose = 1 if verbose", r"^\s*return history"))
    L.append("Tahap **pencarian** (GA dan Grid, `fitness.evaluate`):\n")
    L.append(kutip("optimization/fitness.py", r"search_epochs = int\(", r"^\s{4}\)\s*$"))
    L.append("Tahap **evaluasi akhir** — konfigurasi, walk-forward, dan model pemenang:\n")
    L.append(kutip("optimization/experiment.py", r"final_splits = int\(", r"eval_cfg = build_cfg_for_hp\("))
    L.append(kutip("model/validator.py", r"# Fresh model per fold", r"save_path=None,", extra=1))
    L.append(kutip("optimization/experiment.py", r"train_sl, val_sl, _ = split_chronological",
                   r"save_path=None, verbose=0,", extra=1))
    L.append("Nilai yang dibaca dari konfigurasi eksperimen:\n")
    L.append(kutip(CONFIG, r"^model:", r"n_splits_cv:"))
    L.append(kutip(CONFIG, r"^optimization:", r"final_epochs:"))

    # Tangkap argumen fit() yang sebenarnya untuk tiap tahap (tanpa melatih)
    def tangkap(model_cfg: dict, Xa, ya, Xb, yb) -> dict:
        got: dict = {}

        def fake_fit(self, *args, **kwargs):
            got.update(kwargs)
            got["_args"] = args
            return None

        with mock.patch.object(type(model), "fit", fake_fit):
            train_model(model, Xa, ya, Xb, yb, model_cfg, save_path=None, verbose=0)
        return got

    cfg_final = build_cfg_for_hp(cfg, hp, epochs=cfg["optimization"].get("final_epochs"),
                                 n_splits=n_splits)
    cap_s = tangkap(cfg_search["model"], Xtr, ytr, Xva, yva)
    cap_f = tangkap(cfg_final["model"], Xtr, ytr, Xva, yva)

    def cb_desc(cb) -> str:
        name = type(cb).__name__
        if name == "EarlyStopping":
            keys = ["monitor", "mode", "patience", "min_delta", "restore_best_weights",
                    "baseline", "start_from_epoch"]
        elif name == "ReduceLROnPlateau":
            keys = ["monitor", "mode", "factor", "patience", "min_lr", "min_delta", "cooldown"]
        else:
            keys = ["monitor", "mode"]
        vals = []
        for k in keys:
            if hasattr(cb, k):
                v = getattr(cb, k)
                vals.append(f"{k}={v!r}")
        return f"`{name}(" + ", ".join(vals) + ")`"

    L.append("**Argumen `fit()` yang sebenarnya** — ditangkap dengan memanggil `train_model` "
             "asli dan menyadap `model.fit` (tanpa melatih). Parameter yang tidak ditulis di kode "
             "memakai nilai bawaan Keras terpasang, ditampilkan apa adanya:\n")
    L.append("| Tahap | epochs | batch_size | shuffle | Callbacks |")
    L.append("|---|---:|---:|---|---|")
    for label, cap in [("Pencarian (GA & Grid)", cap_s),
                       (f"Evaluasi akhir: walk-forward ({n_splits} lipatan) & model pemenang", cap_f)]:
        sh = cap.get("shuffle", "tidak diberikan → bawaan Keras `True`")
        L.append(f"| {label} | {cap['epochs']} | {cap['batch_size']} | {sh} | "
                 + "<br>".join(cb_desc(c) for c in cap["callbacks"]) + " |")
    L.append(
        "\n`ModelCheckpoint` tidak aktif di kedua tahap (`save_path=None`). `restore_best_weights=True` "
        "berarti bobot dikembalikan ke epoch dengan `val_auc` tertinggi.\n"
    )

    # ------------------------------------------------------------------ B.8
    L.append("### B.8 `class_weight` BTC-USD\n")
    cw = cap_s["class_weight"]
    n0s, n1s = int((ytr == 0).sum()), int((ytr == 1).sum())
    L.append(
        f"Dihitung oleh `train_model` dari label **sekuens** latih (bukan baris), "
        f"rumus `total / (2 × cacah_kelas)`: {fid(len(ytr), 0)} sekuens latih, kelas 0 = "
        f"{fid(n0s, 0)}, kelas 1 = {fid(n1s, 0)}.\n"
    )
    L.append("| Skema | Sekuens latih | Kelas 0 | Kelas 1 | Bobot kelas 0 | Bobot kelas 1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(f"| Bagian latih 70% (pencarian & model pemenang) | {fid(len(ytr), 0)} | {fid(n0s, 0)} "
             f"| {fid(n1s, 0)} | {fid(cw[0], 4)} | {fid(cw[1], 4)} |")
    for k, (a, b) in enumerate(wf, 1):
        sc = fit_scaler(X_all[a])
        Xa, ya = create_sequences(sc.transform(X_all[a]), y_all[a], seq_len)
        Xb, yb = create_sequences(sc.transform(X_all[b]), y_all[b], seq_len)
        cwk = tangkap(cfg_final["model"], Xa, ya, Xb, yb)["class_weight"]
        L.append(f"| Walk-forward lipatan {k} | {fid(len(ya), 0)} | {fid(int((ya == 0).sum()), 0)} "
                 f"| {fid(int((ya == 1).sum()), 0)} | {fid(cwk[0], 4)} | {fid(cwk[1], 4)} |")
    L.append("")

    # ------------------------------------------------------------------ C.9
    L.append("## C. Verifikasi untuk naskah\n")
    L.append("### C.9 Cara agregasi metrik\n")
    L.append(
        "**Dua tahap: rata-rata antar-lipatan per seed, lalu mean ± std antar-seed.**\n\n"
        f"1. Untuk satu seed, `walk_forward_validate` menghitung metrik di tiap lipatan lalu "
        "merangkumnya menjadi baris MEAN dan STD (pandas, `std` dengan ddof=1):\n"
    )
    L.append(kutip("model/validator.py", r"metric_cols = \[", r"out = pd\.concat"))
    L.append("2. `final_evaluate_and_save` hanya mengambil baris **MEAN** (rata-rata antar-lipatan). "
             "Baris STD antar-lipatan tidak dipakai lebih lanjut:\n")
    L.append(kutip("optimization/experiment.py", r"report = walk_forward_validate\(", r"wf_recall = float\(mean_row"))
    L.append("3. Satu baris CSV per seed. Agregat lintas seed di `reeval_final.py` memakai "
             "`statistics.mean` dan `statistics.stdev` (simpangan baku sampel, ddof=1):\n")
    L.append(kutip("optimization/reeval_final.py", r"def _agg\(", r"return f\"\{mean\(values\)"))
    L.append(kutip("optimization/reeval_final.py", r"groups: dict\[tuple", r"return \[float\(r\[key\]\)"))
    L.append("4. Tabel skripsi dibuat `verifikasi_multiseed.py` dengan pandas `.agg(['mean', 'std'])` "
             "— juga ddof=1, jadi konsisten dengan tabel konsol `reeval_final.py`. Skrip itu "
             "membuang duplikat per (ticker, metode, seed) dengan mempertahankan baris terbaru:\n")
    L.append(kutip("verifikasi_multiseed.py", r"new_rows = new_rows\.sort_values", r"subset=\["))
    L.append(kutip("verifikasi_multiseed.py", r"agg = new_rows\.groupby", r"agg = new_rows\.groupby"))
    L.append(
        "Implikasi untuk naskah: angka “mean ± std” menggambarkan **variasi antar-seed dari "
        f"rata-rata {n_splits} lipatan**. Variasi antar-lipatan di dalam satu seed tidak masuk "
        "ke ± tersebut.\n"
    )

    # ------------------------------------------------------------------ C.10
    L.append("### C.10 Apa saja yang diubah oleh seed 43/44 pada evaluasi akhir\n")
    L.append("Seed disetel **sekali** per run oleh pemanggil, sebelum data dimuat dan sebelum model apa pun dibangun:\n")
    L.append(kutip("optimization/reeval_final.py", r"for s_i, seed in enumerate\(combo_seeds\)",
                   r"persist_artifacts=persist,", extra=1))
    L.append(kutip("utils/helpers.py", r"def set_global_seed", r"tf\.random\.set_seed\(seed\)"))
    L.append(
        "Yang dipengaruhi seed:\n\n"
        f"- **Pelatihan {n_splits} model walk-forward** (satu per lipatan, dibangun berurutan) dan "
        "**model pemenang**: inisialisasi bobot, mask dropout, dan urutan mini-batch. Kode `fit()` "
        "tidak memberikan argumen `shuffle`, sehingga berlaku bawaan Keras `shuffle=True` — urutan "
        "sekuens latih diacak tiap epoch. Pengacakan ini **di dalam** bagian latih saja dan tidak "
        "melanggar urutan waktu antar-bagian; namun kalimat “tanpa shuffle” di naskah sebaiknya "
        "dibatasi pada pembagian data, bukan pada urutan mini-batch.\n"
        "- **Backtest secara tidak langsung**: backtest memakai model pemenang dari seed tersebut, "
        "sehingga prediksi dan sinyalnya berbeda antar-seed. Prosedur backtest sendiri tidak "
        "memakai bilangan acak — `model.predict` dan aturan sinyal bersifat deterministik:\n"
    )
    L.append(kutip("optimization/experiment.py", r"bt = run_backtest\(", r"n_candles = int\(bt"))
    L.append(
        "Yang **tidak** dipengaruhi seed: data dan praproses (identik untuk ketiga seed; di "
        "`reeval_final.py` data dimuat sekali per koin), pembagian lipatan, scaler, dan data "
        "backtest (dalam mode cache-only, semua dibaca dari snapshot yang sama).\n\n"
        "Selain itu, posisi seed — bukan nilainya — menentukan penyimpanan artefak: hanya "
        "seed pertama (`persist = (s_i == 0)`) yang menulis `model.keras`, `scaler.pkl`, "
        "`fgi_encoder.pkl`, dan `hyperparams.json`. Argumen `seed` di dalam "
        "`final_evaluate_and_save` hanya dicatat ke `hyperparams.json` dan log; penyetelan "
        "seed dilakukan oleh pemanggil.\n"
    )
    L.append(
        "Catatan untuk Langkah 2: `set_global_seed` menyetel `random`, `numpy`, dan "
        "`tf.random`, tetapi belum memanggil `keras.utils.set_random_seed` maupun "
        "`tf.config.experimental.enable_op_determinism()`. Apakah ini cukup agar hasil tidak "
        "bergantung pada posisi run dalam satu proses akan dibuktikan oleh uji determinisme "
        "(Langkah 2b).\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Ditulis: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
