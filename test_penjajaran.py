"""Uji otomatis penjajaran waktu jendela dan label (offline, data snapshot).

Untuk sejumlah t acak per koin (mencakup menit :00, :15, :30, :45):

 (a) Jendela untuk y(t) tidak memuat informasi dari candle >= t.
     Uji perturbasi: setiap observasi yang BELUM tersedia pada awal candle t
     diganti nilai acak — candle 15m dengan stempel >= t, bar 1 jam yang belum
     ditutup pada t (stempel + 1 jam > t), dan FGI bertanggal > t — lalu fitur
     dibangun ulang. Jendela (sebelum penskalaan) harus identik bit demi bit.
 (b) Baris terakhir jendela berisi nilai candle t-1, yaitu baris fitur
     tak-tergeser milik candle t-1 (dan kolom close = close mentah t-1).
 (c) y(t) = 1 jika close(t+4) > close(t) x (1 + ambang), dari close mentah.
     Label tidak berubah bila semua close selain t dan t+4 diganggu, dan
     berbalik bila close(t+4) dipindah melewati ambang.

Ditambah: create_sequences menghasilkan n - L + 1 sampel yang identik dengan
sequence_window, dan backtest serta inferensi memanggil sequence_window yang
sama (satu definisi jendela).

Usage: python test_penjajaran.py      (keluar dengan kode 1 bila ada yang gagal)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.fetcher import fetch_all  # noqa: E402
from data.preprocessor import (  # noqa: E402
    FEATURE_COLUMNS,
    FGI_CLASSES,
    add_target,
    apply_anti_leakage,
    build_features,
    create_sequences,
    sequence_window,
)
from utils.helpers import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

get_logger().setLevel(logging.WARNING)  # senyapkan log pipeline

CONFIG = "config_snapshot_20260912.yaml"
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
PER_MINUTE = 2          # t acak per menit-dalam-jam per koin -> 8 t per koin
SEED = 2026

cfg = load_config(CONFIG)
L = int(cfg["features"]["sequence_length"])
N = int(cfg["features"]["lookahead_n"])
THR = float(cfg["features"]["move_threshold"])
H1 = pd.Timedelta(hours=1)


def build(bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_df, _ = build_features(bundle, cfg)
    labeled = add_target(feat_df, N, THR)
    clean = apply_anti_leakage(labeled[FEATURE_COLUMNS + ["target"]])
    return feat_df, clean


def perturb(frame: pd.DataFrame, mask: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    k = int(mask.sum())
    f = rng.uniform(0.5, 1.5, size=k)
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].values * f
    # Pertahankan dtype kolom (volume Yahoo = int64; pandas 3 menolak upcast)
    vol = out.loc[mask, "volume"].values * rng.uniform(0.0, 5.0, size=k) + 1.0
    out.loc[mask, "volume"] = np.round(vol).astype(out["volume"].dtype)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    fails: list[str] = []
    print(f"Uji penjajaran — L={L}, lookahead={N}, ambang={THR}  (data: {CONFIG})\n")
    print(f"{'koin':9} {'t (UTC)':17} {'mnt':>3}  {'(a) tanpa info >= t':28} {'(b) baris akhir=t-1':20} {'(c) label':10}")
    print("-" * 96)

    for ticker in TICKERS:
        bundle = fetch_all(cfg, ticker)
        raw = bundle["primary"]
        feat_df, clean = build(bundle)
        Xc = clean[FEATURE_COLUMNS].values

        # --- satu definisi jendela ------------------------------------------ #
        y = clean["target"].values
        Xs, ys = create_sequences(Xc, y, L)
        if len(Xs) != len(Xc) - L + 1:
            fails.append(f"{ticker}: create_sequences {len(Xs)} sampel, harus {len(Xc) - L + 1}")
        for k in rng.integers(0, len(Xs), size=20):
            # create_sequences mengembalikan float32 — bandingkan pada dtype yang sama
            ref = sequence_window(Xc, k + L - 1, L).astype(np.float32)
            if not (np.array_equal(Xs[k], ref) and ys[k] == y[k + L - 1]):
                fails.append(f"{ticker}: create_sequences[{k}] != sequence_window({k + L - 1})")

        # --- pilih t: PER_MINUTE per menit-dalam-jam ------------------------- #
        cand = np.arange(L - 1, len(clean))
        minutes = clean.index[cand].minute
        picks = []
        for m in (0, 15, 30, 45):
            pool = cand[minutes == m]
            picks += list(rng.choice(pool, size=PER_MINUTE, replace=False))

        for k in sorted(picks):
            t = clean.index[k]
            win = sequence_window(Xc, k, L)

            # (a) perturbasi semua yang belum tersedia pada awal candle t
            pb = {
                "primary": perturb(raw, (raw.index >= t), rng),
                "secondary": perturb(bundle["secondary"],
                                     (bundle["secondary"].index + H1 > t), rng),
                "fgi": bundle["fgi"].copy(),
            }
            fm = pb["fgi"].index > t
            pb["fgi"].loc[fm, "fgi_value"] = rng.integers(0, 101, size=int(fm.sum()))
            pb["fgi"].loc[fm, "fgi_classification"] = rng.choice(FGI_CLASSES, size=int(fm.sum()))
            _, clean_p = build(pb)
            win_p = sequence_window(clean_p[FEATURE_COLUMNS].values, clean_p.index.get_loc(t), L)
            if np.array_equal(win, win_p):
                a_txt = "OK"
            else:
                diff = [c for j, c in enumerate(FEATURE_COLUMNS) if not np.array_equal(win[:, j], win_p[:, j])]
                a_txt = "BOCOR: " + ",".join(diff)
                fails.append(f"{ticker} t={t:%Y-%m-%d %H:%M} (a) fitur memuat info >= t: {diff}")

            # (b) baris terakhir = nilai candle t-1
            pr = raw.index.get_loc(t)
            prev_row = feat_df[FEATURE_COLUMNS].iloc[pr - 1].values
            b_ok = np.array_equal(win[-1], prev_row) and win[-1][FEATURE_COLUMNS.index("close")] == raw["close"].iloc[pr - 1]
            if not b_ok:
                fails.append(f"{ticker} t={t:%Y-%m-%d %H:%M} (b) baris akhir jendela != candle t-1")

            # (c) label dari close(t) dan close(t+N) saja
            c_t, c_tn = raw["close"].iloc[pr], raw["close"].iloc[pr + N]
            lab = int(clean.loc[t, "target"])
            c_ok = lab == int(c_tn > c_t * (1 + THR))
            other = raw[["close"]].copy()
            keep = np.zeros(len(other), dtype=bool)
            keep[[pr, pr + N]] = True
            other.loc[~keep, "close"] = other.loc[~keep, "close"].values * rng.uniform(0.5, 1.5, size=int((~keep).sum()))
            c_ok &= int(add_target(other, N, THR).loc[t, "target"]) == lab
            flip = raw[["close"]].copy()
            flip.iloc[pr + N, 0] = c_t * (1 + THR) * (0.999 if lab else 1.001)
            c_ok &= int(add_target(flip, N, THR).loc[t, "target"]) == 1 - lab
            if not c_ok:
                fails.append(f"{ticker} t={t:%Y-%m-%d %H:%M} (c) label tidak sesuai definisi")

            print(f"{ticker:9} {t:%Y-%m-%d %H:%M}  :{t.minute:02d}  {a_txt[:28]:28} "
                  f"{'OK' if b_ok else 'GAGAL':20} {'OK' if c_ok else 'GAGAL':10}")

    # --- backtest & inferensi memakai fungsi jendela yang sama ------------- #
    import main as _main  # noqa: PLC0415
    from data import preprocessor as _pre  # noqa: PLC0415
    for name, fn in [("main.run_backtest", _main.run_backtest),
                     ("preprocessor.prepare_for_prediction", _pre.prepare_for_prediction),
                     ("preprocessor.create_sequences", _pre.create_sequences)]:
        if "sequence_window(" not in inspect.getsource(fn):
            fails.append(f"{name} tidak memakai sequence_window")

    print()
    if fails:
        print(f"HASIL: GAGAL ({len(fails)})")
        for f in fails[:40]:
            print("  -", f)
        return 1
    print("HASIL: LULUS — (a), (b), (c) terpenuhi untuk semua t; satu definisi jendela dipakai di semua fase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
