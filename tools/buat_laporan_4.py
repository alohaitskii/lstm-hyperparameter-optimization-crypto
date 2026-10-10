"""Susun bahan_bab4/laporan_4.md dari hasil Langkah 4 (reeval 3 seed di atas snapshot).

Sumber (semuanya di logs/):
  optimization_results.csv  15 baris pencarian (seed 42) + 45 baris reeval
  optimization_folds.csv    metrik per fold reeval (45 x 3 fold)
  backtest_*_2026100[9]_*   backtest reeval (sinyal LONG/SHORT/HOLD)
  ga_trace_*, grid_history_* jejak pencarian (fitness terbaik per evaluasi)

std = simpangan baku sampel (ddof=1) antar 3 seed.

Usage: python tools/buat_laporan_4.py
"""
from __future__ import annotations

import glob
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "bahan_bab4" / "laporan_4.md"
TICK = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
METH = ["ga", "grid", "manual"]
NAMA = {"ga": "GA", "grid": "Grid", "manual": "Manual"}
SEEDS = [42, 43, 44]

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fmt(x: float, d: int = 4) -> str:
    # Pembulatan setengah ke atas (359,55 -> 359,6), sama dengan laporan_3a/3b.
    if pd.isna(x):
        return "–"
    q = Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:.{d}f}".replace(".", ",")


def ms(s: pd.Series, d: int = 4) -> str:
    s = s.dropna()
    if s.empty:
        return "–"
    sd = s.std(ddof=1) if len(s) > 1 else float("nan")
    tail = "" if len(s) == 3 else f" (n={len(s)})"
    return f"{fmt(s.mean(), d)} ± {fmt(sd, d)}{tail}"


def ms_hit(s: pd.Series) -> str:
    """Hit-rate: hanya seed yang menerbitkan sinyal (NaN = tanpa sinyal), n selalu ditulis."""
    n = int(s.notna().sum())
    if n == 0:
        return "tidak terdefinisi (n=0)"
    return ms(s).split(" (n=")[0] + f" (n={n})"


def table(head: list[str], rows: list[list[str]], right_from: int = 1) -> str:
    al = ["---" if i < right_from else "---:" for i in range(len(head))]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(al) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def hp_str(h: dict) -> str:
    return (f"seq {h['sequence_length']}, u {h['lstm_units_1']}/{h['lstm_units_2']}, "
            f"drop {h['dropout_rate']}, lr {h['learning_rate']}, bs {h['batch_size']}")


# --------------------------------------------------------------------------- #
res = pd.read_csv(LOGS / "optimization_results.csv")
src, ree = res.iloc[:15].copy(), res.iloc[15:].copy()
folds = pd.read_csv(LOGS / "optimization_folds.csv")
ree["ts"] = pd.to_datetime(ree["timestamp"], utc=True)

# ---- verifikasi ------------------------------------------------------------ #
assert len(src) == 15 and len(ree) == 45, (len(src), len(ree))
assert not ree.duplicated(["ticker", "method", "seed"]).any()
assert set(map(tuple, ree[["ticker", "method", "seed"]].values)) == {
    (t, m, s) for t in TICK for m in METH for s in SEEDS}
mrg = ree.merge(src, on=["ticker", "method"], suffixes=("", "_src"))
for c in ("val_auc_search", "search_duration_min", "n_evals", "best_hp_json", "search_epochs"):
    assert (mrg[c].astype(str) == mrg[c + "_src"].astype(str)).all(), c
assert len(folds) == 135 and not folds.duplicated(["ticker", "method", "seed", "fold"]).any()

# ---- sinyal LONG/SHORT/HOLD dari backtest reeval --------------------------- #
sig_rows = []
for t in TICK:
    for m in METH:
        fs = sorted(glob.glob(str(LOGS / f"backtest_{t.replace('-', '_')}_{m}_20261009_*.csv")))
        assert len(fs) == 3, (t, m, fs)
        rows = ree[(ree.ticker == t) & (ree.method == m)].sort_values("ts")
        for f, (_, r) in zip(fs, rows.iterrows()):
            stamp = pd.Timestamp(Path(f).stem[-15:].replace("_", "T"), tz="UTC")
            assert abs((stamp - r.ts).total_seconds()) < 90, (f, r.ts)
            bt = pd.read_csv(f)
            vc = bt["signal"].value_counts()
            n_iss = int(vc.get("LONG", 0) + vc.get("SHORT", 0))
            assert n_iss == r.n_signals_issued and int(vc.get("HOLD", 0)) == r.n_signals_hold, f
            sig_rows.append({"ticker": t, "method": m, "seed": int(r.seed),
                             "LONG": int(vc.get("LONG", 0)), "SHORT": int(vc.get("SHORT", 0)),
                             "HOLD": int(vc.get("HOLD", 0))})
sig = pd.DataFrame(sig_rows)

L: list[str] = []
P = L.append
P("# Laporan Langkah 4 — Evaluasi akhir 3 seed di atas snapshot\n")
P("Data: `data/snapshot_20260912/` (cache-only, diverifikasi terhadap MANIFEST).")
P("Setiap konfigurasi terpilih (GA, Grid, Manual; 5 koin) dievaluasi ulang dengan")
P("`optimization/reeval_final.py --seeds 42 43 44`: walk-forward 3 fold")
P("(`TimeSeriesSplit`, maks. 100 epoch, patience 10) + backtest 200 candle.")
P(f"Run dari terminal VS Code, 9 Okt 17:45–22:16 WIB (271,4 menit, kode keluar 0).\n")
P("Semua std = simpangan baku sampel (ddof = 1) antar 3 seed. Dihasilkan oleh")
P("`tools/buat_laporan_4.py`.\n")

P("## Verifikasi\n")
P("- `optimization_results.csv`: 15 baris pencarian + **45 baris reeval** "
  "(5 koin × 3 metode × seed 42/43/44), tanpa duplikat.")
P("- `val_auc_search`, `search_duration_min`, `n_evals`, `best_hp_json`, `search_epochs` "
  "pada 45 baris = baris pencarian snapshot (GA BTC: 59,86 menit, run ulang 7 Okt).")
P("- `optimization_folds.csv`: 135 baris (45 run × 3 fold), tanpa duplikat.")
P("- Jumlah LONG+SHORT dan HOLD di 45 berkas backtest = kolom `n_signals_*` di CSV hasil.")
P("- `tools/cek_jendela_data.py --semua-harus-cocok`: **60/60** backtest (15 pencarian + "
  "45 reeval) tepat pada snapshot.\n")
d42 = mrg[mrg.seed == 42]
dd = (d42.wf_auc - d42.wf_auc_src).abs()
P("**Catatan seed 42.** Metrik reeval seed 42 tidak sama dengan metrik evaluasi akhir "
  "pada baris pencarian (selisih |wf_auc| maks. "
  f"{fmt(dd.max())}, median {fmt(dd.median())}). Penyebabnya, evaluasi akhir di "
  "`run_optimization.py` berjalan langsung sesudah pencarian tanpa reset seed, "
  "sehingga keadaan RNG-nya bergantung pada jalannya pencarian. `reeval_final.py` "
  "memanggil `set_global_seed(seed)` tepat sebelum setiap evaluasi, sehingga "
  "terdeterminasi oleh seed saja. Angka yang dilaporkan adalah 45 baris reeval.\n")

# ---- a ------------------------------------------------------------------- #
P("## a. Metrik walk-forward dan backtest (mean ± std, 3 seed)\n")
P("AUC, F1, precision, recall = rerata 3 fold walk-forward. Hit-rate = proporsi "
  "sinyal LONG/SHORT yang benar pada backtest 200 candle. Seed tanpa sinyal tidak "
  "punya hit-rate (tercatat NaN di CSV, bukan 0) dan tidak ikut dihitung; n = jumlah "
  "seed yang menerbitkan sinyal. Std dengan n = 1 tidak terdefinisi (–).\n")
MET = [("wf_auc", "AUC"), ("wf_f1", "F1"), ("wf_precision", "Precision"),
       ("wf_recall", "Recall"), ("backtest_hit_rate", "Hit-rate")]
rows = []
for t in TICK:
    for m in METH:
        g = ree[(ree.ticker == t) & (ree.method == m)]
        rows.append([t, NAMA[m]] + [ms(g[c]) for c, _ in MET[:-1]] + [ms_hit(g["backtest_hit_rate"])])
P(table(["Koin", "Metode"] + [n for _, n in MET], rows, 2))
KOIN4 = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]  # ketiga metode bersinyal
P("\n**Rerata lintas koin** (rerata dari mean per koin, setiap koin berbobot sama; "
  "± = std antar koin). Baris *5 koin*: semua metrik; hit-rate dari mean per koin "
  "yang terdefinisi (GA tanpa SHIB-USD, ketiga seed tanpa sinyal). Baris *4 koin*: "
  "hit-rate pada BTC, ETH, SOL, LINK, tempat ketiga metode bersinyal, sehingga "
  "ketiga metode dibandingkan pada koin yang sama (mean SOL-USD Grid dan "
  "LINK-USD Manual masing-masing dari 2 seed bersinyal).\n")
rows = []
per_coin = ree.groupby(["method", "ticker"])[[c for c, _ in MET]].mean()
for m in METH:
    pc = per_coin.loc[m]
    rows.append([NAMA[m], "5 koin"] + [ms(pc[c]).split(" (n=")[0] + (
        "" if pc[c].notna().sum() == 5 else f" ({pc[c].notna().sum()} koin)") for c, _ in MET])
    h4 = pc.loc[KOIN4, "backtest_hit_rate"]
    assert h4.notna().all(), (m, h4)
    rows.append([NAMA[m], "4 koin (BTC, ETH, SOL, LINK)"] + [""] * (len(MET) - 1)
                + [ms(h4).split(" (n=")[0] + " (4 koin)"])
P(table(["Metode", "Cakupan"] + [n for _, n in MET], rows, 2))
P("")

# ---- b ------------------------------------------------------------------- #
P("## b. Sinyal backtest LONG / SHORT / HOLD (200 candle)\n")
rows = []
for t in TICK:
    for m in METH:
        g = sig[(sig.ticker == t) & (sig.method == m)].sort_values("seed")
        per = " · ".join(f"{r.LONG}/{r.SHORT}/{r.HOLD}" for r in g.itertuples())
        rows.append([t, NAMA[m], fmt(g.LONG.mean(), 1), fmt(g.SHORT.mean(), 1),
                     fmt(g.HOLD.mean(), 1), per])
P(table(["Koin", "Metode", "LONG (mean)", "SHORT (mean)", "HOLD (mean)",
         "Per seed 42 · 43 · 44 (L/S/H)"], rows, 2))
P("")
tot = sig.groupby("method")[["LONG", "SHORT", "HOLD"]].sum()
P("Total 15 backtest per metode: " + "; ".join(
    f"{NAMA[m]} {tot.loc[m, 'LONG']} LONG / {tot.loc[m, 'SHORT']} SHORT / {tot.loc[m, 'HOLD']} HOLD"
    for m in METH) + ".")
P("\n**Rerata per backtest per metode** (rerata dari mean per koin, setiap koin "
  "berbobot sama; ± = std antar koin):\n")
sig["TERBIT"] = sig.LONG + sig.SHORT
sig_pc = sig.groupby(["method", "ticker"])[["LONG", "SHORT", "HOLD", "TERBIT"]].mean()
rows = []
for m in METH:
    for label, coins in (("5 koin", TICK), ("4 koin (BTC, ETH, SOL, LINK)", KOIN4)):
        pc = sig_pc.loc[m].loc[coins]
        n0 = int(((sig.method == m) & sig.ticker.isin(coins) & (sig.TERBIT == 0)).sum())
        rows.append([NAMA[m], label] + [ms(pc[c], 1).split(" (n=")[0]
                                        for c in ("LONG", "SHORT", "HOLD", "TERBIT")]
                    + [f"{n0} dari {3 * len(coins)}"])
P(table(["Metode", "Cakupan", "LONG", "SHORT", "HOLD", "Terbit (LONG+SHORT)",
         "Run tanpa sinyal"], rows, 2))
P("")
nz = sig[(sig.LONG + sig.SHORT) == 0]
P(f"Run tanpa sinyal sama sekali: {len(nz)} dari 45 (" +
  ", ".join(f"{r.ticker} {NAMA[r.method]} s{r.seed}" for r in nz.itertuples()) + ").\n")

# ---- c ------------------------------------------------------------------- #
P("## c. Metrik per fold walk-forward (mean ± std, 3 seed)\n")
P("Jumlah sekuens dan % label 1 bergantung pada `sequence_length` konfigurasi, "
  "bukan pada seed (diperiksa identik antar seed).\n")
rows = []
for t in TICK:
    for m in METH:
        for fd in (1, 2, 3):
            g = folds[(folds.ticker == t) & (folds.method == m) & (folds.fold == fd)]
            assert len(g) == 3
            for c in ("n_train_seq", "n_val_seq", "pct_label1_train", "pct_label1_val"):
                assert g[c].nunique() == 1, (t, m, fd, c)
            r0 = g.iloc[0]
            rows.append([t, NAMA[m], fd, ms(g.auc), ms(g.f1), ms(g.precision), ms(g.recall),
                         int(r0.n_train_seq), int(r0.n_val_seq),
                         fmt(r0.pct_label1_train, 1), fmt(r0.pct_label1_val, 1)])
P(table(["Koin", "Metode", "Fold", "AUC", "F1", "Precision", "Recall",
         "n latih", "n validasi", "% label 1 latih", "% label 1 validasi"], rows, 2))
P("")
fa = folds.groupby("fold").auc.agg(["mean", "std"])
P("Rerata AUC seluruh 45 run per fold: " + "; ".join(
    f"fold {f} {fmt(r['mean'])} ± {fmt(r['std'])}" for f, r in fa.iterrows()) + ".\n")

# ---- d ------------------------------------------------------------------- #
P("## d. Efisiensi pencarian\n")
P("Waktu = `search_duration_min` (pencarian saja, tanpa evaluasi akhir). "
  "Manual = satu evaluasi konfigurasi baseline.\n")
rows = []
tot = {m: [0, 0.0] for m in METH}
for t in TICK:
    r = []
    for m in METH:
        s = src[(src.ticker == t) & (src.method == m)].iloc[0]
        tot[m][0] += int(s.n_evals)
        tot[m][1] += float(s.search_duration_min)
        r += [int(s.n_evals), fmt(s.search_duration_min, 1)]
    rows.append([t] + r)
rows.append(["**Total**"] + sum(([f"**{tot[m][0]}**", f"**{fmt(tot[m][1], 1)}**"] for m in METH), []))
rows.append(["Menit per evaluasi"] + sum((["", fmt(tot[m][1] / tot[m][0], 2)] for m in METH), []))
P(table(["Koin", "GA: evaluasi", "GA: menit", "Grid: evaluasi", "Grid: menit",
         "Manual: evaluasi", "Manual: menit"], rows, 1))
P("")
P(f"GA memakai {tot['ga'][0]} evaluasi ({fmt(100 * tot['ga'][0] / tot['grid'][0], 1)}% dari Grid) "
  f"dan {fmt(tot['ga'][1], 1)} menit ({fmt(100 * tot['ga'][1] / tot['grid'][1], 1)}% dari Grid). "
  "Evaluasi GA yang terealisasi < 60 (pop 10 × 6 generasi turunan) karena individu "
  "berulang diambil dari cache fitness. Waktu GA BTC adalah run ulang 7 Okt (lihat "
  "`laporan_3a.md`, catatan waktu).\n")

# ---- e ------------------------------------------------------------------- #
P("## e. Fitness terbaik sejauh ini (AUC pencarian) pada evaluasi ke-k\n")
P("GA: maksimum fitness individu non-cache dengan `eval_count_kumulatif` ≤ k "
  "(`ga_trace`). Grid: `best_so_far` (`grid_history`). Bila evaluasi GA < k, dipakai "
  "nilai evaluasi terakhirnya (ditandai \\*).\n")
K = [10, 20, 30, 40, 50]
rows = []
for t in TICK:
    safe = t.replace("-", "_")
    gf = glob.glob(str(LOGS / f"ga_trace_{safe}_*.csv"))
    hf = glob.glob(str(LOGS / f"grid_history_{safe}_*.csv"))
    assert len(gf) == 1 and len(hf) == 1, (gf, hf)
    tr = pd.read_csv(gf[0], dtype={"from_cache": str})
    tr = tr[tr.from_cache != "True"]
    gh = pd.read_csv(hf[0])
    n_ga = int(tr.eval_count_kumulatif.max())
    assert n_ga == int(src[(src.ticker == t) & (src.method == "ga")].n_evals.iloc[0])
    for name, getter in (
        ("GA", lambda k: (tr[tr.eval_count_kumulatif <= k].fitness.max(), k > n_ga)),
        ("Grid", lambda k: (gh[gh["eval"] <= k].fitness.max(), False)),
    ):
        cells = []
        for k in K:
            v, star = getter(k)
            cells.append(fmt(v) + ("\\*" if star else ""))
        rows.append([t, name] + cells)
    rows.append([t, "GA − Grid"] + [
        fmt(tr[tr.eval_count_kumulatif <= k].fitness.max() - gh[gh["eval"] <= k].fitness.max())
        for k in K])
P(table(["Koin", "Metode"] + [f"k = {k}" for k in K], rows, 2))
P("")

# ---- f ------------------------------------------------------------------- #
P("## f. Fitness pencarian vs AUC walk-forward\n")
P("`val_auc_search` = AUC konfigurasi terpilih pada satu split kronologis 70/15/15 "
  "(`search_epochs` 20). wf_auc = rerata 3 fold walk-forward (100 epoch), mean ± std "
  "3 seed. Selisih = wf_auc (mean) − val_auc_search.\n")
rows = []
gaps = []
for t in TICK:
    for m in METH:
        s = src[(src.ticker == t) & (src.method == m)].iloc[0]
        g = ree[(ree.ticker == t) & (ree.method == m)].wf_auc
        gap = g.mean() - s.val_auc_search
        gaps.append((m, gap))
        rows.append([t, NAMA[m], hp_str(json.loads(s.best_hp_json)),
                     fmt(s.val_auc_search), ms(g), fmt(gap)])
P(table(["Koin", "Metode", "Konfigurasi", "val_auc_search", "wf_auc", "Selisih"], rows, 3))
P("")
gp = pd.DataFrame(gaps, columns=["m", "gap"]).groupby("m").gap.mean()
P("Rerata selisih per metode: " + "; ".join(f"{NAMA[m]} {fmt(gp[m])}" for m in METH) + ".\n")

# ---- g ------------------------------------------------------------------- #
P("## g. Peringkat AUC walk-forward per koin\n")
P("Peringkat berdasarkan mean wf_auc 3 seed (1 = tertinggi). Rentang = mean "
  "tertinggi − mean terendah di koin itu.\n")
mean_auc = ree.groupby(["ticker", "method"]).wf_auc.mean().unstack()[METH].loc[TICK]
ranks = mean_auc.rank(axis=1, ascending=False, method="min")
rows = []
for t in TICK:
    r = mean_auc.loc[t]
    rows.append([t] + [f"{fmt(r[m])} ({int(ranks.loc[t, m])})" for m in METH]
                + [NAMA[r.idxmax()], fmt(r.max() - r.min())])
P(table(["Koin", "GA", "Grid", "Manual", "Terbaik", "Rentang"], rows, 1))
P("")
wins = mean_auc.idxmax(axis=1).value_counts()
rows = [[NAMA[m], int(wins.get(m, 0)), fmt(ranks[m].mean(), 2)] for m in METH]
P(table(["Metode", "Menang (dari 5 koin)", "Peringkat rata-rata"], rows, 1))
rng = mean_auc.max(axis=1) - mean_auc.min(axis=1)
P(f"\nRentang selisih antarmetode per koin: {fmt(rng.min())} ({rng.idxmin()}) "
  f"sampai {fmt(rng.max())} ({rng.idxmax()}).\n")

# ---- h ------------------------------------------------------------------- #
P("## h. Selisih AUC antarmetode vs variasi antar-seed\n")
P("Selisih = mean wf_auc metode pertama − metode kedua. Std = std antar 3 seed "
  "masing-masing. **Tumpang tindih** = interval mean ± std kedua metode beririsan; "
  "rasio = |selisih| / std terbesar dari keduanya.\n")
std_auc = ree.groupby(["ticker", "method"]).wf_auc.std(ddof=1).unstack()[METH].loc[TICK]
rows = []
n_ov = 0
for t in TICK:
    for a, b in combinations(METH, 2):
        ma, mb, sa, sb = mean_auc.loc[t, a], mean_auc.loc[t, b], std_auc.loc[t, a], std_auc.loc[t, b]
        ov = (ma - sa <= mb + sb) and (mb - sb <= ma + sa)
        n_ov += ov
        rows.append([t, f"{NAMA[a]} − {NAMA[b]}", fmt(ma - mb), fmt(sa), fmt(sb),
                     fmt(abs(ma - mb) / max(sa, sb), 2), "**ya**" if ov else "tidak"])
P(table(["Koin", "Pasangan", "Selisih", "Std pertama", "Std kedua", "Rasio", "Tumpang tindih"], rows, 2))
P(f"\n{n_ov} dari {len(rows)} pasangan tumpang tindih.\n")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"Ditulis: {OUT}")
