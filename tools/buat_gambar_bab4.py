#!/usr/bin/env python
"""Bahan D3/D4 Bab 4: konvergensi.csv dan gambar (PNG 300 dpi + SVG).

Hanya membaca CSV di logs/ (tanpa pelatihan, tanpa TensorFlow), sehingga bisa
dijalankan dengan Python mana pun yang punya pandas + matplotlib; venv eksperimen
sengaja tidak ditambah paket.

Keluaran:
  bahan_bab4/konvergensi.csv
      ticker, evaluasi (1..50), ga_fitness, ga_terbaik, grid_fitness, grid_terbaik.
      GA: hanya evaluasi non-cache dari ga_trace, diurutkan eval_count_kumulatif;
      kosong setelah evaluasi terakhir GA. Grid: grid_history.
  bahan_bab4/gambar/konvergensi.{png,svg}
      5 panel (satu per koin), fungsi tangga fitness terbaik vs cacah evaluasi.
  bahan_bab4/gambar/auc_walkforward.{png,svg}
      Batang mean wf_auc per koin per metode, error bar = std antar 3 seed
      (45 baris reeval terbaru di optimization_results.csv).

Usage:
    python tools/buat_gambar_bab4.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "bahan_bab4"
FIG = OUT / "gambar"
TICK = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
METH = ["ga", "grid", "manual"]
NAMA = {"ga": "Algoritma Genetika", "grid": "Grid Search", "manual": "Manual (baseline)"}
# Palet kategorikal tervalidasi (slot 1-3, lolos CVD); identitas juga dibawa
# gaya garis / arsiran agar tetap terbaca saat dicetak hitam-putih.
WARNA = {"ga": "#2a78d6", "grid": "#eb6834", "manual": "#1baf7a"}
GARIS = {"ga": "-", "grid": "--"}
ARSIR = {"ga": "", "grid": "//", "manual": ".."}
TINTA, TINTA2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
K_MAX = 50

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
})


def koma(dec: int):
    return FuncFormatter(lambda v, _p: f"{v:.{dec}f}".replace(".", ","))


def konvergensi() -> pd.DataFrame:
    rows = []
    for t in TICK:
        safe = t.replace("-", "_")
        gf = glob.glob(str(LOGS / f"ga_trace_{safe}_*.csv"))
        hf = glob.glob(str(LOGS / f"grid_history_{safe}_*.csv"))
        assert len(gf) == 1 and len(hf) == 1, (t, gf, hf)
        tr = pd.read_csv(gf[0], dtype={"from_cache": str})
        tr = tr[tr.from_cache != "True"].sort_values("eval_count_kumulatif")
        assert list(tr.eval_count_kumulatif) == list(range(1, len(tr) + 1)), t
        gh = pd.read_csv(hf[0]).sort_values("eval")
        assert list(gh["eval"]) == list(range(1, K_MAX + 1)), t
        ga_f = dict(zip(tr.eval_count_kumulatif, tr.fitness))
        ga_b = dict(zip(tr.eval_count_kumulatif, tr.fitness.cummax()))
        for k in range(1, K_MAX + 1):
            g = gh.iloc[k - 1]
            rows.append({"ticker": t, "evaluasi": k,
                         "ga_fitness": ga_f.get(k, np.nan), "ga_terbaik": ga_b.get(k, np.nan),
                         "grid_fitness": g.fitness, "grid_terbaik": g.best_so_far})
    df = pd.DataFrame(rows)
    # best_so_far grid_history = maksimum berjalan (dicek, pembulatan 6 desimal)
    for t, g in df.groupby("ticker"):
        assert np.allclose(g.grid_terbaik, g.grid_fitness.cummax(), atol=1e-6), t
    df.to_csv(OUT / "konvergensi.csv", index=False, float_format="%.6f")
    return df


def simpan(fig, nama: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{nama}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{nama}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gambar_konvergensi(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    axes = axes.ravel()
    for ax, t in zip(axes, TICK):
        g = df[df.ticker == t]
        for m, col in (("ga", "ga_terbaik"), ("grid", "grid_terbaik")):
            s = g.dropna(subset=[col])
            ax.step(s.evaluasi, s[col], where="post", color=WARNA[m], ls=GARIS[m], lw=2,
                    label=NAMA[m])
            ax.plot(s.evaluasi.iloc[-1], s[col].iloc[-1], "o", ms=5, color=WARNA[m],
                    mec="white", mew=1.5, zorder=3)
        ga, gr = g.dropna(subset=["ga_terbaik"]), g
        nilai = lambda v: f"{v:.4f}".replace(".", ",")  # noqa: E731
        teks = (f"GA    {nilai(ga.ga_terbaik.iloc[-1])} ({int(ga.evaluasi.iloc[-1])} eval.)\n"
                f"Grid  {nilai(gr.grid_terbaik.iloc[-1])} ({K_MAX} eval.)")
        ax.text(0.98, 0.04, teks, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color=TINTA, family="DejaVu Sans Mono",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRID_C})
        ax.set_title(t, fontsize=10, color=TINTA, loc="left")
        ax.set_xlim(0.5, K_MAX + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_formatter(koma(3))
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        ax.set_xlabel("Cacah evaluasi")
        ax.set_ylabel("Fitness terbaik (AUC validasi)")
    leg_ax = axes[-1]
    leg_ax.axis("off")
    h, lab = axes[0].get_legend_handles_labels()
    leg_ax.legend(h, lab, loc="center", frameon=False, fontsize=10,
                  title="Metode pencarian", title_fontsize=10)
    simpan(fig, "konvergensi")


def gambar_auc() -> pd.DataFrame:
    res = pd.read_csv(LOGS / "optimization_results.csv")
    ree = res.iloc[15:]
    assert len(ree) == 45 and not ree.duplicated(["ticker", "method", "seed"]).any()
    agg = ree.groupby(["ticker", "method"]).wf_auc.agg(["mean", "std", "count"])
    assert (agg["count"] == 3).all()
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    x = np.arange(len(TICK))
    w = 0.26
    for j, m in enumerate(METH):
        mu = [agg.loc[(t, m), "mean"] for t in TICK]
        sd = [agg.loc[(t, m), "std"] for t in TICK]
        pos = x + (j - 1) * (w + 0.01)
        # Batang bermula dari AUC 0,5 (tebakan acak): tingginya = keunggulan atas acak
        ax.bar(pos, np.array(mu) - 0.5, w, bottom=0.5, color=WARNA[m], hatch=ARSIR[m],
               edgecolor="white", lw=0, label=NAMA[m], zorder=2)
        ax.errorbar(pos, mu, yerr=sd, fmt="none", ecolor=TINTA, elinewidth=1, capsize=3, zorder=3)
    ax.set_xticks(x, TICK)
    ax.set_ylim(0.5, max(agg["mean"] + agg["std"]) + 0.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(koma(2))
    ax.grid(axis="y", color=GRID_C, lw=0.8, zorder=0)
    ax.set_xlabel("Koin")
    ax.set_ylabel("AUC walk-forward (rerata 3 seed)\ngaris dasar = 0,5 (acak)")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    simpan(fig, "auc_walkforward")
    return agg


def main() -> int:
    df = konvergensi()
    gambar_konvergensi(df)
    agg = gambar_auc()
    print(f"Ditulis: {OUT / 'konvergensi.csv'} ({len(df)} baris)")
    print(f"Ditulis: {FIG} (konvergensi, auc_walkforward; PNG 300 dpi + SVG)")
    print(agg.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
