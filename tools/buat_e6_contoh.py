#!/usr/bin/env python
"""Hasilkan bahan_bab4/E6_contoh.md: contoh agregasi per fold dan contoh sinyal backtest.

Tanpa pelatihan. Probabilitas dan sinyal diambil dari berkas backtest reeval di
logs/; nilai fitur konfirmasi dari pipeline (baris bersih yang sama dengan yang
dibaca main.run_backtest), lalu keputusan sinyal dihitung ulang dengan
signals.generator.confirm_signal dan dicocokkan dengan sinyal di berkas.

Usage:
    python tools/buat_e6_contoh.py
"""
from __future__ import annotations

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

from optimization.experiment import prepare_ticker_data  # noqa: E402
from signals.generator import confirm_signal  # noqa: E402
from utils.helpers import load_config  # noqa: E402

CONFIG = "config_snapshot_20260912.yaml"
OUT = ROOT / "bahan_bab4" / "E6_contoh.md"
LOGS = ROOT / "logs"
TICK = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
METH = ["ga", "grid", "manual"]
NAMA = {"ga": "GA", "grid": "Grid", "manual": "Manual"}
FITUR = ["rsi_14", "macd_hist", "close", "bb_lower", "bb_upper", "stoch_k", "fgi_normalized"]


def fid(x: float, dec: int) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def ts(t) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def backtest_file(row: pd.Series) -> Path:
    """Berkas backtest milik satu baris reeval: stempel nama berkas terdekat ke timestamp baris."""
    stamp = pd.Timestamp(row.timestamp)
    cand = sorted(LOGS.glob(f"backtest_{row.ticker.replace('-', '_')}_{row.method}_*.csv"))
    best = min(cand, key=lambda p: abs((pd.Timestamp(p.stem[-15:].replace("_", "T"), tz="UTC")
                                         - stamp).total_seconds()))
    gap = abs((pd.Timestamp(best.stem[-15:].replace("_", "T"), tz="UTC") - stamp).total_seconds())
    assert gap < 90, (row.ticker, row.method, row.seed, best, gap)
    return best


def main() -> int:
    cfg = load_config(CONFIG)
    sc = cfg["signals"]
    long_thr, short_thr, min_conf = sc["long_threshold"], sc["short_threshold"], sc["min_confirmations"]
    res = pd.read_csv(LOGS / "optimization_results.csv")
    ree = res.iloc[15:].copy()
    assert len(ree) == 45 and not ree.duplicated(["ticker", "method", "seed"]).any()
    folds = pd.read_csv(LOGS / "optimization_folds.csv")
    assert len(folds) == 135

    L: list[str] = []
    P = L.append
    P("# Bahan E6 — contoh agregasi dan contoh sinyal\n")
    P(f"Dihasilkan oleh `tools/buat_e6_contoh.py` pada {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
      "dari 45 baris reeval di `logs/optimization_results.csv`, `logs/optimization_folds.csv`, dan "
      "berkas backtest reeval. Tanpa pelatihan. Waktu dalam UTC.\n")

    # ------------------------------------------------------------------ a
    P("## a. Agregasi AUC BTC-USD Manual: 3 seed × 3 fold\n")
    P("wf_auc satu run = rerata AUC 3 fold; angka yang dilaporkan = mean ± std (ddof = 1) "
      "dari 3 wf_auc.\n")
    f = folds[(folds.ticker == "BTC-USD") & (folds.method == "manual")]
    P("| Seed | Fold 1 | Fold 2 | Fold 3 | Rerata (= wf_auc) |")
    P("|---:|---:|---:|---:|---:|")
    per_seed = []
    for s in (42, 43, 44):
        g = f[f.seed == s].sort_values("fold")
        assert list(g.fold) == [1, 2, 3]
        m = g.auc.mean()
        r = ree[(ree.ticker == "BTC-USD") & (ree.method == "manual") & (ree.seed == s)].iloc[0]
        assert abs(m - r.wf_auc) < 1e-6, (s, m, r.wf_auc)
        per_seed.append(r.wf_auc)
        P(f"| {s} | " + " | ".join(fid(x, 6) for x in g.auc) + f" | {fid(r.wf_auc, 6)} |")
    ps = pd.Series(per_seed)
    P(f"\nmean = ({' + '.join(fid(x, 6) for x in ps)}) / 3 = **{fid(ps.mean(), 6)}**; "
      f"std (ddof = 1) = **{fid(ps.std(ddof=1), 6)}** → dilaporkan "
      f"{fid(ps.mean(), 4)} ± {fid(ps.std(ddof=1), 4)}.\n")

    # ------------------------------------------------------------------ b, c
    data = prepare_ticker_data(cfg, "BTC-USD")
    clean = data["raw_df"]
    r42 = ree[(ree.ticker == "BTC-USD") & (ree.method == "manual") & (ree.seed == 42)].iloc[0]
    bt = pd.read_csv(backtest_file(r42))
    bts = pd.to_datetime(bt.timestamp, utc=True)
    snap = clean.loc[bts]
    rows = []
    for (_, b), (t, x) in zip(bt.iterrows(), snap.iterrows()):
        p = b.model_probability
        raw = "LONG" if p >= long_thr else ("SHORT" if p <= short_thr else "HOLD")
        reasons = confirm_signal(raw, x.to_dict())[1] if raw != "HOLD" else []
        final = raw if raw == "HOLD" or len(reasons) >= min_conf else "HOLD"
        assert final == b.signal, (t, p, raw, reasons, b.signal)
        assert int(x.target) == int(b.actual_label)
        rows.append({"t": t, "p": p, "raw": raw, "n_conf": len(reasons), "reasons": reasons,
                     "final": final, "label": int(b.actual_label), "hit": bool(b.hit), **{c: x[c] for c in FITUR}})
    sig = pd.DataFrame(rows)

    P("## b. Contoh candle backtest BTC-USD Manual seed 42\n")
    P(f"Aturan (`signals/generator.py`): kandidat LONG bila probabilitas ≥ {fid(long_thr, 2)}, "
      f"kandidat SHORT bila ≤ {fid(short_thr, 2)}, selainnya HOLD. Kandidat menjadi sinyal bila "
      f"minimal {min_conf} syarat konfirmasi terpenuhi; bila tidak, HOLD. Syarat LONG: RSI(14) < 50; "
      "MACD histogram > 0; close < BB bawah × 1,005; Stochastic %K < 25; FGI < 0,35. Nilai fitur = "
      "baris t (candle asal t − 15 menit). Tepat: LONG bila label = 1, SHORT bila label = 0; HOLD "
      "tidak dinilai. Keputusan dihitung ulang dan cocok dengan berkas backtest pada 200/200 candle.\n")
    contoh = [
        ("LONG pertama yang terbit", sig[sig.final == "LONG"]),
        (f"HOLD pertama, probabilitas di antara {fid(short_thr, 2)} dan {fid(long_thr, 2)}",
         sig[sig.raw == "HOLD"]),
        (f"HOLD pertama, probabilitas ≥ {fid(long_thr, 2)} tetapi konfirmasi < {min_conf}",
         sig[(sig.raw == "LONG") & (sig.final == "HOLD")]),
    ]
    P("| Besaran | " + " | ".join(c[0] for c in contoh) + " |")
    P("|---|" + "---:|" * len(contoh))
    pick = [c[1].iloc[0] if len(c[1]) else None for c in contoh]

    def cell(r, fn):
        return "tidak ada" if r is None else fn(r)

    P("| Timestamp t | " + " | ".join(cell(r, lambda r: ts(r.t)) for r in pick) + " |")
    P("| Probabilitas | " + " | ".join(cell(r, lambda r: fid(r.p, 6)) for r in pick) + " |")
    for c, d in (("rsi_14", 2), ("macd_hist", 4), ("close", 2), ("bb_lower", 2), ("bb_upper", 2),
                 ("stoch_k", 2), ("fgi_normalized", 2)):
        P(f"| `{c}` | " + " | ".join(cell(r, lambda r, c=c, d=d: fid(r[c], d)) for r in pick) + " |")
    P("| Kandidat (dari probabilitas) | " + " | ".join(cell(r, lambda r: r.raw) for r in pick) + " |")
    P("| Syarat terpenuhi | " + " | ".join(
        cell(r, lambda r: ("; ".join(r.reasons) + f" ({r.n_conf})") if r.raw != "HOLD" else "– (tidak diperiksa)")
        for r in pick) + " |")
    P("| Sinyal akhir | " + " | ".join(cell(r, lambda r: f"**{r.final}**") for r in pick) + " |")
    P("| Label | " + " | ".join(cell(r, lambda r: str(r.label)) for r in pick) + " |")
    P("| Tepat? | " + " | ".join(
        cell(r, lambda r: ("ya" if r.hit else "tidak") if r.final != "HOLD" else "tidak dinilai")
        for r in pick) + " |")
    P("")

    def rekap(s: pd.DataFrame) -> dict:
        return {"kand_long": int((s.raw == "LONG").sum()), "kand_short": int((s.raw == "SHORT").sum()),
                "long": int((s.final == "LONG").sum()), "short": int((s.final == "SHORT").sum()),
                "hold": int((s.final == "HOLD").sum()), "hold_prob": int((s.raw == "HOLD").sum())}

    k = rekap(sig)
    P("## c. Rekap 200 candle BTC-USD Manual seed 42\n")
    P("| Tahap | LONG | SHORT | HOLD |")
    P("|---|---:|---:|---:|")
    P(f"| Kandidat menurut probabilitas | {k['kand_long']} | {k['kand_short']} | {k['hold_prob']} |")
    P(f"| Lolos konfirmasi (sinyal akhir) | {k['long']} | {k['short']} | {k['hold']} |")
    P(f"\nHOLD akhir {k['hold']} = {k['hold_prob']} karena probabilitas di zona tengah + "
      f"{k['kand_long'] - k['long']} kandidat LONG + {k['kand_short'] - k['short']} kandidat SHORT "
      f"yang gagal konfirmasi. Tepat: {int(sig[sig.final != 'HOLD'].hit.sum())} dari "
      f"{k['long'] + k['short']} sinyal terbit.\n")

    # ------------------------------------------------------------------ d
    P("## d. Kandidat dan sinyal akhir, 45 run\n")
    P("Kandidat dari kolom probabilitas berkas backtest; sinyal akhir dari kolom sinyal (sama "
      "dengan `n_signals_issued`/`n_signals_hold` di CSV hasil).\n")
    P("| Koin | Metode | Seed | Kandidat LONG | Kandidat SHORT | Zona tengah | LONG akhir | SHORT akhir | HOLD akhir |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    tot = {m: np.zeros(6, int) for m in METH}
    for t in TICK:
        for m in METH:
            for s in (42, 43, 44):
                r = ree[(ree.ticker == t) & (ree.method == m) & (ree.seed == s)].iloc[0]
                b = pd.read_csv(backtest_file(r))
                p = b.model_probability
                v = np.array([(p >= long_thr).sum(), (p <= short_thr).sum(),
                              ((p > short_thr) & (p < long_thr)).sum(),
                              (b.signal == "LONG").sum(), (b.signal == "SHORT").sum(), (b.signal == "HOLD").sum()])
                assert v[3] + v[4] == r.n_signals_issued and v[5] == r.n_signals_hold
                tot[m] += v
                P(f"| {t} | {NAMA[m]} | {s} | " + " | ".join(str(int(x)) for x in v) + " |")
    P("")
    P("| Metode (15 run) | Kandidat LONG | Kandidat SHORT | Zona tengah | LONG akhir | SHORT akhir | HOLD akhir |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for m in METH:
        P(f"| {NAMA[m]} | " + " | ".join(str(int(x)) for x in tot[m]) + " |")
    P("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Ditulis: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
