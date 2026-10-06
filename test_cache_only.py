"""Uji mode cache-only dan integritas snapshot (Langkah 0 + 2d) — offline.

Jaringan dijebak: yfinance.download dan requests.get meledak bila dipanggil.
 1. Kelima koin terbaca dari snapshot; setiap berkas cocok dengan MANIFEST.md
    (SHA256 dan candle terakhir), tanpa satu pun pemanggilan jaringan.
 2. Berkas yang diubah satu byte -> SnapshotMismatchError (SHA256).
 3. MANIFEST dengan candle terakhir berbeda -> SnapshotMismatchError (candle).
 4. Bundle dalam memori yang candle terakhirnya terpotong -> assert per fase
    (assert_snapshot_bundle) gagal dengan menyebut fasenya.
 5. final_evaluate_and_save dalam mode cache-only tanpa bundle -> gagal
    seketika, sebelum pelatihan apa pun.
 6. Berkas hilang -> CacheMissingError dari preflight, direktori tidak dibuat.
 7. Snapshot dan data/cache tidak berubah setelah semua uji.

Usage: python test_cache_only.py   (kode keluar 1 bila gagal)
"""
from __future__ import annotations

import copy
import hashlib
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import pandas as pd  # noqa: E402

from utils.logger import get_logger  # noqa: E402
get_logger().setLevel(logging.WARNING)

import data.fetcher as F  # noqa: E402
from utils.helpers import load_config  # noqa: E402

CONFIG = "config_snapshot_20260912.yaml"
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "SHIB-USD"]
calls: list[str] = []
fails: list[str] = []


def _boom(name):
    def f(*a, **k):
        calls.append(name)
        raise AssertionError(f"JARINGAN DIPANGGIL: {name}")
    return f


F.yf.download = _boom("yfinance.download")
F.requests.get = _boom("requests.get")


def check(cond: bool, msg: str) -> None:
    print(f"  {'OK  ' if cond else 'GAGAL'} {msg}")
    if not cond:
        fails.append(msg)


def expect(exc_type, fn, msg: str) -> str:
    try:
        fn()
    except exc_type as e:
        check(True, msg)
        return str(e)
    except Exception as e:  # noqa: BLE001
        check(False, f"{msg} (exception lain: {type(e).__name__}: {e})")
        return ""
    check(False, f"{msg} (tidak ada exception)")
    return ""


def digest(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.parquet"))}


def fresh_manifest_cache() -> None:
    F._MANIFEST_CACHE.clear()


cfg = load_config(CONFIG)
SNAP = ROOT / cfg["data"]["cache_dir"]
snap_before, cache_before = digest(SNAP), digest(ROOT / "data" / "cache")

print("1) Kelima koin terbaca dan cocok dengan MANIFEST")
F.require_cached_inputs(cfg, TICKERS)
manifest = F._snapshot_manifest(cfg["data"]["cache_dir"])
check(manifest is not None and len(manifest) == 11, f"MANIFEST terbaca: {len(manifest or {})} berkas")
for t in TICKERS:
    b = F.fetch_all(cfg, t)
    F.assert_snapshot_bundle(cfg, t, b, "uji")
    end = b["primary"].index.max().strftime("%Y-%m-%d %H:%M")
    check(True, f"{t}: SHA256 & candle terakhir cocok (15m berakhir {end} UTC)")
check(not calls, "tidak ada pemanggilan jaringan")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    print("\n2) Satu byte berubah -> SHA256 tidak cocok")
    d2 = tmp / "rusak"
    shutil.copytree(SNAP, d2)
    f = d2 / "ohlcv_yf_ETH_USD_15m_5000.parquet"
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    f.write_bytes(bytes(raw))
    c2 = copy.deepcopy(cfg)
    c2["data"]["cache_dir"] = str(d2)
    fresh_manifest_cache()
    m = expect(F.SnapshotMismatchError, lambda: F.fetch_all(c2, "ETH-USD"),
               "berkas yang diubah ditolak (SnapshotMismatchError)")
    check("SHA256" in m, f"pesan: {m[:90]}…")

    print("\n3) MANIFEST mencatat candle terakhir lain -> ditolak")
    d3 = tmp / "manifest_lain"
    shutil.copytree(SNAP, d3)
    mf = d3 / "MANIFEST.md"
    txt = mf.read_text(encoding="utf-8")
    old_end = manifest["ohlcv_yf_SOL_USD_15m_5000.parquet"]["end"]
    line = next(l for l in txt.splitlines() if "`ohlcv_yf_SOL_USD_15m_5000.parquet`" in l)
    mf.write_text(txt.replace(line, line.replace(old_end, "2026-09-12 23:59")), encoding="utf-8")
    c3 = copy.deepcopy(cfg)
    c3["data"]["cache_dir"] = str(d3)
    fresh_manifest_cache()
    m = expect(F.SnapshotMismatchError, lambda: F.fetch_all(c3, "SOL-USD"),
               "candle terakhir != MANIFEST ditolak")
    check("candle terakhir" in m, f"pesan: {m[:90]}…")

    print("\n4) Bundle terpotong -> assert per fase gagal")
    fresh_manifest_cache()
    b = F.fetch_all(cfg, "LINK-USD")
    b["primary"] = b["primary"].iloc[:-1]
    m = expect(F.SnapshotMismatchError,
               lambda: F.assert_snapshot_bundle(cfg, "LINK-USD", b, "backtest"),
               "bundle yang berakhir 1 candle lebih awal ditolak")
    check("[backtest]" in m, f"pesan menyebut fase: {m[:90]}…")

    print("\n5) final_evaluate_and_save cache-only tanpa bundle -> gagal seketika")
    from optimization.experiment import final_evaluate_and_save, manual_hp_from_cfg  # noqa: E402
    t0 = time.time()
    m = expect(RuntimeError, lambda: final_evaluate_and_save(
        "BTC-USD", "manual", manual_hp_from_cfg(cfg), None, None, None, cfg, tmp, 42,
        persist_artifacts=False, bundle=None), "ditolak tanpa bundle")
    check(time.time() - t0 < 5, f"gagal sebelum pelatihan ({time.time() - t0:.2f} detik)")

    print("\n6) Berkas hilang -> preflight gagal, direktori tidak dibuat")
    ghost = tmp / "tidak_ada"
    c6 = copy.deepcopy(cfg)
    c6["data"]["cache_dir"] = str(ghost)
    fresh_manifest_cache()
    m = expect(F.CacheMissingError, lambda: F.require_cached_inputs(c6, TICKERS),
               "preflight gagal (CacheMissingError)")
    check(m.count(".parquet") == 11, f"menyebut {m.count('.parquet')}/11 berkas hilang")
    check(not ghost.exists(), "direktori palsu tidak dibuat")

check(not calls, "tidak ada pemanggilan jaringan di seluruh uji")
print("\n7) Integritas")
check(digest(SNAP) == snap_before, "snapshot: SHA256 seluruh berkas tidak berubah")
check(digest(ROOT / "data" / "cache") == cache_before, "data/cache: tidak berubah")

print("\nHASIL:", "LULUS" if not fails else f"GAGAL ({len(fails)})")
sys.exit(0 if not fails else 1)
