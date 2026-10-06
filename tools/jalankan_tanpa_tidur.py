#!/usr/bin/env python
"""Jalankan perintah panjang (mis. run optimasi semalam) sambil mencegah Windows
masuk idle sleep, dan catat seluruh keluarannya ke berkas log.

Memakai SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED): permintaan
sementara milik proses ini yang dilepas otomatis saat proses selesai. Pengaturan
daya tidak diubah. Menutup lid atau memilih Sleep secara manual tetap membuat
laptop tidur.

Usage:
    python tools/jalankan_tanpa_tidur.py logs/run_3a.log -- python run_optimization.py --methods ga manual
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    if "--" not in sys.argv or sys.argv.index("--") != 2:
        sys.exit(__doc__)
    log = Path(sys.argv[1])
    cmd = sys.argv[3:]
    log.parent.mkdir(parents=True, exist_ok=True)

    awake = False
    if sys.platform == "win32":
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"===== MULAI {now()} | cegah tidur: {'ya' if awake else 'TIDAK'} | {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"\n===== SELESAI {now()} | kode keluar {rc} | durasi {(time.time() - t0) / 60:.1f} menit\n")

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return rc


if __name__ == "__main__":
    sys.exit(main())
