# Arsip pra-snapshot

Hasil eksperimen optimasi hyperparameter yang dibuat **sebelum** data dibekukan
pada snapshot `data/snapshot_20260912/` dan **sebelum** perbaikan pipeline 1–5.
Disimpan untuk keterlacakan saja — **tidak dipakai** di tabel skripsi dan
**tidak sebanding** dengan hasil yang dibuat sesudahnya.

Diarsipkan pada 2026-10-06 (butir 3). Tidak ada berkas yang dihapus.

## Isi

| Berkas | Jumlah | Keterangan |
|---|---:|---|
| `optimization_results.csv` | 1 (74 baris) | Master hasil lama, lihat klasifikasi di bawah |
| `optimization_results_backup_*.csv` | 2 | Cadangan otomatis saat migrasi header (13→16 dan 16→18 kolom) |
| `optimization_summary_*.csv` | 7 | Ringkasan per run |
| `optimization_{KOIN}_*.csv` | 12 | Tabel per koin per run |
| `ga_history_*.csv` | 10 | Riwayat GA per generasi (5 lengkap, 2 uji `--fast`, 3 SOL terputus) |
| `grid_history_*.csv` | 5 | Riwayat Grid Search 11 Sep (stride relatif prima) |
| `backtest_*.csv` | 74 | Backtest per run (22 GA, 27 Grid, 25 Manual) |
| `cek_jendela_data.txt` | 1 | Keluaran `tools/cek_jendela_data.py` atas folder ini |

Artefak model lama dipindahkan ke `model/optimized/_archive_pra_snapshot/`
(diabaikan git), agar tidak tertimpa run baru.

## Klasifikasi 74 baris `optimization_results.csv`

| Kelompok | Baris | Waktu (UTC) | Keterangan |
|---|---:|---|---|
| A | 6 | 29 Jul | Uji `--fast` (`search_epochs` = 2) — bukan hasil penelitian |
| B | 15 | 30–31 Jul | Run penuh GA, Grid (stride lama yang membekukan `batch_size`), Manual |
| C | 5 | 11 Sep | Grid Search diulang dengan stride relatif prima |
| D | 3 | 12 Sep 04:28–04:46 | **Uji coba duplikat** reeval: BTC-USD manual seed 42 (04:28:34), seed 42 (04:43:15), seed 43 (04:46:38) |
| E | 45 | 12 Sep 04:56–08:51 | Reeval 5 koin × 3 metode × 3 seed |

## Mengapa tidak dipakai

1. **Jendela data berbeda-beda** (dibuktikan pada Langkah 1 dengan sidik jari
   backtest, lihat `cek_jendela_data.txt`). Yahoo membatasi riwayat 15m ke 60
   hari dan cache ber-TTL 15 menit, sehingga setiap run memakai jendela yang
   bergeser:
   - Pencarian GA dan Manual (B): data 30–31 Jul (≈ 7 Jun – 31 Jul).
   - Pencarian Grid (C): data 11 Sep sore, ±46–51 candle sebelum snapshot.
   - Reeval (E): walk-forward 1–4 candle sebelum snapshot; backtest bergeser
     selama run karena unduh ulang di tengah jalan. Hanya 12 dari 74 backtest
     yang tepat pada snapshot.
   - `val_auc_search` pada baris reeval disalin dari baris pencarian, sehingga
     satu baris bisa memuat dua jendela berbeda.
2. **Pipeline berubah** sesudahnya (commit `dfda464`, `37dc110`): penjajaran
   jendela (info terbaru t−1, bukan t−2), `vol_ratio` saat SMA-20 = 0, FGI
   ordinal, dan perbaikan kebocoran fitur 1 jam (hingga 45 menit data masa
   depan). Seluruh angka di sini dihitung dengan pipeline lama.
3. Grid kelompok B memakai stride yang hanya menguji `batch_size` = 16.

## Reproduksi bukti

```bash
python tools/cek_jendela_data.py --dir logs/archive_pra_snapshot
```
