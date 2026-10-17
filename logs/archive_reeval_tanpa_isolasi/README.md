# Reeval Langkah 4 lama — tanpa isolasi, kondisi numerik bawaan (K1)

Evaluasi akhir 5 koin × 3 metode × seed 42/43/44 yang dijalankan 9 Okt 2026
(`reeval_final.py --seeds 42 43 44`, semua run dalam satu proses, 271,4 menit).
Diarsipkan 17 Okt 2026 dan **tidak dipakai** di tabel skripsi. Penggantinya adalah
reeval final dengan isolasi subprocess dan kondisi K2 (lihat `bahan_bab4/laporan_4.md`
butir i.3). Tidak ada berkas yang dihapus.

| Berkas | Isi |
|---|---|
| `optimization_results_reeval_lama.csv` | 45 baris reeval yang dikeluarkan dari `logs/optimization_results.csv` (15 baris pencarian tetap di sana) |
| `optimization_folds_reeval_lama.csv` | 135 baris metrik per fold (45 run × 3 fold) |
| `backtest_*_20261009_*.csv` | 45 backtest reeval |

## Mengapa diganti

Konfigurasi, data snapshot, seed, dan jalur kode yang sama tidak selalu memberi hasil
yang sama. Pada CPU laptop penelitian (Intel Core i5-12450H: 4 P-core + 4 E-core) dengan
TensorFlow 2.21 + oneDNN, evaluasi akhir sesekali menghasilkan angka lain yang juga
terulang persis. Contoh BTC-USD Manual seed 42: wf_auc 0,648133 (13 sinyal) pada sebagian
besar run, dan 0,664560 (142 sinyal) pada sebagian kecil run. Angka kedua inilah yang
tercatat di reeval lama.

Bukti:

1. `bahan_bab4/uji_isolasi_urutan.md`: mengulang urutan Langkah 4 dalam satu proses
   tidak mereproduksi Langkah 4 (GA cocok, Grid dan Manual tidak), dan subprocess baru pun
   bisa memberi 0,664560. Jadi penyebabnya bukan urutan run. Fitness pencarian identik
   di semua kondisi, sehingga **tahap pencarian tidak terdampak**.
2. `bahan_bab4/uji_penyebab.md`: 8 run per kondisi, setiap run di proses baru.
   K1 (bawaan) 7× 0,648133 + 1× 0,664560; K2 (affinity P-core, oneDNN aktif) 8/8 identik,
   Grid s42 4/4 identik; K3 (oneDNN mati + op determinism) 7× + 1× berbeda.

## Reproduksi

```bash
python tools/uji_isolasi_urutan.py semua
python tools/uji_penyebab.py semua
```
