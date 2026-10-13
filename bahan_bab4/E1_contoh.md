# Bahan E1 — contoh baris dan bahan pendukung (BTC-USD)

Dihasilkan oleh `tools/buat_e1_contoh.py` pada 2026-10-13 09:25 UTC dari kode commit `fa9a888`. Data: snapshot `data/snapshot_20260912/` lewat `config_snapshot_20260912.yaml` (cache-only, diverifikasi SHA256 terhadap `MANIFEST.md`). Tanpa pelatihan model. Waktu dalam UTC; angka memakai titik ribuan dan koma desimal.

## 1. Contoh baris BTC-USD

Aturan pemilihan (deterministik): baris **pertama** berlabel 1 dan baris **pertama** berlabel 0 pada bagian latih dengan indeks ≥ 119 (indeks 0-based di dalam bagian latih). Dengan begitu kedua baris punya jendela penuh dan menjadi target y(t) untuk semua kandidat panjang sekuens (30, 45, 60, 90, 120). Bagian latih = 3.464 baris (2026-07-22 16:15 – 2026-08-27 18:00).

| | Contoh berlabel 1 | Contoh berlabel 0 |
|---|---:|---:|
| Indeks di bagian latih | 129 | 119 |
| Indeks di data bersih | 129 | 119 |

### 1a. Pembentukan label

Label 1 bila close(t+4) > close(t) × 1,003.

| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |
|---|---:|---:|
| Timestamp t | 2026-07-24 00:30 | 2026-07-23 22:00 |
| close(t) | 64.800,61 | 65.050,00 |
| Timestamp t+4 | 2026-07-24 01:30 | 2026-07-23 23:00 |
| close(t+4) | 65.052,19 | 65.137,68 |
| close(t) × 1,003 | 64.995,01 | 65.245,15 |
| Perubahan close(t+4) terhadap close(t) | 0,3882% | 0,1348% |
| close(t+4) > close(t) × 1,003? | ya | tidak |
| **Label** | **1** | **0** |

### 1b. Nilai 25 fitur pada baris t

Setelah penggeseran satu langkah (`apply_anti_leakage`), baris t berisi fitur candle **t − 15 menit**: 2026-07-24 00:15 (contoh label 1) dan 2026-07-23 21:45 (contoh label 0). Ke-25 nilai diperiksa identik dengan matriks fitur pada candle itu. Kolom *norm.* = MinMaxScaler [0, 1] yang di-fit hanya pada bagian latih (lihat bagian 2); model menerima nilai float32.

| # | Fitur | Candle asal | Label 1: mentah | Label 1: norm. | Candle asal | Label 0: mentah | Label 0: norm. |
|---:|---|---|---:|---:|---|---:|---:|
| 1 | `open` | 2026-07-24 00:15 | 64.879,13 | 0,141011 | 2026-07-23 21:45 | 65.054,43 | 0,150360 |
| 2 | `high` | 2026-07-24 00:15 | 64.940,98 | 0,136756 | 2026-07-23 21:45 | 65.120,19 | 0,146245 |
| 3 | `low` | 2026-07-24 00:15 | 64.829,00 | 0,141179 | 2026-07-23 21:45 | 65.054,43 | 0,153332 |
| 4 | `close` | 2026-07-24 00:15 | 64.831,35 | 0,137272 | 2026-07-23 21:45 | 65.120,19 | 0,152674 |
| 5 | `volume` | 2026-07-24 00:15 | 0,000000 | 0,000000 | 2026-07-23 21:45 | 0,000000 | 0,000000 |
| 6 | `rsi_14` | 2026-07-24 00:15 | 39,4718 | 0,349666 | 2026-07-23 21:45 | 58,3901 | 0,564180 |
| 7 | `macd` | 2026-07-24 00:15 | 10,3681 | 0,277167 | 2026-07-23 21:45 | 32,0532 | 0,291418 |
| 8 | `macd_signal` | 2026-07-24 00:15 | 32,3687 | 0,313896 | 2026-07-23 21:45 | -17,4209 | 0,275867 |
| 9 | `macd_hist` | 2026-07-24 00:15 | -22,0006 | 0,351513 | 2026-07-23 21:45 | 49,4741 | 0,460382 |
| 10 | `bb_upper` | 2026-07-24 00:15 | 65.287,96 | 0,135506 | 2026-07-23 21:45 | 65.207,53 | 0,131174 |
| 11 | `bb_lower` | 2026-07-24 00:15 | 64.772,28 | 0,153182 | 2026-07-23 21:45 | 64.535,78 | 0,140236 |
| 12 | `bb_pct` | 2026-07-24 00:15 | 0,114559 | 0,284437 | 2026-07-23 21:45 | 0,869978 | 0,677839 |
| 13 | `bb_width` | 2026-07-24 00:15 | 0,007930 | 0,068508 | 2026-07-23 21:45 | 0,010355 | 0,092629 |
| 14 | `atr_14` | 2026-07-24 00:15 | 125,2975 | 0,167726 | 2026-07-23 21:45 | 134,0289 | 0,181609 |
| 15 | `ema_9` | 2026-07-24 00:15 | 65.006,45 | 0,139358 | 2026-07-23 21:45 | 65.020,46 | 0,140131 |
| 16 | `ema_21` | 2026-07-24 00:15 | 65.005,77 | 0,135131 | 2026-07-23 21:45 | 64.954,28 | 0,132243 |
| 17 | `ema_50` | 2026-07-24 00:15 | 65.061,51 | 0,135274 | 2026-07-23 21:45 | 65.066,55 | 0,135567 |
| 18 | `stoch_k` | 2026-07-24 00:15 | 18,4627 | 0,180197 | 2026-07-23 21:45 | 86,5285 | 0,872761 |
| 19 | `stoch_d` | 2026-07-24 00:15 | 43,3255 | 0,433063 | 2026-07-23 21:45 | 82,9282 | 0,841301 |
| 20 | `vol_ratio` | 2026-07-24 00:15 | 0,000000 | 0,000000 | 2026-07-23 21:45 | 0,000000 | 0,000000 |
| 21 | `rsi_1h` | 2026-07-24 00:15 | 43,0627 | 0,306372 | 2026-07-23 21:45 | 44,6463 | 0,328838 |
| 22 | `ema_21_1h` | 2026-07-24 00:15 | 65.204,19 | 0,141609 | 2026-07-23 21:45 | 65.227,19 | 0,142990 |
| 23 | `macd_hist_1h` | 2026-07-24 00:15 | 29,0481 | 0,451437 | 2026-07-23 21:45 | -0,638275 | 0,413119 |
| 24 | `fgi_normalized` | 2026-07-24 00:15 | 0,280000 | 0,061225 | 2026-07-23 21:45 | 0,310000 | 0,122449 |
| 25 | `fgi_classification_encoded` | 2026-07-24 00:15 | 1,0000 | 0,333333 | 2026-07-23 21:45 | 1,0000 | 0,333333 |

Tiga fitur 1 jam berasal dari bar 1 jam pada 1c, dan dua fitur FGI dari nilai harian pada 1d. Keduanya sudah melekat pada baris candle asal sebelum penggeseran.

### 1c. Bar 1 jam yang dipakai baris t

Yahoo menstempel bar 1 jam dengan waktu **mulai** h; bar itu baru diketahui saat tutup pada h + 1 jam. Candle asal r diketahui saat tutup pada r + 15 menit (= t), sehingga dipakai bar 1 jam terakhir dengan h + 1 jam ≤ t.

| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |
|---|---:|---:|
| Candle asal r (diketahui r + 15 m) | 2026-07-24 00:15 (2026-07-24 00:30) | 2026-07-23 21:45 (2026-07-23 22:00) |
| Stempel bar 1 jam h | 2026-07-23 23:00 | 2026-07-23 21:00 |
| Waktu tutup h + 1 jam | 2026-07-24 00:00 | 2026-07-23 22:00 |
| Bar berikutnya tutup pada (belum boleh dipakai) | 2026-07-24 01:00 | 2026-07-23 23:00 |
| `rsi_1h` | 43,0627 | 44,6463 |
| `ema_21_1h` | 65.204,19 | 65.227,19 |
| `macd_hist_1h` | 29,0481 | -0,638275 |

Ketiga nilai diperiksa identik dengan indikator yang dihitung langsung dari bar 1 jam h.

### 1d. Fear & Greed Index

FGI harian dipetakan ke candle berdasarkan tanggal UTC candle asal (nilai tanggal terakhir yang tersedia). `fgi_normalized` = nilai / 100; kode ordinal: Extreme Fear = 0, Fear = 1, Neutral = 2, Greed = 3, Extreme Greed = 4, lainnya = −1.

| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |
|---|---:|---:|
| Tanggal FGI (UTC) | 2026-07-24 | 2026-07-23 |
| Nilai mentah (0–100) | 28 | 31 |
| Kategori | Fear | Fear |
| Kode ordinal (`fgi_classification_encoded`) | 1 | 1 |
| `fgi_normalized` | 0,28 | 0,31 |

### 1e. Jendela L = 60 untuk target y(t)

`sequence_window(X, i, 60)` = baris i − 59 … i pada bagian latih (baris t ikut). Karena penggeseran, isi fiturnya berasal dari candle 15 menit lebih awal.

| Besaran | Contoh berlabel 1 | Contoh berlabel 0 |
|---|---:|---:|
| Baris pertama jendela | 2026-07-23 09:45 | 2026-07-23 07:15 |
| Baris terakhir jendela (= t) | 2026-07-24 00:30 | 2026-07-23 22:00 |
| Candle asal fitur: pertama – terakhir | 2026-07-23 09:30 – 2026-07-24 00:15 | 2026-07-23 07:00 – 2026-07-23 21:45 |
| Indeks bagian latih | 70 – 129 | 60 – 119 |

## 2. Minimum dan maksimum bagian latih BTC-USD (dasar MinMaxScaler)

`fit_scaler(X_latih)`, X_latih = 3.464 baris × 25 fitur (float32). x_norm = (x − min) / (maks − min).

| # | Fitur | Minimum | Maksimum | Maks − min |
|---:|---|---:|---:|---:|
| 1 | `open` | 62.235,11 | 80.985,61 | 18.750,50 |
| 2 | `high` | 62.357,99 | 81.245,53 | 18.887,54 |
| 3 | `low` | 62.210,15 | 80.760,00 | 18.549,85 |
| 4 | `close` | 62.257,03 | 81.010,45 | 18.753,42 |
| 5 | `volume` | 0,000000 | 5.466.898.432 | 5.466.898.432 |
| 6 | `rsi_14` | 8,6345 | 96,8255 | 88,1911 |
| 7 | `macd` | -411,3643 | 1.110,22 | 1.521,58 |
| 8 | `macd_signal` | -378,6056 | 930,6647 | 1.309,27 |
| 9 | `macd_hist` | -252,7772 | 403,7457 | 656,5229 |
| 10 | `bb_upper` | 62.772,13 | 81.338,28 | 18.566,15 |
| 11 | `bb_lower` | 61.974,06 | 80.241,33 | 18.267,27 |
| 12 | `bb_pct` | -0,431623 | 1,4886 | 1,9202 |
| 13 | `bb_width` | 0,001042 | 0,101579 | 0,100536 |
| 14 | `atr_14` | 19,8115 | 648,7306 | 628,9191 |
| 15 | `ema_9` | 62.478,83 | 80.616,35 | 18.137,52 |
| 16 | `ema_21` | 62.596,01 | 80.428,87 | 17.832,86 |
| 17 | `ema_50` | 62.736,17 | 79.925,94 | 17.189,77 |
| 18 | `stoch_k` | 0,752852 | 99,0336 | 98,2808 |
| 19 | `stoch_d` | 1,3145 | 98,3233 | 97,0088 |
| 20 | `vol_ratio` | 0,000000 | 18,8633 | 18,8633 |
| 21 | `rsi_1h` | 21,4668 | 91,9559 | 70,4891 |
| 22 | `ema_21_1h` | 62.846,02 | 79.498,77 | 16.652,75 |
| 23 | `macd_hist_1h` | -320,7014 | 454,0460 | 774,7474 |
| 24 | `fgi_normalized` | 0,250000 | 0,740000 | 0,490000 |
| 25 | `fgi_classification_encoded` | 0,000000 | 3,0000 | 3,0000 |

**Contoh perhitungan** (`close`, contoh berlabel 1, candle asal 2026-07-24 00:15):

x_norm = (64.831,3516 − 62.257,0312) / (81.010,4531 − 62.257,0312) = 2.574,3203 / 18.753,4219 = **0,137272**

Keluaran `scaler.transform`: 0,137272 (selisih 8.8e-08, pembulatan float32).

## 3. Pengambilan data (dari `MANIFEST.md`)

Waktu pengambilan = mtime asli berkas cache saat diunduh dari Yahoo Finance (MANIFEST mencatatnya dalam WIB, UTC+7). Snapshot sendiri disalin dari cache pada 2026-10-06 02:34 UTC.

| Koin | Diambil (WIB) | Diambil (UTC) | Candle 15m pertama | Candle 15m terakhir | Candle 15m terakhir tutup | Bar 1 jam pertama | Bar 1 jam terakhir |
|---|---|---|---|---|---|---|---|
| BTC-USD | 2026-09-12 12:36 | 2026-09-12 05:36 | 2026-07-22 03:45 | 2026-09-12 05:30 | 2026-09-12 05:45 | 2026-02-15 21:00 | 2026-09-12 05:00 |
| ETH-USD | 2026-09-12 12:57 | 2026-09-12 05:57 | 2026-07-22 04:00 | 2026-09-12 05:45 | 2026-09-12 06:00 | 2026-02-15 21:00 | 2026-09-12 05:00 |
| SOL-USD | 2026-09-12 13:45 | 2026-09-12 06:45 | 2026-07-22 05:00 | 2026-09-12 06:45 | 2026-09-12 07:00 | 2026-02-15 22:00 | 2026-09-12 06:00 |
| LINK-USD | 2026-09-12 14:49 | 2026-09-12 07:49 | 2026-07-22 06:00 | 2026-09-12 07:45 | 2026-09-12 08:00 | 2026-02-15 23:00 | 2026-09-12 07:00 |
| SHIB-USD | 2026-09-12 15:51 | 2026-09-12 08:51 | 2026-07-22 07:00 | 2026-09-12 08:45 | 2026-09-12 09:00 | 2026-02-16 00:00 | 2026-09-12 08:00 |

Fear & Greed Index: 90 nilai harian 2026-06-15 – 2026-09-12, diambil 2026-09-12 15:34 WIB (2026-09-12 08:34 UTC).

Stempel candle adalah waktu **mulai** candle. Kecocokan candle terakhir 15m di tabel ini dengan isi snapshot diperiksa oleh `data/fetcher.py` setiap kali berkas dimuat.

**Catatan.** Pada 5 dari 5 koin (BTC-USD, ETH-USD, SOL-USD, LINK-USD, SHIB-USD), waktu pengambilan (mtime) lebih awal daripada waktu tutup candle 15m terakhir, sehingga candle terakhir kemungkinan belum lengkap saat diunduh. Candle itu tidak pernah menjadi baris fitur (penggeseran membuang baris sesudahnya) dan tidak berlabel; satu-satunya pemakaiannya adalah sebagai close(t+4) untuk label baris bersih terakhir (t = candle terakhir − 4 × 15 menit), yang berada di bagian uji, jendela backtest, dan fold walk-forward terakhir. Bar 1 jam terakhir juga belum tutup, tetapi tidak pernah dipakai karena penyelarasan hanya mengambil bar yang sudah tutup.

## 4. Jumlah sekuens BTC-USD per kandidat panjang sekuens

Pembagian kronologis 70/15/15 pada 4.946 baris bersih: latih 3.464, validasi 741, uji 741 baris. Sekuens dibentuk di dalam tiap bagian (`create_sequences`), jumlahnya n − L + 1. Ini skema evaluasi fitness (pencarian) dan model pemenang.

| L | Sekuens latih | Sekuens validasi | % label 1 latih | % label 1 validasi |
|---:|---:|---:|---:|---:|
| 30 | 3.435 | 712 | 13,36% | 16,29% |
| 45 | 3.420 | 697 | 13,42% | 16,50% |
| 60 | 3.405 | 682 | 13,48% | 16,86% |
| 90 | 3.375 | 652 | 13,54% | 16,41% |
| 120 | 3.345 | 622 | 13,54% | 16,72% |
