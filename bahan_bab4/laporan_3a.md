# Laporan Langkah 3a — GA search dan Manual di atas snapshot

Data: `data/snapshot_20260912/` (cache-only, setiap berkas diverifikasi terhadap
MANIFEST). Konfigurasi GA: populasi 10, generasi 6, turnamen 3, crossover 0,8,
mutasi 0,15, elitisme 2, seed 42, `search_epochs` 20. Fitness = AUC pada satu
split validasi kronologis.

Run: BTC-USD Manual pada run pertama (6 Okt 11:30 WIB, diluncurkan dari aplikasi
Claude); ETH, SOL, LINK, SHIB pada run lanjutan dari terminal VS Code (6 Okt
14:13–20:40 WIB, 386,9 menit, kode keluar 0) setelah run pertama terhenti karena
aplikasi Claude ter-restart (lihat `logs/archive_terputus/`). GA BTC-USD
dijalankan ulang dari terminal VS Code (7 Okt 09:32–10:35 WIB, 63,4 menit, kode
keluar 0) agar cara peluncurannya sama dengan koin lain. Hasil run pertamanya
diarsipkan di `logs/archive_ga_btc_run1/`.

## Hasil per koin

Waktu pencarian = `search_duration_min` (GA: seluruh pencarian; Manual: satu
evaluasi). Waktu total = pencarian + evaluasi akhir walk-forward + backtest.

| Koin | Metode | Evaluasi | Pencarian (menit) | Total (menit) | Best fitness | best_hp |
|---|---|---:|---:|---:|---:|---|
| BTC-USD | GA | 46 | 59,9 | 63,2 | 0,6770 | seq 90, u 64/16, drop 0,1, lr 0,01, bs 32 |
| BTC-USD | Manual | 1 | 1,4 | 5,5 | 0,6052 | seq 60, u 128/64, drop 0,2, lr 0,001, bs 32 |
| ETH-USD | GA | 49 | 69,6 | 78,8 | 0,6650 | seq 90, u 64/32, drop 0,4, lr 0,005, bs 32 |
| ETH-USD | Manual | 1 | 2,9 | 12,3 | 0,5535 | baseline |
| SOL-USD | GA | 45 | 108,0 | 115,7 | 0,5806 | seq 120, u 64/48, drop 0,2, lr 0,01, bs 64 |
| SOL-USD | Manual | 1 | 3,1 | 8,4 | 0,5486 | baseline |
| LINK-USD | GA | 47 | 50,1 | 51,6 | 0,5810 | seq 30, u 32/32, drop 0,1, lr 0,0005, bs 64 |
| LINK-USD | Manual | 1 | 1,8 | 6,7 | 0,5379 | baseline |
| SHIB-USD | GA | 46 | 71,9 | 95,0 | 0,6183 | seq 120, u 128/64, drop 0,1, lr 0,01, bs 32 |
| SHIB-USD | Manual | 1 | 3,1 | 16,5 | 0,5860 | baseline |

Total pencarian GA lima koin: 359,6 menit untuk 233 evaluasi (rata-rata
1,54 menit per evaluasi; rincian per evaluasi di `ga_trace_*`).

`val_auc_search` Manual identik dengan pengukuran sebelumnya pada pipeline yang
sama (uji dampak kebocoran dan uji determinisme), sehingga reproduksibilitas
lintas run kembali terkonfirmasi.

## Kelengkapan dan konsistensi jejak GA

| Koin | Baris jejak | Turnamen | Evaluasi (non-cache) | Dari cache | Pemeriksaan |
|---|---:|---:|---:|---:|---|
| BTC-USD | 70 | 96 | 46 | 24 | lengkap & konsisten |
| ETH-USD | 70 | 96 | 49 | 21 | lengkap & konsisten |
| SOL-USD | 70 | 96 | 45 | 25 | lengkap & konsisten |
| LINK-USD | 70 | 96 | 47 | 23 | lengkap & konsisten |
| SHIB-USD | 70 | 96 | 46 | 24 | lengkap & konsisten |

Pemeriksaan per koin: 7 generasi × 10 individu; jumlah evaluasi non-cache =
`n_evals` = `n_evals` akhir `ga_history`; turnamen = 2 × 48 anak; setiap anak
terekonstruksi dari induk + mask crossover + mutasi tercatat; setiap pemenang
turnamen = peserta dengan fitness tertinggi; individu terbaik dan best_hp di
jejak = baris hasil (fitness di CSV hasil dibulatkan 6 desimal); jumlah
`waktu_detik` jejak = waktu pencarian.

**Reproduksibilitas lintas run.** GA ETH pada run lanjutan mereproduksi
generasi 0–5 dari run yang terputus secara identik: `ga_history` dan 60 baris
`ga_trace` (kromosom, fitness, induk, mask, mutasi). GA BTC run ulang identik
dengan run pertama di seluruh 70 baris jejak, 96 turnamen, `ga_history`,
backtest, dan baris hasil; hanya waktunya berbeda (pencarian 59,9 vs 53,5
menit).

**Catatan waktu GA BTC.** Pada 46 evaluasi yang sama, rasio waktu run ulang :
run pertama bermedian 0,89. Evaluasi 1–11 dan 41–44 run ulang ±1,3–2,3× lebih
lambat karena laptop sedang dipakai. Waktu run ulang dipakai apa adanya,
sebagai cerminan kondisi laptop penelitian (keputusan peneliti, 7 Okt 2026).

## Jendela data

`tools/cek_jendela_data.py --semua-harus-cocok`: **10/10** backtest baru tepat
pada snapshot (candle terakhir dan 200 harga close per backtest).

## Perbandingan dengan run Juli (arsip)

| Koin | best_hp Juli | Fitness Juli | best_hp kini | Fitness kini | Gen sama |
|---|---|---:|---|---:|---:|
| BTC-USD | seq 60, u 64/48, drop 0,4, lr 0,001, bs 32 | 0,6297 | seq 90, u 64/16, drop 0,1, lr 0,01, bs 32 | 0,6770 | 2/6 |
| ETH-USD | seq 120, u 32/16, drop 0,1, lr 0,005, bs 64 | 0,6684 | seq 90, u 64/32, drop 0,4, lr 0,005, bs 32 | 0,6650 | 1/6 |
| SOL-USD | seq 120, u 64/48, drop 0,2, lr 0,01, bs 32 | 0,6263 | seq 120, u 64/48, drop 0,2, lr 0,01, bs 64 | 0,5806 | 5/6 |
| LINK-USD | seq 120, u 96/48, drop 0,2, lr 0,005, bs 32 | 0,5808 | seq 30, u 32/32, drop 0,1, lr 0,0005, bs 64 | 0,5810 | 0/6 |
| SHIB-USD | seq 90, u 64/32, drop 0,2, lr 0,005, bs 64 | 0,6518 | seq 120, u 128/64, drop 0,1, lr 0,01, bs 32 | 0,6183 | 0/6 |

Angka Juli tidak sebanding (jendela data berbeda dan pipeline sebelum
perbaikan 1–5); tabel ini hanya untuk keterlacakan.
