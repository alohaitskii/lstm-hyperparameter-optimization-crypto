# Laporan Langkah 4 — Evaluasi akhir 3 seed di atas snapshot

Data: `data/snapshot_20260912/` (cache-only, diverifikasi terhadap MANIFEST).
Setiap konfigurasi terpilih (GA, Grid, Manual; 5 koin) dievaluasi ulang dengan
`optimization/reeval_final.py --seeds 42 43 44`: walk-forward 3 fold
(`TimeSeriesSplit`, maks. 100 epoch, patience 10) + backtest 200 candle.
Run: 17 Okt 2026, 17.34–20.36 WIB (10:34–13:36 UTC), proses lepas dari terminal (181,9 menit, kode keluar 0). Kondisi numerik **K2** (affinity proses worker ke P-core (logical 0–7, mask 0xff), oneDNN aktif); isolasi subprocess per run: **ya**. Lihat butir i.3.

Semua std = simpangan baku sampel (ddof = 1) antar 3 seed. Dihasilkan oleh
`tools/buat_laporan_4.py`.

## Verifikasi

- `optimization_results.csv`: 15 baris pencarian + **45 baris reeval** (5 koin × 3 metode × seed 42/43/44), tanpa duplikat.
- `val_auc_search`, `search_duration_min`, `n_evals`, `best_hp_json`, `search_epochs` pada 45 baris = baris pencarian snapshot (GA BTC: 59,86 menit, run ulang 7 Okt).
- `optimization_folds.csv`: 135 baris (45 run × 3 fold), tanpa duplikat.
- Jumlah LONG+SHORT dan HOLD di 45 berkas backtest = kolom `n_signals_*` di CSV hasil.
- `tools/cek_jendela_data.py --semua-harus-cocok`: **60/60** backtest (15 pencarian + 45 reeval) tepat pada snapshot.

**Catatan seed 42.** Metrik reeval seed 42 tidak sama dengan metrik evaluasi akhir pada baris pencarian (selisih |wf_auc| maks. 0,0269, median 0,0098). Evaluasi akhir di `run_optimization.py` berjalan langsung sesudah pencarian tanpa reset seed, dan dengan kondisi numerik K1 yang terbukti tidak deterministik antarproses (butir i.3). Angka yang dilaporkan adalah 45 baris reeval.

## a. Metrik walk-forward dan backtest (mean ± std, 3 seed)

AUC, F1, precision, recall = rerata 3 fold walk-forward. Hit-rate = proporsi sinyal LONG/SHORT yang benar pada backtest 200 candle. Seed tanpa sinyal tidak punya hit-rate (tercatat NaN di CSV, bukan 0) dan tidak ikut dihitung; n = jumlah seed yang menerbitkan sinyal. Std dengan n = 1 tidak terdefinisi (–).

| Koin | Metode | AUC | F1 | Precision | Recall | Hit-rate |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | GA | 0,6363 ± 0,0137 | 0,1744 ± 0,0547 | 0,1276 ± 0,0748 | 0,4937 ± 0,0344 | 0,1233 ± 0,0034 (n=3) |
| BTC-USD | Grid | 0,6424 ± 0,0145 | 0,1797 ± 0,0481 | 0,2104 ± 0,0212 | 0,4998 ± 0,1057 | 0,1278 ± 0,0014 (n=2) |
| BTC-USD | Manual | 0,6244 ± 0,0276 | 0,1023 ± 0,0431 | 0,1287 ± 0,0995 | 0,5409 ± 0,1861 | 0,1396 ± 0,0136 (n=3) |
| ETH-USD | GA | 0,6082 ± 0,0100 | 0,2763 ± 0,0666 | 0,2148 ± 0,0354 | 0,7807 ± 0,1975 | 0,2434 ± 0,0367 (n=3) |
| ETH-USD | Grid | 0,5946 ± 0,0072 | 0,3011 ± 0,0332 | 0,2160 ± 0,0026 | 0,7568 ± 0,1136 | 0,2628 ± 0,0772 (n=3) |
| ETH-USD | Manual | 0,5976 ± 0,0194 | 0,1977 ± 0,1164 | 0,2190 ± 0,0475 | 0,5757 ± 0,2041 | 0,2718 ± 0,0467 (n=3) |
| SOL-USD | GA | 0,5702 ± 0,0261 | 0,3431 ± 0,0581 | 0,2532 ± 0,0083 | 0,7965 ± 0,1759 | 0,2061 ± 0,0086 (n=2) |
| SOL-USD | Grid | 0,5738 ± 0,0103 | 0,3177 ± 0,0665 | 0,2240 ± 0,0458 | 0,7115 ± 0,2191 | 0,2039 ± 0,0142 (n=3) |
| SOL-USD | Manual | 0,5654 ± 0,0085 | 0,3101 ± 0,0648 | 0,2515 ± 0,0561 | 0,5164 ± 0,1603 | 0,2121 ± 0,0000 (n=3) |
| LINK-USD | GA | 0,5631 ± 0,0092 | 0,3678 ± 0,0464 | 0,3127 ± 0,0087 | 0,6495 ± 0,0809 | 0,2454 ± 0,0510 (n=3) |
| LINK-USD | Grid | 0,5635 ± 0,0065 | 0,4151 ± 0,0341 | 0,3185 ± 0,0237 | 0,7937 ± 0,1483 | 0,3327 ± 0,1292 (n=3) |
| LINK-USD | Manual | 0,5800 ± 0,0126 | 0,3217 ± 0,0270 | 0,3529 ± 0,0392 | 0,4578 ± 0,0440 | 0,2809 ± 0,0905 (n=3) |
| SHIB-USD | GA | 0,5428 ± 0,0147 | 0,1871 ± 0,1534 | 0,2052 ± 0,1265 | 0,3672 ± 0,3241 | 0,5736 ± 0,2809 (n=2) |
| SHIB-USD | Grid | 0,5395 ± 0,0055 | 0,2706 ± 0,0217 | 0,2094 ± 0,0701 | 0,5202 ± 0,0720 | 0,3959 ± 0,0138 (n=3) |
| SHIB-USD | Manual | 0,5531 ± 0,0168 | 0,2015 ± 0,0527 | 0,1536 ± 0,0130 | 0,4394 ± 0,1709 | 0,3926 ± 0,1817 (n=3) |

**Rerata lintas koin** (rerata dari mean per koin, setiap koin berbobot sama; ± = std antar koin). Baris *5 koin*: semua metrik; hit-rate dari mean per koin yang terdefinisi (terdefinisi di kelima koin). Baris *4 koin*: hit-rate pada BTC, ETH, SOL, LINK, cakupan yang ditetapkan saat SHIB-USD GA tidak bersinyal pada reeval lama; dipertahankan agar sebanding. Mean per koin dengan seed bersinyal < 3: BTC-USD Grid (n=2), SHIB-USD GA (n=2), SOL-USD GA (n=2).

| Metode | Cakupan | AUC | F1 | Precision | Recall | Hit-rate |
|---|---|---:|---:|---:|---:|---:|
| GA | 5 koin | 0,5841 ± 0,0375 | 0,2698 ± 0,0880 | 0,2227 ± 0,0679 | 0,6175 ± 0,1855 | 0,2784 ± 0,1723 |
| GA | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2045 ± 0,0571 (4 koin) |
| Grid | 5 koin | 0,5828 ± 0,0388 | 0,2968 ± 0,0849 | 0,2357 ± 0,0466 | 0,6564 ± 0,1370 | 0,2646 ± 0,1052 |
| Grid | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2318 ± 0,0870 (4 koin) |
| Manual | 5 koin | 0,5841 ± 0,0280 | 0,2267 ± 0,0907 | 0,2212 ± 0,0886 | 0,5060 ± 0,0569 | 0,2594 ± 0,0934 |
| Manual | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2261 ± 0,0652 (4 koin) |

## b. Sinyal backtest LONG / SHORT / HOLD (200 candle)

| Koin | Metode | LONG (mean) | SHORT (mean) | HOLD (mean) | Per seed 42 · 43 · 44 (L/S/H) |
|---|---|---:|---:|---:|---:|
| BTC-USD | GA | 126,7 | 0,0 | 73,3 | 138/0/62 · 142/0/58 · 100/0/100 |
| BTC-USD | Grid | 91,3 | 0,0 | 108,7 | 0/0/200 · 142/0/58 · 132/0/68 |
| BTC-USD | Manual | 83,0 | 0,0 | 117,0 | 13/0/187 · 94/0/106 · 142/0/58 |
| ETH-USD | GA | 87,0 | 0,0 | 113,0 | 99/0/101 · 99/0/101 · 63/0/137 |
| ETH-USD | Grid | 83,7 | 0,0 | 116,3 | 54/0/146 · 98/0/102 · 99/0/101 |
| ETH-USD | Manual | 77,3 | 0,0 | 122,7 | 54/0/146 · 79/0/121 · 99/0/101 |
| SOL-USD | GA | 36,3 | 0,0 | 163,7 | 99/0/101 · 10/0/190 · 0/0/200 |
| SOL-USD | Grid | 92,7 | 0,0 | 107,3 | 80/0/120 · 99/0/101 · 99/0/101 |
| SOL-USD | Manual | 99,0 | 0,0 | 101,0 | 99/0/101 · 99/0/101 · 99/0/101 |
| LINK-USD | GA | 64,7 | 2,3 | 133,0 | 66/0/134 · 26/7/167 · 102/0/98 |
| LINK-USD | Grid | 52,0 | 0,0 | 148,0 | 13/0/187 · 128/0/72 · 15/0/185 |
| LINK-USD | Manual | 61,0 | 0,0 | 139,0 | 21/0/179 · 127/0/73 · 35/0/165 |
| SHIB-USD | GA | 2,7 | 33,7 | 163,7 | 8/0/192 · 0/0/200 · 0/101/99 |
| SHIB-USD | Grid | 33,3 | 0,0 | 166,7 | 18/0/182 · 31/0/169 · 51/0/149 |
| SHIB-USD | Manual | 42,3 | 0,0 | 157,7 | 84/0/116 · 38/0/162 · 5/0/195 |

Total 15 backtest per metode: GA 952 LONG / 108 SHORT / 1940 HOLD; Grid 1059 LONG / 0 SHORT / 1941 HOLD; Manual 1088 LONG / 0 SHORT / 1912 HOLD.

**Rerata per backtest per metode** (rerata dari mean per koin, setiap koin berbobot sama; ± = std antar koin):

| Metode | Cakupan | LONG | SHORT | HOLD | Terbit (LONG+SHORT) | Run tanpa sinyal |
|---|---|---:|---:|---:|---:|---:|
| GA | 5 koin | 63,5 ± 47,4 | 7,2 ± 14,8 | 129,3 ± 38,0 | 70,7 ± 38,0 | 2 dari 15 |
| GA | 4 koin (BTC, ETH, SOL, LINK) | 78,7 ± 38,1 | 0,6 ± 1,2 | 120,8 ± 37,9 | 79,3 ± 37,9 | 1 dari 12 |
| Grid | 5 koin | 70,6 ± 26,6 | 0,0 ± 0,0 | 129,4 ± 26,6 | 70,6 ± 26,6 | 1 dari 15 |
| Grid | 4 koin (BTC, ETH, SOL, LINK) | 79,9 ± 19,0 | 0,0 ± 0,0 | 120,1 ± 19,0 | 79,9 ± 19,0 | 1 dari 12 |
| Manual | 5 koin | 72,5 ± 21,7 | 0,0 ± 0,0 | 127,5 ± 21,7 | 72,5 ± 21,7 | 0 dari 15 |
| Manual | 4 koin (BTC, ETH, SOL, LINK) | 80,1 ± 15,7 | 0,0 ± 0,0 | 119,9 ± 15,7 | 80,1 ± 15,7 | 0 dari 12 |

Run tanpa sinyal sama sekali: 3 dari 45 (BTC-USD Grid s42, SOL-USD GA s44, SHIB-USD GA s43).

**Catatan penilaian SHORT.** `main.run_backtest` menilai LONG tepat bila label = 1 (harga naik ≥ 0,3% dalam 4 candle), tetapi SHORT tepat bila label = 0, yaitu cukup bila harga *tidak* naik ≥ 0,3%. Karena 70–90% candle berlabel 0, kriteria SHORT jauh lebih longgar, sehingga hit-rate run yang banyak menerbitkan SHORT tidak sebanding dengan run yang hanya LONG. Run dengan SHORT: LINK-USD GA s43 (7 SHORT), SHIB-USD GA s44 (101 SHORT).

## c. Metrik per fold walk-forward (mean ± std, 3 seed)

Jumlah sekuens dan % label 1 bergantung pada `sequence_length` konfigurasi, bukan pada seed (diperiksa identik antar seed).

| Koin | Metode | Fold | AUC | F1 | Precision | Recall | n latih | n validasi | % label 1 latih | % label 1 validasi |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | GA | 1 | 0,6321 ± 0,0453 | 0,1657 ± 0,0351 | 0,1046 ± 0,0266 | 0,4507 ± 0,1344 | 1149 | 1147 | 10,4 | 6,2 |
| BTC-USD | GA | 2 | 0,6255 ± 0,0035 | 0,1014 ± 0,1756 | 0,1294 ± 0,2241 | 0,0833 ± 0,1443 | 2385 | 1147 | 8,6 | 23,0 |
| BTC-USD | GA | 3 | 0,6512 ± 0,0027 | 0,2562 ± 0,0266 | 0,1489 ± 0,0209 | 0,9469 ± 0,0919 | 3621 | 1147 | 13,2 | 13,7 |
| BTC-USD | Grid | 1 | 0,6509 ± 0,0376 | 0,1361 ± 0,0241 | 0,1013 ± 0,0616 | 0,6250 ± 0,4072 | 1119 | 1117 | 10,3 | 5,7 |
| BTC-USD | Grid | 2 | 0,6169 ± 0,0305 | 0,1838 ± 0,1296 | 0,3684 ± 0,1140 | 0,1576 ± 0,1265 | 2355 | 1117 | 8,5 | 23,1 |
| BTC-USD | Grid | 3 | 0,6596 ± 0,0194 | 0,2192 ± 0,0377 | 0,1617 ± 0,0427 | 0,7168 ± 0,4906 | 3591 | 1117 | 13,2 | 13,7 |
| BTC-USD | Manual | 1 | 0,6494 ± 0,0720 | 0,1274 ± 0,0116 | 0,0683 ± 0,0069 | 0,9556 ± 0,0468 | 1179 | 1177 | 10,3 | 6,4 |
| BTC-USD | Manual | 2 | 0,5931 ± 0,0136 | 0,0050 ± 0,0086 | 0,1667 ± 0,2887 | 0,0025 ± 0,0044 | 2415 | 1177 | 8,5 | 22,5 |
| BTC-USD | Manual | 3 | 0,6307 ± 0,0156 | 0,1747 ± 0,1318 | 0,1512 ± 0,0138 | 0,6646 ± 0,5652 | 3651 | 1177 | 13,2 | 14,0 |
| ETH-USD | GA | 1 | 0,6213 ± 0,0357 | 0,2020 ± 0,0285 | 0,1213 ± 0,0229 | 0,6962 ± 0,2352 | 1149 | 1147 | 16,9 | 9,9 |
| ETH-USD | GA | 2 | 0,6181 ± 0,0070 | 0,2942 ± 0,2224 | 0,3236 ± 0,0910 | 0,6460 ± 0,5441 | 2385 | 1147 | 13,5 | 26,8 |
| ETH-USD | GA | 3 | 0,5853 ± 0,0131 | 0,3328 ± 0,0000 | 0,1997 ± 0,0000 | 1,0000 ± 0,0000 | 3621 | 1147 | 17,6 | 20,0 |
| ETH-USD | Grid | 1 | 0,5958 ± 0,0530 | 0,2097 ± 0,0184 | 0,1206 ± 0,0132 | 0,8251 ± 0,1217 | 1194 | 1192 | 16,7 | 10,2 |
| ETH-USD | Grid | 2 | 0,6168 ± 0,0048 | 0,3608 ± 0,0982 | 0,3279 ± 0,0086 | 0,4452 ± 0,2236 | 2430 | 1192 | 13,4 | 26,3 |
| ETH-USD | Grid | 3 | 0,5712 ± 0,0315 | 0,3329 ± 0,0000 | 0,1997 ± 0,0000 | 1,0000 ± 0,0000 | 3666 | 1192 | 17,5 | 20,0 |
| ETH-USD | Manual | 1 | 0,5865 ± 0,0331 | 0,1894 ± 0,0226 | 0,1073 ± 0,0170 | 0,8616 ± 0,1493 | 1179 | 1177 | 16,9 | 10,0 |
| ETH-USD | Manual | 2 | 0,6175 ± 0,0070 | 0,1795 ± 0,1986 | 0,4150 ± 0,0834 | 0,1989 ± 0,2833 | 2415 | 1177 | 13,5 | 26,3 |
| ETH-USD | Manual | 3 | 0,5889 ± 0,0268 | 0,2243 ± 0,1942 | 0,1348 ± 0,1167 | 0,6667 ± 0,5774 | 3651 | 1177 | 17,6 | 20,2 |
| SOL-USD | GA | 1 | 0,5812 ± 0,0516 | 0,2615 ± 0,0113 | 0,1525 ± 0,0082 | 0,9206 ± 0,0690 | 1119 | 1117 | 17,0 | 15,0 |
| SOL-USD | GA | 2 | 0,5714 ± 0,0267 | 0,4812 ± 0,0301 | 0,3535 ± 0,0209 | 0,7889 ± 0,2106 | 2355 | 1117 | 16,3 | 32,9 |
| SOL-USD | GA | 3 | 0,5581 ± 0,0193 | 0,2866 ± 0,1879 | 0,2536 ± 0,0128 | 0,6800 ± 0,5543 | 3591 | 1117 | 21,6 | 24,6 |
| SOL-USD | Grid | 1 | 0,5790 ± 0,0405 | 0,1770 ± 0,1434 | 0,1512 ± 0,0088 | 0,5556 ± 0,5053 | 1119 | 1117 | 17,0 | 15,0 |
| SOL-USD | Grid | 2 | 0,5874 ± 0,0213 | 0,5127 ± 0,0020 | 0,3567 ± 0,0040 | 0,9121 ± 0,0381 | 2355 | 1117 | 16,3 | 32,9 |
| SOL-USD | Grid | 3 | 0,5549 ± 0,0330 | 0,2634 ± 0,2281 | 0,1641 ± 0,1421 | 0,6667 ± 0,5774 | 3591 | 1117 | 21,6 | 24,6 |
| SOL-USD | Manual | 1 | 0,5685 ± 0,0114 | 0,2139 ± 0,0276 | 0,2121 ± 0,0299 | 0,2365 ± 0,0848 | 1179 | 1177 | 16,7 | 15,2 |
| SOL-USD | Manual | 2 | 0,5700 ± 0,0257 | 0,4520 ± 0,0892 | 0,3775 ± 0,0246 | 0,6459 ± 0,2939 | 2415 | 1177 | 16,1 | 32,8 |
| SOL-USD | Manual | 3 | 0,5579 ± 0,0061 | 0,2643 ± 0,2289 | 0,1648 ± 0,1427 | 0,6667 ± 0,5774 | 3651 | 1177 | 21,5 | 24,7 |
| LINK-USD | GA | 1 | 0,5516 ± 0,0049 | 0,3533 ± 0,0716 | 0,2743 ± 0,0073 | 0,5761 ± 0,2816 | 1209 | 1207 | 19,8 | 24,7 |
| LINK-USD | GA | 2 | 0,5704 ± 0,0175 | 0,3656 ± 0,2163 | 0,3548 ± 0,0417 | 0,6558 ± 0,5109 | 2445 | 1207 | 22,2 | 33,1 |
| LINK-USD | GA | 3 | 0,5674 ± 0,0163 | 0,3845 ± 0,1212 | 0,3090 ± 0,0228 | 0,7167 ± 0,4544 | 3681 | 1207 | 25,8 | 29,2 |
| LINK-USD | Grid | 1 | 0,5281 ± 0,0022 | 0,3851 ± 0,0064 | 0,2517 ± 0,0122 | 0,8443 ± 0,1545 | 1179 | 1177 | 19,8 | 24,6 |
| LINK-USD | Grid | 2 | 0,5774 ± 0,0147 | 0,4017 ± 0,1057 | 0,4058 ± 0,0694 | 0,5425 ± 0,4098 | 2415 | 1177 | 22,2 | 33,3 |
| LINK-USD | Grid | 3 | 0,5850 ± 0,0079 | 0,4585 ± 0,0019 | 0,2980 ± 0,0025 | 0,9943 ± 0,0099 | 3651 | 1177 | 25,8 | 29,7 |
| LINK-USD | Manual | 1 | 0,5664 ± 0,0244 | 0,2850 ± 0,1092 | 0,2901 ± 0,0553 | 0,2953 ± 0,1512 | 1179 | 1177 | 19,8 | 24,6 |
| LINK-USD | Manual | 2 | 0,5759 ± 0,0163 | 0,2125 ± 0,0334 | 0,4552 ± 0,0844 | 0,1412 ± 0,0308 | 2415 | 1177 | 22,2 | 33,3 |
| LINK-USD | Manual | 3 | 0,5978 ± 0,0060 | 0,4676 ± 0,0137 | 0,3135 ± 0,0253 | 0,9370 ± 0,1018 | 3651 | 1177 | 25,8 | 29,7 |
| SHIB-USD | GA | 1 | 0,5307 ± 0,0277 | 0,1068 ± 0,1688 | 0,1715 ± 0,1669 | 0,3023 ± 0,5151 | 1119 | 1117 | 31,8 | 18,3 |
| SHIB-USD | GA | 2 | 0,5376 ± 0,0130 | 0,1306 ± 0,2261 | 0,1346 ± 0,2331 | 0,1268 ± 0,2196 | 2355 | 1117 | 24,6 | 35,5 |
| SHIB-USD | GA | 3 | 0,5602 ± 0,0172 | 0,3238 ± 0,2516 | 0,3095 ± 0,0055 | 0,6725 ± 0,5672 | 3591 | 1117 | 27,8 | 30,5 |
| SHIB-USD | Grid | 1 | 0,5487 ± 0,0122 | 0,2188 ± 0,1470 | 0,2040 ± 0,0097 | 0,4512 ± 0,3867 | 1179 | 1177 | 30,8 | 18,3 |
| SHIB-USD | Grid | 2 | 0,5174 ± 0,0142 | 0,1305 ± 0,2260 | 0,1203 ± 0,2083 | 0,1426 ± 0,2470 | 2415 | 1177 | 24,3 | 34,7 |
| SHIB-USD | Grid | 3 | 0,5525 ± 0,0158 | 0,4625 ± 0,0148 | 0,3040 ± 0,0070 | 0,9667 ± 0,0577 | 3651 | 1177 | 27,5 | 30,6 |
| SHIB-USD | Manual | 1 | 0,5523 ± 0,0244 | 0,1327 ± 0,1557 | 0,1499 ± 0,0359 | 0,3395 ± 0,5399 | 1179 | 1177 | 30,8 | 18,3 |
| SHIB-USD | Manual | 2 | 0,5383 ± 0,0223 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 2415 | 1177 | 24,3 | 34,7 |
| SHIB-USD | Manual | 3 | 0,5687 ± 0,0059 | 0,4719 ± 0,0032 | 0,3110 ± 0,0054 | 0,9787 ± 0,0278 | 3651 | 1177 | 27,5 | 30,6 |

Rerata AUC seluruh 45 run per fold: fold 1 0,5828 ± 0,0495; fold 2 0,5822 ± 0,0357; fold 3 0,5860 ± 0,0372.

## d. Efisiensi pencarian

Waktu = `search_duration_min` (pencarian saja, tanpa evaluasi akhir). Manual = satu evaluasi konfigurasi baseline.

| Koin | GA: evaluasi | GA: menit | Grid: evaluasi | Grid: menit | Manual: evaluasi | Manual: menit |
|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 46 | 59,9 | 50 | 65,0 | 1 | 1,4 |
| ETH-USD | 49 | 69,6 | 50 | 73,6 | 1 | 2,9 |
| SOL-USD | 45 | 108,0 | 50 | 75,7 | 1 | 3,1 |
| LINK-USD | 47 | 50,1 | 50 | 89,2 | 1 | 1,8 |
| SHIB-USD | 46 | 71,9 | 50 | 106,8 | 1 | 3,2 |
| **Total** | **233** | **359,6** | **250** | **410,3** | **5** | **12,3** |
| Menit per evaluasi |  | 1,54 |  | 1,64 |  | 2,47 |

GA memakai 233 evaluasi (93,2% dari Grid) dan 359,6 menit (87,6% dari Grid). Evaluasi GA yang terealisasi < 60 (pop 10 × 6 generasi turunan) karena individu berulang diambil dari cache fitness. Waktu GA BTC adalah run ulang 7 Okt (lihat `laporan_3a.md`, catatan waktu).

## e. Fitness terbaik sejauh ini (AUC pencarian) pada evaluasi ke-k

GA: maksimum fitness individu non-cache dengan `eval_count_kumulatif` ≤ k (`ga_trace`). Grid: `best_so_far` (`grid_history`). Bila evaluasi GA < k, dipakai nilai evaluasi terakhirnya (ditandai \*).

| Koin | Metode | k = 10 | k = 20 | k = 30 | k = 40 | k = 50 |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | GA | 0,6316 | 0,6701 | 0,6770 | 0,6770 | 0,6770\* |
| BTC-USD | Grid | 0,6447 | 0,6447 | 0,6536 | 0,6536 | 0,6585 |
| BTC-USD | GA − Grid | -0,0131 | 0,0254 | 0,0235 | 0,0235 | 0,0186 |
| ETH-USD | GA | 0,6283 | 0,6415 | 0,6415 | 0,6415 | 0,6650\* |
| ETH-USD | Grid | 0,6365 | 0,6640 | 0,6640 | 0,6640 | 0,6640 |
| ETH-USD | GA − Grid | -0,0081 | -0,0224 | -0,0224 | -0,0224 | 0,0010 |
| SOL-USD | GA | 0,5806 | 0,5806 | 0,5806 | 0,5806 | 0,5806\* |
| SOL-USD | Grid | 0,5762 | 0,5762 | 0,5762 | 0,5762 | 0,5918 |
| SOL-USD | GA − Grid | 0,0044 | 0,0044 | 0,0044 | 0,0044 | -0,0111 |
| LINK-USD | GA | 0,5791 | 0,5791 | 0,5804 | 0,5810 | 0,5810\* |
| LINK-USD | Grid | 0,5533 | 0,5639 | 0,5824 | 0,5824 | 0,5824 |
| LINK-USD | GA − Grid | 0,0258 | 0,0152 | -0,0020 | -0,0014 | -0,0014 |
| SHIB-USD | GA | 0,5823 | 0,5904 | 0,5946 | 0,5981 | 0,6183\* |
| SHIB-USD | Grid | 0,5964 | 0,6008 | 0,6154 | 0,6154 | 0,6154 |
| SHIB-USD | GA − Grid | -0,0141 | -0,0104 | -0,0207 | -0,0173 | 0,0030 |

## f. Fitness pencarian vs AUC walk-forward

`val_auc_search` = AUC konfigurasi terpilih pada satu split kronologis 70/15/15 (`search_epochs` 20). wf_auc = rerata 3 fold walk-forward (100 epoch), mean ± std 3 seed. Selisih = wf_auc (mean) − val_auc_search.

| Koin | Metode | Konfigurasi | val_auc_search | wf_auc | Selisih |
|---|---|---|---:|---:|---:|
| BTC-USD | GA | seq 90, u 64/16, drop 0.1, lr 0.01, bs 32 | 0,6770 | 0,6363 ± 0,0137 | -0,0408 |
| BTC-USD | Grid | seq 120, u 128/32, drop 0.3, lr 0.001, bs 16 | 0,6585 | 0,6424 ± 0,0145 | -0,0160 |
| BTC-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,6052 | 0,6244 ± 0,0276 | 0,0191 |
| ETH-USD | GA | seq 90, u 64/32, drop 0.4, lr 0.005, bs 32 | 0,6650 | 0,6082 ± 0,0100 | -0,0568 |
| ETH-USD | Grid | seq 45, u 32/32, drop 0.4, lr 0.01, bs 64 | 0,6640 | 0,5946 ± 0,0072 | -0,0693 |
| ETH-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5535 | 0,5976 ± 0,0194 | 0,0441 |
| SOL-USD | GA | seq 120, u 64/48, drop 0.2, lr 0.01, bs 64 | 0,5806 | 0,5702 ± 0,0261 | -0,0104 |
| SOL-USD | Grid | seq 120, u 32/48, drop 0.2, lr 0.01, bs 64 | 0,5918 | 0,5738 ± 0,0103 | -0,0179 |
| SOL-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5486 | 0,5654 ± 0,0085 | 0,0169 |
| LINK-USD | GA | seq 30, u 32/32, drop 0.1, lr 0.0005, bs 64 | 0,5810 | 0,5631 ± 0,0092 | -0,0179 |
| LINK-USD | Grid | seq 60, u 96/16, drop 0.2, lr 0.0005, bs 32 | 0,5824 | 0,5635 ± 0,0065 | -0,0189 |
| LINK-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5379 | 0,5800 ± 0,0126 | 0,0421 |
| SHIB-USD | GA | seq 120, u 128/64, drop 0.1, lr 0.01, bs 32 | 0,6183 | 0,5428 ± 0,0147 | -0,0755 |
| SHIB-USD | Grid | seq 60, u 128/48, drop 0.4, lr 0.001, bs 64 | 0,6154 | 0,5395 ± 0,0055 | -0,0758 |
| SHIB-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5860 | 0,5531 ± 0,0168 | -0,0329 |

Rerata selisih per metode: GA -0,0403; Grid -0,0396; Manual 0,0178.

## g. Peringkat AUC walk-forward per koin

Peringkat berdasarkan mean wf_auc 3 seed (1 = tertinggi). Rentang = mean tertinggi − mean terendah di koin itu.

| Koin | GA | Grid | Manual | Terbaik | Rentang |
|---|---:|---:|---:|---:|---:|
| BTC-USD | 0,6363 (2) | 0,6424 (1) | 0,6244 (3) | Grid | 0,0181 |
| ETH-USD | 0,6082 (1) | 0,5946 (3) | 0,5976 (2) | GA | 0,0136 |
| SOL-USD | 0,5702 (2) | 0,5738 (1) | 0,5654 (3) | Grid | 0,0084 |
| LINK-USD | 0,5631 (3) | 0,5635 (2) | 0,5800 (1) | Manual | 0,0169 |
| SHIB-USD | 0,5428 (2) | 0,5395 (3) | 0,5531 (1) | Manual | 0,0136 |

| Metode | Menang (dari 5 koin) | Peringkat rata-rata |
|---|---:|---:|
| GA | 1 | 2,00 |
| Grid | 2 | 2,00 |
| Manual | 2 | 2,00 |

Rentang selisih antarmetode per koin: 0,0084 (SOL-USD) sampai 0,0181 (BTC-USD).

## h. Selisih AUC antarmetode vs variasi antar-seed

Selisih = mean wf_auc metode pertama − metode kedua. Std = std antar 3 seed masing-masing. **Tumpang tindih** = interval mean ± std kedua metode beririsan; rasio = |selisih| / std terbesar dari keduanya.

| Koin | Pasangan | Selisih | Std pertama | Std kedua | Rasio | Tumpang tindih |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | GA − Grid | -0,0062 | 0,0137 | 0,0145 | 0,43 | **ya** |
| BTC-USD | GA − Manual | 0,0119 | 0,0137 | 0,0276 | 0,43 | **ya** |
| BTC-USD | Grid − Manual | 0,0181 | 0,0145 | 0,0276 | 0,66 | **ya** |
| ETH-USD | GA − Grid | 0,0136 | 0,0100 | 0,0072 | 1,35 | **ya** |
| ETH-USD | GA − Manual | 0,0106 | 0,0100 | 0,0194 | 0,55 | **ya** |
| ETH-USD | Grid − Manual | -0,0030 | 0,0072 | 0,0194 | 0,15 | **ya** |
| SOL-USD | GA − Grid | -0,0036 | 0,0261 | 0,0103 | 0,14 | **ya** |
| SOL-USD | GA − Manual | 0,0048 | 0,0261 | 0,0085 | 0,18 | **ya** |
| SOL-USD | Grid − Manual | 0,0084 | 0,0103 | 0,0085 | 0,81 | **ya** |
| LINK-USD | GA − Grid | -0,0004 | 0,0092 | 0,0065 | 0,04 | **ya** |
| LINK-USD | GA − Manual | -0,0169 | 0,0092 | 0,0126 | 1,34 | **ya** |
| LINK-USD | Grid − Manual | -0,0165 | 0,0065 | 0,0126 | 1,31 | **ya** |
| SHIB-USD | GA − Grid | 0,0033 | 0,0147 | 0,0055 | 0,23 | **ya** |
| SHIB-USD | GA − Manual | -0,0103 | 0,0147 | 0,0168 | 0,61 | **ya** |
| SHIB-USD | Grid − Manual | -0,0136 | 0,0055 | 0,0168 | 0,81 | **ya** |

15 dari 15 pasangan tumpang tindih.

## i. Pemeriksaan tambahan

### i.1 Waktu pencarian, 2 desimal

`search_duration_min` apa adanya (2 desimal, seperti di CSV); total = jumlah nilai yang sama. Tabel d, `laporan_3a.md`, dan `laporan_3b.md` menampilkan nilai ini dibulatkan ke 1 desimal (setengah ke atas), sehingga jumlah nilai 1 desimal bisa berbeda 0,1 dari total (mis. Manual: 1,4 + 2,9 + 3,1 + 1,8 + 3,2 = 12,4, sedangkan totalnya 12,34 ≈ 12,3).

| Koin | GA (menit) | Grid (menit) | Manual (menit) |
|---|---:|---:|---:|
| BTC-USD | 59,86 | 65,01 | 1,37 |
| ETH-USD | 69,64 | 73,62 | 2,91 |
| SOL-USD | 108,03 | 75,72 | 3,13 |
| LINK-USD | 50,12 | 89,17 | 1,78 |
| SHIB-USD | 71,90 | 106,79 | 3,15 |
| **Total** | **359,55** | **410,31** | **12,34** |

### i.2 Fold yang prediksinya runtuh ke satu kelas (ambang 0,5)

Dari `optimization_folds.csv` (135 fold = 45 run × 3 fold). *Semua positif*: recall = 1 dan precision = proporsi label 1 di fold validasi (toleransi 1e-4, karena % label 1 disimpan 4 desimal). *Semua negatif*: recall = 0 dan precision = 0. Keduanya dicek silang dengan accuracy: semua positif ⇒ accuracy = proporsi label 1; semua negatif ⇒ accuracy = 1 − proporsi label 1.

| Koin | GA | Grid | Manual | Semua metode |
|---|---:|---:|---:|---:|
| BTC-USD | 2 + 2 = 4 dari 9 | 2 + 0 = 2 dari 9 | 1 + 2 = 3 dari 9 | 5 + 4 = 9 dari 27 |
| ETH-USD | 4 + 0 = 4 dari 9 | 3 + 0 = 3 dari 9 | 3 + 1 = 4 dari 9 | 10 + 1 = 11 dari 27 |
| SOL-USD | 4 + 0 = 4 dari 9 | 3 + 1 = 4 dari 9 | 2 + 1 = 3 dari 9 | 9 + 2 = 11 dari 27 |
| LINK-USD | 1 + 0 = 1 dari 9 | 3 + 0 = 3 dari 9 | 1 + 0 = 1 dari 9 | 5 + 0 = 5 dari 27 |
| SHIB-USD | 1 + 3 = 4 dari 9 | 0 + 2 = 2 dari 9 | 1 + 3 = 4 dari 9 | 2 + 8 = 10 dari 27 |
| **Total** | 12 + 5 = 17 dari 45 | 11 + 3 = 14 dari 45 | 8 + 7 = 15 dari 45 | 31 + 15 = 46 dari 135 |

Format sel: semua positif + semua negatif = runtuh dari jumlah fold.

Per nomor fold: fold 1 3 positif / 1 negatif; fold 2 3 positif / 11 negatif; fold 3 25 positif / 3 negatif.
Hampir runtuh (recall = 1, tetapi ada sedikit prediksi negatif): BTC-USD Manual s42 fold 1 (precision 0,0644 vs proporsi 0,0637); SHIB-USD GA s42 fold 3 (precision 0,3075 vs proporsi 0,3053); SHIB-USD Grid s42 fold 3 (precision 0,3082 vs proporsi 0,3059); SHIB-USD Grid s44 fold 3 (precision 0,3080 vs proporsi 0,3059).
AUC fold yang runtuh tetap terdefinisi, karena dihitung dari peringkat probabilitas, bukan dari ambang. Runtuh inilah yang membuat F1/precision/recall bernilai 0 atau recall = 1 pada tabel c.

### i.3 Ketidakdeterministikan evaluasi akhir antarproses

**Masalah.** BTC-USD Manual seed 42 (konfigurasi, data snapshot, dan jalur kode sama) memberi wf_auc 0,648133 dan 13 sinyal pada uji determinisme Langkah 2b, tetapi 0,664560 dan 142 sinyal pada reeval Langkah 4 (9 Okt, run ke-7 dalam satu proses).

**Uji 1 — urutan run** (`bahan_bab4/uji_isolasi_urutan.md`). Urutan Langkah 4 untuk BTC diulang dalam satu proses:
- GA s42–s44 (posisi 1–3) identik dengan Langkah 4; Grid s42–s44 (posisi 4–6) tidak; Manual s42 di posisi 7 = 0,648133, identik dengan proses terpisah (0,648133; AUC per fold, sinyal, dan 200 probabilitas sama). Urutan run tidak mengubah hasil Manual s42.
- Fitness pencarian (Manual, GA terbaik, Grid terbaik) identik dengan `val_auc_search`, di proses baru maupun setelah 7 evaluasi akhir: manual 0,605245, ga 0,677047, grid 0,658459. **Tahap pencarian tidak terdampak.**
- Isolasi subprocess tidak cukup: Manual s42 di dua subprocess baru memberi 0,648133 (13/187) dan 0,664560 (142/58); yang kedua sama persis dengan angka Langkah 4.

**Uji 2 — penyebab** (`bahan_bab4/uji_penyebab.md`). Setiap run = evaluasi akhir lengkap di proses Python baru; kondisi dipasang sebelum TensorFlow diimpor; putaran bergiliran. CPU: Intel Core i5-12450H (4 P-core + 4 E-core).

| Run | Kondisi | Run | Hasil (wf_auc, terbit/HOLD) × kemunculan | Identik semua |
|---|---|---:|---:|---:|
| Manual s42 | K1: bawaan (oneDNN aktif) | 8 | 0,648133 (13/187) × 7; 0,664560 (142/58) × 1 | tidak |
| Manual s42 | K2: affinity P-core, oneDNN aktif | 8 | 0,648133 (13/187) × 8 | **ya** |
| Manual s42 | K3: oneDNN mati + `enable_op_determinism()` | 8 | 0,653397 (142/58) × 7; 0,628646 (11/189) × 1 | tidak |
| Grid s42 | K2: affinity P-core, oneDNN aktif | 4 | 0,630755 (0/200) × 4 | **ya** |

Rekap kemunculan Manual s42 kondisi K1 (bawaan) di semua uji: uji determinisme 2×, cek 10 Okt 1×, uji urutan 2× (proses terpisah, posisi 7), dan isolasi 1× memberi 0,648133; Langkah 4 1× dan isolasi 1× memberi 0,664560; uji penyebab K1: 0,648133 × 7; 0,664560 × 1.

**Kesimpulan dan perbaikan.** Penyebabnya adalah ketidakdeterministikan numerik antarproses yang tidak dikendalikan seed, bukan urutan run. Pola hasilnya konsisten dengan pemilihan jalur komputasi yang berbeda antara P-core dan E-core pada CPU hibrida (dugaan; mekanisme internalnya tidak diuji langsung). Mematikan oneDNN (K3) tidak menghilangkannya. Kondisi **K2** (affinity P-core, oneDNN aktif) memberi hasil identik pada semua run Manual dan Grid, sehingga dipakai untuk evaluasi akhir final: `reeval_final.py --kondisi K2`, setiap run di subprocess baru. Tahap pencarian tidak diulang: pencarian berjalan dengan kondisi bawaan, dan fitness terbukti stabil. 45 baris, 135 fold, dan 45 backtest reeval lama diarsipkan di `logs/archive_reeval_tanpa_isolasi/`.

**Batas bukti.** Pada K1 hasil menyimpang muncul 1 dari 8 run (secara keseluruhan 2 dari 12 run Manual s42 yang masing-masing dimulai di proses baru: cek 10 Okt, proses terpisah, 2 isolasi, 8 uji penyebab). Bila K2 sebenarnya tidak lebih baik dari K1, peluang mendapat 8/8 identik secara kebetulan kira-kira (7/8)^8 ≈ 0,34. Jadi 8/8 + 4/4 adalah bukti pendukung, belum jaminan. Karena itu reeval final juga diverifikasi dengan menjalankan ulang dua kombinasi acak di proses terpisah (di bawah), dan BTC-USD Manual s42 dibandingkan dengan acuan K2.

**Verifikasi reeval final.** Dua kombinasi acak dijalankan ulang di proses terpisah dengan kondisi yang sama (`bahan_bab4/uji_ulang_reeval.md`): identik. BTC-USD Manual s42 pada reeval final: 0,648133.
