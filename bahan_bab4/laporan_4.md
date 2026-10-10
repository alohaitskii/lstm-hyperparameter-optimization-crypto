# Laporan Langkah 4 — Evaluasi akhir 3 seed di atas snapshot

Data: `data/snapshot_20260912/` (cache-only, diverifikasi terhadap MANIFEST).
Setiap konfigurasi terpilih (GA, Grid, Manual; 5 koin) dievaluasi ulang dengan
`optimization/reeval_final.py --seeds 42 43 44`: walk-forward 3 fold
(`TimeSeriesSplit`, maks. 100 epoch, patience 10) + backtest 200 candle.
Run dari terminal VS Code, 9 Okt 17:45–22:16 WIB (271,4 menit, kode keluar 0).

Semua std = simpangan baku sampel (ddof = 1) antar 3 seed. Dihasilkan oleh
`tools/buat_laporan_4.py`.

## Verifikasi

- `optimization_results.csv`: 15 baris pencarian + **45 baris reeval** (5 koin × 3 metode × seed 42/43/44), tanpa duplikat.
- `val_auc_search`, `search_duration_min`, `n_evals`, `best_hp_json`, `search_epochs` pada 45 baris = baris pencarian snapshot (GA BTC: 59,86 menit, run ulang 7 Okt).
- `optimization_folds.csv`: 135 baris (45 run × 3 fold), tanpa duplikat.
- Jumlah LONG+SHORT dan HOLD di 45 berkas backtest = kolom `n_signals_*` di CSV hasil.
- `tools/cek_jendela_data.py --semua-harus-cocok`: **60/60** backtest (15 pencarian + 45 reeval) tepat pada snapshot.

**Catatan seed 42.** Metrik reeval seed 42 tidak sama dengan metrik evaluasi akhir pada baris pencarian (selisih |wf_auc| maks. 0,0389, median 0,0103). Penyebabnya, evaluasi akhir di `run_optimization.py` berjalan langsung sesudah pencarian tanpa reset seed, sehingga keadaan RNG-nya bergantung pada jalannya pencarian. `reeval_final.py` memanggil `set_global_seed(seed)` tepat sebelum setiap evaluasi, sehingga terdeterminasi oleh seed saja. Angka yang dilaporkan adalah 45 baris reeval.

## a. Metrik walk-forward dan backtest (mean ± std, 3 seed)

AUC, F1, precision, recall = rerata 3 fold walk-forward. Hit-rate = proporsi sinyal LONG/SHORT yang benar pada backtest 200 candle. Seed tanpa sinyal tidak punya hit-rate (tercatat NaN di CSV, bukan 0) dan tidak ikut dihitung; n = jumlah seed yang menerbitkan sinyal. Std dengan n = 1 tidak terdefinisi (–).

| Koin | Metode | AUC | F1 | Precision | Recall | Hit-rate |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | GA | 0,6363 ± 0,0137 | 0,1744 ± 0,0547 | 0,1276 ± 0,0748 | 0,4937 ± 0,0344 | 0,1233 ± 0,0034 (n=3) |
| BTC-USD | Grid | 0,6478 ± 0,0102 | 0,2117 ± 0,0557 | 0,2116 ± 0,0135 | 0,5339 ± 0,1267 | 0,1363 ± 0,0135 (n=3) |
| BTC-USD | Manual | 0,6430 ± 0,0192 | 0,1856 ± 0,0709 | 0,1988 ± 0,0463 | 0,6467 ± 0,0468 | 0,1266 ± 0,0003 (n=3) |
| ETH-USD | GA | 0,6082 ± 0,0100 | 0,2763 ± 0,0666 | 0,2148 ± 0,0354 | 0,7807 ± 0,1975 | 0,2434 ± 0,0367 (n=3) |
| ETH-USD | Grid | 0,5946 ± 0,0072 | 0,3011 ± 0,0332 | 0,2160 ± 0,0026 | 0,7568 ± 0,1136 | 0,2628 ± 0,0772 (n=3) |
| ETH-USD | Manual | 0,6009 ± 0,0133 | 0,1932 ± 0,1575 | 0,1899 ± 0,0468 | 0,4768 ± 0,4142 | 0,2222 ± 0,0000 (n=3) |
| SOL-USD | GA | 0,5754 ± 0,0131 | 0,2881 ± 0,0728 | 0,2880 ± 0,0484 | 0,5733 ± 0,1668 | 0,2121 ± 0,0000 (n=3) |
| SOL-USD | Grid | 0,5765 ± 0,0182 | 0,2734 ± 0,0655 | 0,2267 ± 0,0494 | 0,5568 ± 0,0770 | 0,3382 ± 0,1783 (n=2) |
| SOL-USD | Manual | 0,5706 ± 0,0011 | 0,3330 ± 0,0413 | 0,2670 ± 0,0042 | 0,5676 ± 0,1552 | 0,2121 ± 0,0000 (n=3) |
| LINK-USD | GA | 0,5631 ± 0,0092 | 0,3678 ± 0,0464 | 0,3127 ± 0,0087 | 0,6495 ± 0,0809 | 0,2454 ± 0,0510 (n=3) |
| LINK-USD | Grid | 0,5592 ± 0,0104 | 0,4250 ± 0,0282 | 0,3140 ± 0,0052 | 0,7202 ± 0,1591 | 0,3176 ± 0,0471 (n=3) |
| LINK-USD | Manual | 0,5785 ± 0,0042 | 0,3363 ± 0,0588 | 0,3381 ± 0,0363 | 0,5466 ± 0,1323 | 0,3977 ± 0,1446 (n=2) |
| SHIB-USD | GA | 0,5526 ± 0,0068 | 0,1404 ± 0,1250 | 0,1366 ± 0,1183 | 0,2494 ± 0,2180 | tidak terdefinisi (n=0) |
| SHIB-USD | Grid | 0,5547 ± 0,0188 | 0,2208 ± 0,0439 | 0,1854 ± 0,0216 | 0,3792 ± 0,1422 | 0,4148 ± 0,0562 (n=2) |
| SHIB-USD | Manual | 0,5546 ± 0,0097 | 0,2099 ± 0,0787 | 0,1873 ± 0,0282 | 0,4636 ± 0,1404 | 0,3001 ± 0,0128 (n=3) |

**Rerata lintas koin** (rerata dari mean per koin, setiap koin berbobot sama; ± = std antar koin). Baris *5 koin*: semua metrik; hit-rate dari mean per koin yang terdefinisi (GA tanpa SHIB-USD, ketiga seed tanpa sinyal). Baris *4 koin*: hit-rate pada BTC, ETH, SOL, LINK, tempat ketiga metode bersinyal, sehingga ketiga metode dibandingkan pada koin yang sama (mean SOL-USD Grid dan LINK-USD Manual masing-masing dari 2 seed bersinyal).

| Metode | Cakupan | AUC | F1 | Precision | Recall | Hit-rate |
|---|---|---:|---:|---:|---:|---:|
| GA | 5 koin | 0,5871 ± 0,0345 | 0,2494 ± 0,0919 | 0,2160 ± 0,0846 | 0,5493 ± 0,1983 | 0,2061 ± 0,0572 (4 koin) |
| GA | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2061 ± 0,0572 (4 koin) |
| Grid | 5 koin | 0,5866 ± 0,0377 | 0,2864 ± 0,0859 | 0,2307 ± 0,0490 | 0,5894 ± 0,1529 | 0,2939 ± 0,1036 |
| Grid | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2637 ± 0,0907 (4 koin) |
| Manual | 5 koin | 0,5895 ± 0,0342 | 0,2516 ± 0,0763 | 0,2362 ± 0,0657 | 0,5402 ± 0,0742 | 0,2518 ± 0,1022 |
| Manual | 4 koin (BTC, ETH, SOL, LINK) |  |  |  |  | 0,2397 ± 0,1138 (4 koin) |

## b. Sinyal backtest LONG / SHORT / HOLD (200 candle)

| Koin | Metode | LONG (mean) | SHORT (mean) | HOLD (mean) | Per seed 42 · 43 · 44 (L/S/H) |
|---|---|---:|---:|---:|---:|
| BTC-USD | GA | 126,7 | 0,0 | 73,3 | 138/0/62 · 142/0/58 · 100/0/100 |
| BTC-USD | Grid | 130,7 | 0,0 | 69,3 | 112/0/88 · 142/0/58 · 138/0/62 |
| BTC-USD | Manual | 129,0 | 0,0 | 71,0 | 142/0/58 · 103/0/97 · 142/0/58 |
| ETH-USD | GA | 87,0 | 0,0 | 113,0 | 99/0/101 · 99/0/101 · 63/0/137 |
| ETH-USD | Grid | 83,7 | 0,0 | 116,3 | 54/0/146 · 98/0/102 · 99/0/101 |
| ETH-USD | Manual | 99,0 | 0,0 | 101,0 | 99/0/101 · 99/0/101 · 99/0/101 |
| SOL-USD | GA | 99,0 | 0,0 | 101,0 | 99/0/101 · 99/0/101 · 99/0/101 |
| SOL-USD | Grid | 42,7 | 9,0 | 148,3 | 99/0/101 · 0/0/200 · 29/27/144 |
| SOL-USD | Manual | 99,0 | 0,0 | 101,0 | 99/0/101 · 99/0/101 · 99/0/101 |
| LINK-USD | GA | 64,7 | 2,3 | 133,0 | 66/0/134 · 26/7/167 · 102/0/98 |
| LINK-USD | Grid | 38,0 | 0,0 | 162,0 | 31/0/169 · 68/0/132 · 15/0/185 |
| LINK-USD | Manual | 16,7 | 0,0 | 183,3 | 0/0/200 · 6/0/194 · 44/0/156 |
| SHIB-USD | GA | 0,0 | 0,0 | 200,0 | 0/0/200 · 0/0/200 · 0/0/200 |
| SHIB-USD | Grid | 30,7 | 2,7 | 166,7 | 0/0/200 · 56/0/144 · 36/8/156 |
| SHIB-USD | Manual | 48,0 | 0,0 | 152,0 | 58/0/142 · 63/0/137 · 23/0/177 |

Total 15 backtest per metode: GA 1132 LONG / 7 SHORT / 1861 HOLD; Grid 977 LONG / 35 SHORT / 1988 HOLD; Manual 1175 LONG / 0 SHORT / 1825 HOLD.

**Rerata per backtest per metode** (rerata dari mean per koin, setiap koin berbobot sama; ± = std antar koin):

| Metode | Cakupan | LONG | SHORT | HOLD | Terbit (LONG+SHORT) | Run tanpa sinyal |
|---|---|---:|---:|---:|---:|---:|
| GA | 5 koin | 75,5 ± 47,7 | 0,5 ± 1,0 | 124,1 ± 47,6 | 75,9 ± 47,6 | 3 dari 15 |
| GA | 4 koin (BTC, ETH, SOL, LINK) | 94,3 ± 25,8 | 0,6 ± 1,2 | 105,1 ± 24,9 | 94,9 ± 24,9 | 0 dari 12 |
| Grid | 5 koin | 65,1 ± 42,0 | 2,3 ± 3,9 | 132,5 ± 40,4 | 67,5 ± 40,4 | 2 dari 15 |
| Grid | 4 koin (BTC, ETH, SOL, LINK) | 73,8 ± 43,1 | 2,3 ± 4,5 | 124,0 ± 41,2 | 76,0 ± 41,2 | 1 dari 12 |
| Manual | 5 koin | 78,3 ± 45,1 | 0,0 ± 0,0 | 121,7 ± 45,1 | 78,3 ± 45,1 | 1 dari 15 |
| Manual | 4 koin (BTC, ETH, SOL, LINK) | 85,9 ± 48,3 | 0,0 ± 0,0 | 114,1 ± 48,3 | 85,9 ± 48,3 | 1 dari 12 |

Run tanpa sinyal sama sekali: 6 dari 45 (SOL-USD Grid s43, LINK-USD Manual s42, SHIB-USD GA s42, SHIB-USD GA s43, SHIB-USD GA s44, SHIB-USD Grid s42).

## c. Metrik per fold walk-forward (mean ± std, 3 seed)

Jumlah sekuens dan % label 1 bergantung pada `sequence_length` konfigurasi, bukan pada seed (diperiksa identik antar seed).

| Koin | Metode | Fold | AUC | F1 | Precision | Recall | n latih | n validasi | % label 1 latih | % label 1 validasi |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | GA | 1 | 0,6321 ± 0,0453 | 0,1657 ± 0,0351 | 0,1046 ± 0,0266 | 0,4507 ± 0,1344 | 1149 | 1147 | 10,4 | 6,2 |
| BTC-USD | GA | 2 | 0,6255 ± 0,0035 | 0,1014 ± 0,1756 | 0,1294 ± 0,2241 | 0,0833 ± 0,1443 | 2385 | 1147 | 8,6 | 23,0 |
| BTC-USD | GA | 3 | 0,6512 ± 0,0027 | 0,2562 ± 0,0266 | 0,1489 ± 0,0209 | 0,9469 ± 0,0919 | 3621 | 1147 | 13,2 | 13,7 |
| BTC-USD | Grid | 1 | 0,6591 ± 0,0097 | 0,1140 ± 0,0989 | 0,0918 ± 0,0867 | 0,2708 ± 0,3428 | 1119 | 1117 | 10,3 | 5,7 |
| BTC-USD | Grid | 2 | 0,6283 ± 0,0071 | 0,2800 ± 0,1199 | 0,4061 ± 0,1033 | 0,3307 ± 0,3323 | 2355 | 1117 | 8,5 | 23,1 |
| BTC-USD | Grid | 3 | 0,6561 ± 0,0159 | 0,2409 ± 0,0000 | 0,1370 ± 0,0000 | 1,0000 ± 0,0000 | 3591 | 1117 | 13,2 | 13,7 |
| BTC-USD | Manual | 1 | 0,6548 ± 0,0582 | 0,1350 ± 0,0121 | 0,0772 ± 0,0133 | 0,7689 ± 0,3774 | 1179 | 1177 | 10,3 | 6,4 |
| BTC-USD | Manual | 2 | 0,6355 ± 0,0242 | 0,1759 ± 0,2037 | 0,3791 ± 0,1332 | 0,1711 ± 0,2389 | 2415 | 1177 | 8,5 | 22,5 |
| BTC-USD | Manual | 3 | 0,6385 ± 0,0202 | 0,2459 ± 0,0000 | 0,1402 ± 0,0000 | 1,0000 ± 0,0000 | 3651 | 1177 | 13,2 | 14,0 |
| ETH-USD | GA | 1 | 0,6213 ± 0,0357 | 0,2020 ± 0,0285 | 0,1213 ± 0,0229 | 0,6962 ± 0,2352 | 1149 | 1147 | 16,9 | 9,9 |
| ETH-USD | GA | 2 | 0,6181 ± 0,0070 | 0,2942 ± 0,2224 | 0,3236 ± 0,0910 | 0,6460 ± 0,5441 | 2385 | 1147 | 13,5 | 26,8 |
| ETH-USD | GA | 3 | 0,5853 ± 0,0131 | 0,3328 ± 0,0000 | 0,1997 ± 0,0000 | 1,0000 ± 0,0000 | 3621 | 1147 | 17,6 | 20,0 |
| ETH-USD | Grid | 1 | 0,5958 ± 0,0530 | 0,2097 ± 0,0184 | 0,1206 ± 0,0132 | 0,8251 ± 0,1217 | 1194 | 1192 | 16,7 | 10,2 |
| ETH-USD | Grid | 2 | 0,6168 ± 0,0048 | 0,3608 ± 0,0982 | 0,3279 ± 0,0086 | 0,4452 ± 0,2236 | 2430 | 1192 | 13,4 | 26,3 |
| ETH-USD | Grid | 3 | 0,5712 ± 0,0315 | 0,3329 ± 0,0000 | 0,1997 ± 0,0000 | 1,0000 ± 0,0000 | 3666 | 1192 | 17,5 | 20,0 |
| ETH-USD | Manual | 1 | 0,6172 ± 0,0224 | 0,1599 ± 0,1246 | 0,1432 ± 0,0071 | 0,4294 ± 0,3875 | 1179 | 1177 | 16,9 | 10,0 |
| ETH-USD | Manual | 2 | 0,6225 ± 0,0044 | 0,1953 ± 0,1993 | 0,2917 ± 0,0363 | 0,3344 ± 0,4959 | 2415 | 1177 | 13,5 | 26,3 |
| ETH-USD | Manual | 3 | 0,5630 ± 0,0196 | 0,2243 ± 0,1942 | 0,1348 ± 0,1167 | 0,6667 ± 0,5774 | 3651 | 1177 | 17,6 | 20,2 |
| SOL-USD | GA | 1 | 0,6050 ± 0,0101 | 0,2033 ± 0,1486 | 0,2577 ± 0,1494 | 0,6052 ± 0,5128 | 1119 | 1117 | 17,0 | 15,0 |
| SOL-USD | GA | 2 | 0,5619 ± 0,0118 | 0,3097 ± 0,1628 | 0,3506 ± 0,0180 | 0,3632 ± 0,3488 | 2355 | 1117 | 16,3 | 32,9 |
| SOL-USD | GA | 3 | 0,5592 ± 0,0355 | 0,3515 ± 0,0756 | 0,2556 ± 0,0163 | 0,7515 ± 0,4304 | 3591 | 1117 | 21,6 | 24,6 |
| SOL-USD | Grid | 1 | 0,5935 ± 0,0331 | 0,2756 ± 0,0209 | 0,1700 ± 0,0314 | 0,8472 ± 0,2595 | 1119 | 1117 | 17,0 | 15,0 |
| SOL-USD | Grid | 2 | 0,5731 ± 0,0094 | 0,2908 ± 0,2430 | 0,3280 ± 0,0916 | 0,3333 ± 0,2855 | 2355 | 1117 | 16,3 | 32,9 |
| SOL-USD | Grid | 3 | 0,5628 ± 0,0204 | 0,2537 ± 0,2202 | 0,1821 ± 0,1600 | 0,4897 ± 0,5003 | 3591 | 1117 | 21,6 | 24,6 |
| SOL-USD | Manual | 1 | 0,5635 ± 0,0064 | 0,2322 ± 0,0088 | 0,1811 ± 0,0196 | 0,3426 ± 0,0870 | 1179 | 1177 | 16,7 | 15,2 |
| SOL-USD | Manual | 2 | 0,5819 ± 0,0128 | 0,4324 ± 0,1330 | 0,3761 ± 0,0190 | 0,6304 ± 0,3598 | 2415 | 1177 | 16,1 | 32,8 |
| SOL-USD | Manual | 3 | 0,5665 ± 0,0143 | 0,3344 ± 0,1075 | 0,2438 ± 0,0059 | 0,7297 ± 0,4682 | 3651 | 1177 | 21,5 | 24,7 |
| LINK-USD | GA | 1 | 0,5516 ± 0,0049 | 0,3533 ± 0,0716 | 0,2743 ± 0,0073 | 0,5761 ± 0,2816 | 1209 | 1207 | 19,8 | 24,7 |
| LINK-USD | GA | 2 | 0,5704 ± 0,0175 | 0,3656 ± 0,2163 | 0,3548 ± 0,0417 | 0,6558 ± 0,5109 | 2445 | 1207 | 22,2 | 33,1 |
| LINK-USD | GA | 3 | 0,5674 ± 0,0163 | 0,3845 ± 0,1212 | 0,3090 ± 0,0228 | 0,7167 ± 0,4544 | 3681 | 1207 | 25,8 | 29,2 |
| LINK-USD | Grid | 1 | 0,5356 ± 0,0115 | 0,3287 ± 0,0626 | 0,2522 ± 0,0036 | 0,5398 ± 0,3199 | 1179 | 1177 | 19,8 | 24,6 |
| LINK-USD | Grid | 2 | 0,5565 ± 0,0220 | 0,4997 ± 0,0000 | 0,3331 ± 0,0000 | 1,0000 ± 0,0000 | 2415 | 1177 | 22,2 | 33,3 |
| LINK-USD | Grid | 3 | 0,5856 ± 0,0033 | 0,4467 ± 0,0278 | 0,3568 ± 0,0190 | 0,6208 ± 0,1630 | 3651 | 1177 | 25,8 | 29,7 |
| LINK-USD | Manual | 1 | 0,5572 ± 0,0110 | 0,2670 ± 0,0371 | 0,2623 ± 0,0209 | 0,2734 ± 0,0540 | 1179 | 1177 | 19,8 | 24,6 |
| LINK-USD | Manual | 2 | 0,5817 ± 0,0134 | 0,2813 ± 0,2070 | 0,4482 ± 0,1044 | 0,4056 ± 0,5182 | 2415 | 1177 | 22,2 | 33,3 |
| LINK-USD | Manual | 3 | 0,5964 ± 0,0050 | 0,4607 ± 0,0058 | 0,3036 ± 0,0123 | 0,9608 ± 0,0678 | 3651 | 1177 | 25,8 | 29,7 |
| SHIB-USD | GA | 1 | 0,5658 ± 0,0123 | 0,0836 ± 0,1448 | 0,1031 ± 0,1786 | 0,0703 ± 0,1217 | 1119 | 1117 | 31,8 | 18,3 |
| SHIB-USD | GA | 2 | 0,5352 ± 0,0073 | 0,0264 ± 0,0457 | 0,1034 ± 0,1792 | 0,0151 ± 0,0262 | 2355 | 1117 | 24,6 | 35,5 |
| SHIB-USD | GA | 3 | 0,5567 ± 0,0087 | 0,3112 ± 0,2695 | 0,2033 ± 0,1761 | 0,6628 ± 0,5740 | 3591 | 1117 | 27,8 | 30,5 |
| SHIB-USD | Grid | 1 | 0,5586 ± 0,0278 | 0,2168 ± 0,1455 | 0,2184 ± 0,0256 | 0,3876 ± 0,3806 | 1179 | 1177 | 30,8 | 18,3 |
| SHIB-USD | Grid | 2 | 0,5349 ± 0,0270 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 2415 | 1177 | 24,3 | 34,7 |
| SHIB-USD | Grid | 3 | 0,5705 ± 0,0081 | 0,4455 ± 0,0350 | 0,3377 ± 0,0418 | 0,7500 ± 0,2929 | 3651 | 1177 | 27,5 | 30,6 |
| SHIB-USD | Manual | 1 | 0,5600 ± 0,0152 | 0,3003 ± 0,0290 | 0,1986 ± 0,0123 | 0,7194 ± 0,3122 | 1179 | 1177 | 30,8 | 18,3 |
| SHIB-USD | Manual | 2 | 0,5359 ± 0,0124 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 2415 | 1177 | 24,3 | 34,7 |
| SHIB-USD | Manual | 3 | 0,5679 ± 0,0032 | 0,3296 ± 0,2443 | 0,3634 ± 0,0955 | 0,6713 ± 0,5597 | 3651 | 1177 | 27,5 | 30,6 |

Rerata AUC seluruh 45 run per fold: fold 1 0,5914 ± 0,0452; fold 2 0,5852 ± 0,0376; fold 3 0,5866 ± 0,0362.

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
| BTC-USD | Grid | seq 120, u 128/32, drop 0.3, lr 0.001, bs 16 | 0,6585 | 0,6478 ± 0,0102 | -0,0106 |
| BTC-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,6052 | 0,6430 ± 0,0192 | 0,0377 |
| ETH-USD | GA | seq 90, u 64/32, drop 0.4, lr 0.005, bs 32 | 0,6650 | 0,6082 ± 0,0100 | -0,0568 |
| ETH-USD | Grid | seq 45, u 32/32, drop 0.4, lr 0.01, bs 64 | 0,6640 | 0,5946 ± 0,0072 | -0,0693 |
| ETH-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5535 | 0,6009 ± 0,0133 | 0,0474 |
| SOL-USD | GA | seq 120, u 64/48, drop 0.2, lr 0.01, bs 64 | 0,5806 | 0,5754 ± 0,0131 | -0,0052 |
| SOL-USD | Grid | seq 120, u 32/48, drop 0.2, lr 0.01, bs 64 | 0,5918 | 0,5765 ± 0,0182 | -0,0153 |
| SOL-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5486 | 0,5706 ± 0,0011 | 0,0221 |
| LINK-USD | GA | seq 30, u 32/32, drop 0.1, lr 0.0005, bs 64 | 0,5810 | 0,5631 ± 0,0092 | -0,0179 |
| LINK-USD | Grid | seq 60, u 96/16, drop 0.2, lr 0.0005, bs 32 | 0,5824 | 0,5592 ± 0,0104 | -0,0232 |
| LINK-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5379 | 0,5785 ± 0,0042 | 0,0405 |
| SHIB-USD | GA | seq 120, u 128/64, drop 0.1, lr 0.01, bs 32 | 0,6183 | 0,5526 ± 0,0068 | -0,0658 |
| SHIB-USD | Grid | seq 60, u 128/48, drop 0.4, lr 0.001, bs 64 | 0,6154 | 0,5547 ± 0,0188 | -0,0607 |
| SHIB-USD | Manual | seq 60, u 128/64, drop 0.2, lr 0.001, bs 32 | 0,5860 | 0,5546 ± 0,0097 | -0,0315 |

Rerata selisih per metode: GA -0,0373; Grid -0,0358; Manual 0,0232.

## g. Peringkat AUC walk-forward per koin

Peringkat berdasarkan mean wf_auc 3 seed (1 = tertinggi). Rentang = mean tertinggi − mean terendah di koin itu.

| Koin | GA | Grid | Manual | Terbaik | Rentang |
|---|---:|---:|---:|---:|---:|
| BTC-USD | 0,6363 (3) | 0,6478 (1) | 0,6430 (2) | Grid | 0,0116 |
| ETH-USD | 0,6082 (1) | 0,5946 (3) | 0,6009 (2) | GA | 0,0136 |
| SOL-USD | 0,5754 (2) | 0,5765 (1) | 0,5706 (3) | Grid | 0,0058 |
| LINK-USD | 0,5631 (2) | 0,5592 (3) | 0,5785 (1) | Manual | 0,0193 |
| SHIB-USD | 0,5526 (3) | 0,5547 (1) | 0,5546 (2) | Grid | 0,0021 |

| Metode | Menang (dari 5 koin) | Peringkat rata-rata |
|---|---:|---:|
| GA | 1 | 2,20 |
| Grid | 3 | 1,80 |
| Manual | 1 | 2,00 |

Rentang selisih antarmetode per koin: 0,0021 (SHIB-USD) sampai 0,0193 (LINK-USD).

## h. Selisih AUC antarmetode vs variasi antar-seed

Selisih = mean wf_auc metode pertama − metode kedua. Std = std antar 3 seed masing-masing. **Tumpang tindih** = interval mean ± std kedua metode beririsan; rasio = |selisih| / std terbesar dari keduanya.

| Koin | Pasangan | Selisih | Std pertama | Std kedua | Rasio | Tumpang tindih |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | GA − Grid | -0,0116 | 0,0137 | 0,0102 | 0,84 | **ya** |
| BTC-USD | GA − Manual | -0,0067 | 0,0137 | 0,0192 | 0,35 | **ya** |
| BTC-USD | Grid − Manual | 0,0049 | 0,0102 | 0,0192 | 0,25 | **ya** |
| ETH-USD | GA − Grid | 0,0136 | 0,0100 | 0,0072 | 1,35 | **ya** |
| ETH-USD | GA − Manual | 0,0073 | 0,0100 | 0,0133 | 0,55 | **ya** |
| ETH-USD | Grid − Manual | -0,0063 | 0,0072 | 0,0133 | 0,47 | **ya** |
| SOL-USD | GA − Grid | -0,0011 | 0,0131 | 0,0182 | 0,06 | **ya** |
| SOL-USD | GA − Manual | 0,0048 | 0,0131 | 0,0011 | 0,36 | **ya** |
| SOL-USD | Grid − Manual | 0,0058 | 0,0182 | 0,0011 | 0,32 | **ya** |
| LINK-USD | GA − Grid | 0,0039 | 0,0092 | 0,0104 | 0,38 | **ya** |
| LINK-USD | GA − Manual | -0,0153 | 0,0092 | 0,0042 | 1,66 | tidak |
| LINK-USD | Grid − Manual | -0,0193 | 0,0104 | 0,0042 | 1,86 | tidak |
| SHIB-USD | GA − Grid | -0,0021 | 0,0068 | 0,0188 | 0,11 | **ya** |
| SHIB-USD | GA − Manual | -0,0020 | 0,0068 | 0,0097 | 0,21 | **ya** |
| SHIB-USD | Grid − Manual | 0,0001 | 0,0188 | 0,0097 | 0,00 | **ya** |

13 dari 15 pasangan tumpang tindih.
