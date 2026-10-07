# GA BTC-USD — run pertama 3a (diganti run ulang)

Berkas GA BTC-USD dari run 3a pertama (ts `20261006_043011`, diluncurkan dari
aplikasi Claude, 6 Okt 11:30–12:27 WIB). Diarsipkan 7 Okt 2026 setelah GA BTC
dijalankan ulang dari terminal VS Code (ts `20261007_023237`, 7 Okt 09:32–10:35
WIB, kode keluar 0), sama seperti koin lain pada 3a dan seluruh Grid 3b.

| Berkas | Isi |
|---|---|
| `ga_history_BTC_USD_20261006_043011.csv` | 7 generasi |
| `ga_trace_BTC_USD_20261006_043011.csv` | 70 individu |
| `ga_tournament_BTC_USD_20261006_043011.csv` | 96 turnamen |
| `backtest_BTC_USD_ga_20261006_052742.csv` | 200 candle |
| `optimization_results_btc_ga_run1.csv` | baris hasil yang dikeluarkan dari `logs/optimization_results.csv` |

## Reproduksibilitas

Selain kolom waktu, run ulang **identik** dengan berkas di sini: 70 baris jejak
(kromosom, fitness, cache, induk, mask, mutasi), 96 turnamen, 7 baris
`ga_history`, 200 baris backtest, dan baris hasil (best fitness 0,677047,
46 evaluasi, best_hp, wf_auc 0,625875, wf_f1 0,124422, 11 sinyal / 189 HOLD).

## Waktu

| | Run pertama | Run ulang |
|---|---:|---:|
| Pencarian (menit) | 53,52 | 59,86 |
| Total (menit) | 57,55 | 63,22 |

Rasio waktu per evaluasi (run ulang : run pertama) untuk 46 evaluasi yang sama:
median 0,89. Evaluasi 1–11 dan 41–44 run ulang ±1,3–2,3× lebih lambat, karena
laptop sedang dipakai. Evaluasi lainnya 0,72–0,99×.

Run ulang yang dipakai di skripsi karena cara peluncurannya sama dengan koin
lain. Waktunya mencerminkan kondisi pemakaian laptop yang sebenarnya.
Evaluasi tidak diulang lagi (keputusan peneliti, 7 Okt 2026).
