# Bahan E6 — contoh agregasi dan contoh sinyal

Dihasilkan oleh `tools/buat_e6_contoh.py` pada 2026-10-17 13:47 UTC dari 45 baris reeval di `logs/optimization_results.csv`, `logs/optimization_folds.csv`, dan berkas backtest reeval. Tanpa pelatihan. Waktu dalam UTC.

## a. Agregasi AUC BTC-USD Manual: 3 seed × 3 fold

wf_auc satu run = rerata AUC 3 fold; angka yang dilaporkan = mean ± std (ddof = 1) dari 3 wf_auc.

| Seed | Fold 1 | Fold 2 | Fold 3 | Rerata (= wf_auc) |
|---:|---:|---:|---:|---:|
| 42 | 0,705989 | 0,600753 | 0,637657 | 0,648133 |
| 43 | 0,568385 | 0,601109 | 0,612840 | 0,594111 |
| 44 | 0,673696 | 0,577358 | 0,641508 | 0,630854 |

mean = (0,648133 + 0,594111 + 0,630854) / 3 = **0,624366**; std (ddof = 1) = **0,027589** → dilaporkan 0,6244 ± 0,0276.

## b. Contoh candle backtest BTC-USD Manual seed 42

Aturan (`signals/generator.py`): kandidat LONG bila probabilitas ≥ 0,62, kandidat SHORT bila ≤ 0,38, selainnya HOLD. Kandidat menjadi sinyal bila minimal 2 syarat konfirmasi terpenuhi; bila tidak, HOLD. Syarat LONG: RSI(14) < 50; MACD histogram > 0; close < BB bawah × 1,005; Stochastic %K < 25; FGI < 0,35. Nilai fitur = baris t (candle asal t − 15 menit). Tepat: LONG bila label = 1, SHORT bila label = 0; HOLD tidak dinilai. Keputusan dihitung ulang dan cocok dengan berkas backtest pada 200/200 candle.

| Besaran | LONG pertama yang terbit | HOLD pertama, probabilitas di antara 0,38 dan 0,62 | HOLD pertama, probabilitas ≥ 0,62 tetapi konfirmasi < 2 |
|---|---:|---:|---:|
| Timestamp t | 2026-09-11 16:45 | 2026-09-10 02:45 | 2026-09-11 14:15 |
| Probabilitas | 0,644703 | 0,560551 | 0,628597 |
| `rsi_14` | 48,48 | 50,80 | 73,60 |
| `macd_hist` | -118,2239 | 15,2834 | 211,0616 |
| `close` | 77.780,98 | 78.238,70 | 79.033,42 |
| `bb_lower` | 76.664,34 | 77.869,93 | 76.015,41 |
| `bb_upper` | 79.428,40 | 78.378,43 | 78.713,86 |
| `stoch_k` | 20,16 | 31,30 | 86,62 |
| `fgi_normalized` | 0,56 | 0,69 | 0,56 |
| Kandidat (dari probabilitas) | LONG | HOLD | LONG |
| Syarat terpenuhi | RSI(14)=48.5 — momentum mendekati oversold; Stochastic %K=20.2 — area oversold (2) | – (tidak diperiksa) | MACD histogram positif (momentum naik) (1) |
| Sinyal akhir | **LONG** | **HOLD** | **HOLD** |
| Label | 0 | 0 | 0 |
| Tepat? | tidak | tidak dinilai | tidak dinilai |

## c. Rekap 200 candle BTC-USD Manual seed 42

| Tahap | LONG | SHORT | HOLD |
|---|---:|---:|---:|
| Kandidat menurut probabilitas | 30 | 0 | 170 |
| Lolos konfirmasi (sinyal akhir) | 13 | 0 | 187 |

HOLD akhir 187 = 170 karena probabilitas di zona tengah + 17 kandidat LONG + 0 kandidat SHORT yang gagal konfirmasi. Tepat: 2 dari 13 sinyal terbit.

## d. Kandidat dan sinyal akhir, 45 run

Kandidat dari kolom probabilitas berkas backtest; sinyal akhir dari kolom sinyal (sama dengan `n_signals_issued`/`n_signals_hold` di CSV hasil).

| Koin | Metode | Seed | Kandidat LONG | Kandidat SHORT | Zona tengah | LONG akhir | SHORT akhir | HOLD akhir |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC-USD | GA | 42 | 194 | 0 | 6 | 138 | 0 | 62 |
| BTC-USD | GA | 43 | 200 | 0 | 0 | 142 | 0 | 58 |
| BTC-USD | GA | 44 | 136 | 0 | 64 | 100 | 0 | 100 |
| BTC-USD | Grid | 42 | 0 | 0 | 200 | 0 | 0 | 200 |
| BTC-USD | Grid | 43 | 200 | 0 | 0 | 142 | 0 | 58 |
| BTC-USD | Grid | 44 | 183 | 0 | 17 | 132 | 0 | 68 |
| BTC-USD | Manual | 42 | 30 | 0 | 170 | 13 | 0 | 187 |
| BTC-USD | Manual | 43 | 117 | 0 | 83 | 94 | 0 | 106 |
| BTC-USD | Manual | 44 | 200 | 0 | 0 | 142 | 0 | 58 |
| ETH-USD | GA | 42 | 200 | 0 | 0 | 99 | 0 | 101 |
| ETH-USD | GA | 43 | 200 | 0 | 0 | 99 | 0 | 101 |
| ETH-USD | GA | 44 | 146 | 0 | 54 | 63 | 0 | 137 |
| ETH-USD | Grid | 42 | 99 | 0 | 101 | 54 | 0 | 146 |
| ETH-USD | Grid | 43 | 197 | 0 | 3 | 98 | 0 | 102 |
| ETH-USD | Grid | 44 | 200 | 0 | 0 | 99 | 0 | 101 |
| ETH-USD | Manual | 42 | 119 | 0 | 81 | 54 | 0 | 146 |
| ETH-USD | Manual | 43 | 170 | 0 | 30 | 79 | 0 | 121 |
| ETH-USD | Manual | 44 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | GA | 42 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | GA | 43 | 16 | 0 | 184 | 10 | 0 | 190 |
| SOL-USD | GA | 44 | 0 | 0 | 200 | 0 | 0 | 200 |
| SOL-USD | Grid | 42 | 160 | 0 | 40 | 80 | 0 | 120 |
| SOL-USD | Grid | 43 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | Grid | 44 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | Manual | 42 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | Manual | 43 | 200 | 0 | 0 | 99 | 0 | 101 |
| SOL-USD | Manual | 44 | 200 | 0 | 0 | 99 | 0 | 101 |
| LINK-USD | GA | 42 | 95 | 0 | 105 | 66 | 0 | 134 |
| LINK-USD | GA | 43 | 34 | 16 | 150 | 26 | 7 | 167 |
| LINK-USD | GA | 44 | 134 | 0 | 66 | 102 | 0 | 98 |
| LINK-USD | Grid | 42 | 30 | 0 | 170 | 13 | 0 | 187 |
| LINK-USD | Grid | 43 | 200 | 0 | 0 | 128 | 0 | 72 |
| LINK-USD | Grid | 44 | 34 | 0 | 166 | 15 | 0 | 185 |
| LINK-USD | Manual | 42 | 30 | 0 | 170 | 21 | 0 | 179 |
| LINK-USD | Manual | 43 | 198 | 0 | 2 | 127 | 0 | 73 |
| LINK-USD | Manual | 44 | 57 | 0 | 143 | 35 | 0 | 165 |
| SHIB-USD | GA | 42 | 27 | 0 | 173 | 8 | 0 | 192 |
| SHIB-USD | GA | 43 | 0 | 0 | 200 | 0 | 0 | 200 |
| SHIB-USD | GA | 44 | 0 | 172 | 28 | 0 | 101 | 99 |
| SHIB-USD | Grid | 42 | 39 | 0 | 161 | 18 | 0 | 182 |
| SHIB-USD | Grid | 43 | 74 | 0 | 126 | 31 | 0 | 169 |
| SHIB-USD | Grid | 44 | 86 | 0 | 114 | 51 | 0 | 149 |
| SHIB-USD | Manual | 42 | 162 | 0 | 38 | 84 | 0 | 116 |
| SHIB-USD | Manual | 43 | 95 | 0 | 105 | 38 | 0 | 162 |
| SHIB-USD | Manual | 44 | 13 | 0 | 187 | 5 | 0 | 195 |

| Metode (15 run) | Kandidat LONG | Kandidat SHORT | Zona tengah | LONG akhir | SHORT akhir | HOLD akhir |
|---|---:|---:|---:|---:|---:|---:|
| GA | 1582 | 188 | 1230 | 952 | 108 | 1940 |
| Grid | 1902 | 0 | 1098 | 1059 | 0 | 1941 |
| Manual | 1991 | 0 | 1009 | 1088 | 0 | 1912 |
