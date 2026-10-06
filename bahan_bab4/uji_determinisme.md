# Uji determinisme (Langkah 2b)

Kombinasi baseline `[2, 3, 3, 1, 2, 1]`, seed 42, BTC-USD, data snapshot (cache-only). Posisi 1 dan posisi 7 dalam satu proses, diselingi 5 evaluasi lain (kromosom `[[0, 3, 2, 1, 2, 2], [3, 3, 2, 3, 2, 0], [2, 0, 0, 2, 4, 0], [3, 1, 0, 3, 2, 2], [2, 0, 2, 0, 3, 2]]`). Fitness dan walk-forward diuji di dua proses terpisah; proses walk-forward mengikuti jalur `reeval_final` persis (seed disetel, posisi 1 memuat data).

| Pemeriksaan | Hasil | Rincian |
|---|---|---|
| Fitness pencarian: AUC posisi 1 = posisi 7 | IDENTIK | 0.6052449965493444 vs 0.6052449965493444 |
| Fitness pencarian: F1 posisi 1 = posisi 7 | IDENTIK | 0.28858218318695106 vs 0.28858218318695106 |
| Walk-forward: AUC/F1/precision/recall rata-rata identik | IDENTIK | {"wf_auc": 0.6481331256878202, "wf_f1": 0.12552862667817488, "wf_precision": 0.07043505216949712, "wf_recall": 0.6606060606060606} vs {"wf_auc": 0.6481331256878202, "wf_f1": 0.12552862667817488, "wf_precision": 0.07043505216949712, "wf_recall": 0.6606060606060606} |
| Walk-forward: metrik ketiga fold identik | IDENTIK | 3 fold x 5 metrik |
| Backtest: hit-rate & jumlah sinyal identik | IDENTIK | {"hit_rate": 0.15384615384615385, "n_signals_issued": 13, "n_signals_hold": 187} |
| Backtest: 200 probabilitas per candle identik | IDENTIK | selisih maks 0 |
| fetch_all dipanggil tepat sekali sepanjang run | IDENTIK | 1x |

**Urutan evaluasi — fitness pencarian**

| Posisi | Kromosom | AUC | F1 | Detik |
|---:|---|---:|---:|---:|
| 1 | `[2, 3, 3, 1, 2, 1]` | 0.6052449965 | 0.2885821832 | 72.2 |
| 2 | `[0, 3, 2, 1, 2, 2]` | 0.5844567230 | 0.2801932367 | 28.9 |
| 3 | `[3, 3, 2, 3, 2, 0]` | 0.5957729572 | 0.2819499341 | 93.0 |
| 4 | `[2, 0, 0, 2, 4, 0]` | 0.5518441837 | 0.2885821832 | 67.1 |
| 5 | `[3, 1, 0, 3, 2, 2]` | 0.6277029924 | 0.2819499341 | 42.3 |
| 6 | `[2, 0, 2, 0, 3, 2]` | 0.5534008128 | 0.2885821832 | 18.2 |
| 7 | `[2, 3, 3, 1, 2, 1]` | 0.6052449965 | 0.2885821832 | 73.7 |

**Urutan evaluasi — walk-forward**

| Posisi | Evaluasi | AUC | Detik |
|---:|---|---:|---:|
| 1 | `final_evaluate_and_save` baseline | 0.6481331257 | 230.5 |
| 2 | fitness `[0, 3, 2, 1, 2, 2]` | 0.5844567230 | 28.8 |
| 3 | fitness `[3, 3, 2, 3, 2, 0]` | 0.5957729572 | 102.4 |
| 4 | fitness `[2, 0, 0, 2, 4, 0]` | 0.5518441837 | 72.0 |
| 5 | fitness `[3, 1, 0, 3, 2, 2]` | 0.6277029924 | 48.0 |
| 6 | fitness `[2, 0, 2, 0, 3, 2]` | 0.5534008128 | 20.0 |
| 7 | `final_evaluate_and_save` baseline | 0.6481331257 | 283.4 |

**Metrik per fold (posisi 1)**

| Fold | Accuracy | Precision | Recall | F1 | AUC | n_val_seq |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.074766 | 0.064433 | 1.000000 | 0.121065 | 0.705989 | 1177 |
| 2 | 0.774851 | 0.000000 | 0.000000 | 0.000000 | 0.600753 | 1177 |
| 3 | 0.197961 | 0.146872 | 0.981818 | 0.255521 | 0.637657 | 1177 |

**Kesimpulan: seluruh hasil identik.**
