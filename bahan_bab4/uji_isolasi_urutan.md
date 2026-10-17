# Uji pengaruh urutan run pada evaluasi akhir (butir i.3)

Dihasilkan oleh `tools/uji_isolasi_urutan.py` (2026-10-17 08:19 UTC). Jalur kode = `reeval_final` (set_global_seed → prepare_ticker_data sekali per ticker → final_evaluate_and_save; artefak hanya seed pertama). Keluaran model dan backtest ditulis ke folder sementara; `logs/` dan `model/` tidak disentuh. Data: snapshot, cache-only.

## a. Urutan Langkah 4 dalam satu proses

Setiap run dibandingkan dengan baris Langkah 4 di `optimization_results.csv` (6 desimal, seperti tersimpan) dan, untuk Manual s42, dengan proses terpisah.

| Posisi | Run | wf_auc (uji ini) | wf_auc Langkah 4 | Sama dgn Langkah 4? | Sinyal terbit/HOLD (uji) | Langkah 4 | Detik |
|---:|---|---:|---:|---|---:|---:|---:|
| 1 | ga s42 | 0,621061 | 0,621061 | ya | 138/62 | 138/62 | 433.1 |
| 2 | ga s43 | 0,639955 | 0,639955 | ya | 142/58 | 142/58 | 202.3 |
| 3 | ga s44 | 0,647745 | 0,647745 | ya | 100/100 | 100/100 | 335.2 |
| 4 | grid s42 | 0,630755 | 0,659442 | **tidak** | 0/200 | 112/88 | 543.4 |
| 5 | grid s43 | 0,637905 | 0,640152 | **tidak** | 142/58 | 142/58 | 596.8 |
| 6 | grid s44 | 0,658689 | 0,643932 | **tidak** | 132/68 | 138/62 | 539.1 |
| 7 | manual s42 | 0,648133 | 0,664560 | **tidak** | 13/187 | 142/58 | 280.4 |

3 dari 7 run mereproduksi Langkah 4.

| Manual s42 | wf_auc | AUC per fold | Sinyal terbit/HOLD |
|---|---:|---|---:|
| Uji determinisme (Langkah 2b) | 0.6481331256878202 | – | 13/187 |
| Proses terpisah (uji ini) | 0.6481331256878202 | 0,705989, 0,600753, 0,637657 | 13/187 |
| Posisi 7 dalam satu proses (uji ini) | 0.6481331256878202 | 0,705989, 0,600753, 0,637657 | 13/187 |

Proses terpisah = uji determinisme: **ya**. Posisi 7 = proses terpisah: **ya**.

## b. Tahap pencarian (fitness)

Fitness = `fitness.evaluate` (seed 42, split 70/15/15, `search_epochs`). Dibandingkan dengan `val_auc_search` yang tercatat saat pencarian (6 desimal).

| Konfigurasi | val_auc_search (pencarian) | Proses baru | Setelah 7 evaluasi akhir dalam satu proses |
|---|---:|---:|---:|
| manual | 0,605245 | 0.6052449965493444 | 0.6052449965493444 |
| ga | 0,677047 | 0.6770470719368944 | 0.6770470719368944 |
| grid | 0,658459 | 0.6584589397089397 | 0.6584589397089397 |

Fitness tidak terdampak urutan: **ya**.

## c. Usulan isolasi: setiap (metode, seed) di subprocess baru

| Posisi | Run | wf_auc | Sinyal terbit/HOLD | Identik dgn proses terpisah? |
|---:|---|---:|---:|---|
| 1 | ga s42 | 0.6210611072896125 | 138/62 | – |
| 2 | manual s42 | 0.6481331256878202 | 13/187 | ya |
| 3 | grid s44 | 0.6586887006563443 | 132/68 | – |
| 4 | manual s42 | 0.6645601584570779 | 142/58 | **tidak** |

Identik = wf_auc, AUC tiap fold, jumlah sinyal, dan 200 probabilitas backtest sama persis. Manual s42 di posisi 2 dan 4 identik dengan proses terpisah: **tidak**. GA s42 (posisi 1 di kedua cara) identik: **ya**.
