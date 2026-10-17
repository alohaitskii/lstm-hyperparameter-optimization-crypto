# Uji penyebab ketidakdeterministikan evaluasi akhir (butir i.3)

Dihasilkan oleh `tools/uji_penyebab.py` (selesai 2026-10-17 10:31 UTC, 124.7 menit). Setiap run = evaluasi akhir lengkap (walk-forward 3 fold + model pemenang + backtest) di proses Python baru, jalur `reeval_final`, keluaran ke folder sementara. Kondisi diterapkan sebelum TensorFlow diimpor. Putaran Manual bergiliran K1, K2, K3. CPU: Intel Core i5-12450H (4 P-core dengan hyper-threading = logical 0–7, 4 E-core = logical 8–11).

| Kondisi | Keterangan |
|---|---|
| K1 | seperti sekarang (kontrol) |
| K2 | affinity P-core saja, oneDNN aktif |
| K3 | TF_ENABLE_ONEDNN_OPTS=0 + enable_op_determinism() |

## Manual BTC-USD seed 42 (8 run per kondisi)

Hasil berbeda dikelompokkan berdasarkan wf_auc, AUC per fold, jumlah sinyal, dan 200 probabilitas.

| Kondisi | Hasil (wf_auc, terbit/HOLD) | Muncul | Run ke- | Lulus |
|---|---|---:|---|---|
| K1 | 0,648133, 13/187 | 7/8 | 1, 2, 3, 4, 5, 7, 8 | tidak |
|  | 0,664560, 142/58 | 1/8 | 6 |  |
| K2 | 0,648133, 13/187 | 8/8 | 1, 2, 3, 4, 5, 6, 7, 8 | **ya** |
| K3 | 0,653397, 142/58 | 7/8 | 1, 2, 3, 4, 5, 6, 7 | tidak |
|  | 0,628646, 11/189 | 1/8 | 8 |  |


## Grid BTC-USD seed 42 (4 run per kondisi)

Hasil berbeda dikelompokkan berdasarkan wf_auc, AUC per fold, jumlah sinyal, dan 200 probabilitas.

| Kondisi | Hasil (wf_auc, terbit/HOLD) | Muncul | Run ke- | Lulus |
|---|---|---:|---|---|
| K1 | tidak dijalankan | – | – | – |
| K2 | 0,630755, 0/200 | 4/4 | 1, 2, 3, 4 | **ya** |
| K3 | tidak dijalankan | – | – | – |

Pembanding Manual s42 sebelumnya: 0,648133 (uji determinisme, proses terpisah) dan 0,664560 (Langkah 4). Hasil yang muncul pada uji ini: 0,628646; 0,648133 (= uji determinisme / proses terpisah); 0,653397; 0,664560 (= Langkah 4 (reeval lama)).

**Kondisi terpilih: K2.** Aturan: K2 bila lulus, selain itu K3 bila lulus.
