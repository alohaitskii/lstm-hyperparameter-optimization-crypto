# Uji ulang kombinasi reeval di proses terpisah

Dihasilkan oleh `tools/uji_perbaikan_isolasi.py ulang` (2026-10-17 13:37 UTC). Setiap kombinasi dijalankan ulang dengan `reeval_final.py` di sandbox lalu dibandingkan dengan baris reeval di `logs/`.

| Kombinasi | wf_auc (logs) | wf_auc (ulang) | wf_auc | fold | sinyal | 200 prob. |
|---|---:|---:|---|---|---|---|
| ETH-USD manual s44 | 0.602477 | 0.602477 | ya | ya | ya | ya |
| BTC-USD ga s43 | 0.639955 | 0.639955 | ya | ya | ya | ya |

**HASIL: LULUS**
