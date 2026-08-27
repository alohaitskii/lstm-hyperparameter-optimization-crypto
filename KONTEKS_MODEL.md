# Konteks: Folder `model/saved/` dan `model/optimized/`

> Dokumen ini menjelaskan dua folder berisi artefak model yang **tidak bisa
> di-upload** ke chat karena ukurannya (total ~49 MB, 111 file biner).
> Isinya file biner (`.keras`, `.pkl`) yang tidak bisa dibaca sebagai teks.

## Tentang proyek

Sistem prediksi arah pergerakan harga cryptocurrency berbasis **LSTM**
(klasifikasi biner: naik ≥0,3% dalam 4 candle ke depan / tidak). Data dari
**Yahoo Finance** (`yfinance`), timeframe 15m dengan konfirmasi 1h, 25 fitur
(OHLCV + indikator teknikal + Fear & Greed Index). Output berupa sinyal
LONG/SHORT/HOLD dengan entry, TP, dan SL berbasis ATR.

Proyek ini menjadi basis skripsi: **"Perbandingan Algoritma Genetika dan Grid
Search untuk Optimasi Hyperparameter LSTM pada Prediksi Arah Pergerakan Harga
Cryptocurrency."**

---

## 1. `model/saved/` — Model produksi (baseline manual)

**Peran:** model yang dipakai sistem untuk menghasilkan sinyal trading
sehari-hari. Sekaligus menjadi **baseline manual** dalam penelitian.

**Ukuran:** ~35 MB, 22 folder (satu per ticker).

```
model/saved/
├── BTC-USD/
│   ├── model.keras        (~1,6 MB — arsitektur + bobot LSTM terlatih)
│   ├── scaler.pkl         (~1 KB — MinMaxScaler, di-fit HANYA pada data train)
│   └── fgi_encoder.pkl    (~0,5 KB — LabelEncoder utk kategori Fear & Greed)
├── ETH-USD/               (isi sama)
├── SOL-USD/
└── ... total 22 ticker
```

**22 ticker:** BTC, ETH, BNB, SOL, XRP, ADA, DOGE, TRX, AVAX, LINK, DOT, LTC,
BCH, NEAR, ATOM, ETC, SHIB, HBAR, XLM, FIL, ALGO, VET (semua `-USD`).

**Hyperparameter:** nilai manual dari `config.yaml` — sama untuk semua ticker:

```
sequence_length = 60      lstm_units = [128, 64]
dropout_rate    = 0.2     learning_rate = 0.001
batch_size      = 32
```

**Ditulis oleh:** `python main.py --mode train` atau `python train_all.py`
**Dibaca oleh:** `--mode predict`, `--mode scan`, `--mode backtest`

> Satu model per koin, dilatih independen. Scaler juga per koin — karena
> rentang harga BTC (~$60.000) dan SHIB (~$0,000025) sangat berbeda.

---

## 2. `model/optimized/` — Hasil eksperimen skripsi

**Peran:** menyimpan model pemenang dari tiap metode optimasi hyperparameter.
**Tidak dipakai** untuk menghasilkan sinyal — murni bahan analisis penelitian.

**Ukuran:** ~14 MB, 15 kombinasi (5 ticker × 3 metode).

```
model/optimized/
├── BTC-USD/
│   ├── ga/                    ← hasil Algoritma Genetika
│   │   ├── model.keras
│   │   ├── scaler.pkl
│   │   ├── fgi_encoder.pkl
│   │   └── hyperparams.json   ← TAMBAHAN: hp terpilih + metrik + seed
│   ├── grid/                  ← hasil Grid Search (budget 50 evaluasi)
│   └── manual/                ← baseline, dievaluasi ulang protokol sama
├── ETH-USD/   (ga, grid, manual)
├── SOL-USD/   (ga, grid, manual)
├── LINK-USD/  (ga, grid, manual)
└── SHIB-USD/  (ga, grid, manual)
```

**Isi `hyperparams.json`** — satu-satunya file teks, contoh dari `BTC-USD/ga/`:

```json
{
  "hp": {
    "sequence_length": 60,
    "lstm_units_1": 64,
    "lstm_units_2": 48,
    "dropout_rate": 0.4,
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "wf_auc": 0.6200,
  "wf_f1": 0.1896,
  "seed": 42
}
```

**Ditulis oleh:** `python run_optimization.py` (hanya skrip ini)

---

## 3. Kenapa dipisah — ini keputusan metodologis

`model/saved/` adalah **baseline pembanding** dalam penelitian. Jika eksperimen
optimasi menimpanya, baseline hilang dan perbandingan GA vs Grid vs manual
tidak bisa direproduksi.

Secara teknis dijamin di kode: fungsi `save_artifacts()` punya parameter
`out_dir`. Alur optimasi **selalu** mengopernya ke
`model/optimized/{TICKER}/{method}/`, sehingga `model/saved/` tidak mungkin
tersentuh oleh eksperimen.

| | `model/saved/` | `model/optimized/` |
|---|---|---|
| Cakupan | 22 ticker | 5 ticker × 3 metode |
| Hyperparameter | Manual, seragam | Hasil pencarian per koin |
| Dipakai untuk sinyal | ✅ Ya | ❌ Tidak |
| File ekstra | — | `hyperparams.json` |
| Ukuran | ~35 MB | ~14 MB |

---

## 4. Ruang pencarian hyperparameter (GA & Grid identik)

| Hyperparameter | Kandidat |
|---|---|
| `sequence_length` | 30, 45, 60, 90, 120 |
| `lstm_units_1` | 32, 64, 96, 128 |
| `lstm_units_2` | 16, 32, 48, 64 |
| `dropout_rate` | 0.1, 0.2, 0.3, 0.4 |
| `learning_rate` | 0.01, 0.005, 0.001, 0.0005, 0.0001 |
| `batch_size` | 16, 32, 64 |

Total **4.800 kombinasi**. GA memakai ~48 evaluasi, Grid dibatasi budget 50
evaluasi (subsample deterministik merata) agar perbandingan setara.

**Penting:** `lookahead_n` (4) dan `move_threshold` (0,003) adalah *pendefinisi
tugas*, **bukan** hyperparameter — tidak ikut dicari, karena mengubahnya akan
mengubah distribusi label dan membuat perbandingan tidak valid.

---

## 5. Hasil eksperimen (seed 42, 5 koin)

Rata-rata lintas 5 koin:

| Metode | AUC pencarian | **AUC walk-forward** | F1 wf | Waktu (menit) | Evaluasi |
|---|---|---|---|---|---|
| GA | 0,6314 | 0,5820 | 0,2262 | 69,0 | 47,6 |
| Grid Search | 0,6272 | 0,5783 | 0,2210 | 106,2 | 50,0 |
| Manual | 0,5707 | **0,5857** | **0,2473** | 7,7 | 1,0 |

**Dua temuan utama:**

1. **GA lebih efisien dari Grid Search** — AUC pencarian sedikit lebih tinggi,
   evaluasi lebih sedikit (caching fitness), dan **35% lebih cepat**.
2. **Keunggulan optimasi hilang di walk-forward.** GA/Grid unggul ~0,06 AUC di
   split pencarian, tapi saat dievaluasi walk-forward 3 fold, manual justru
   sedikit di atas. Ini gejala *overfitting terhadap split pencarian*, dan
   membenarkan keputusan metodologis memakai walk-forward untuk metrik akhir.

Selisih AUC walk-forward antar metode sangat kecil (~0,007) dan baru diuji
pada **satu seed**, sehingga belum cukup untuk klaim superioritas yang kuat.

---

## 6. Protokol evaluasi (anti-leakage)

- **Fitness pencarian** = AUC pada satu split validasi kronologis, epoch
  tereduksi (20) + early stopping.
- **Metrik yang dilaporkan** = `walk_forward_validate` penuh (3 fold,
  `TimeSeriesSplit`) + backtest hit-rate.
- Scaler di-fit **hanya** pada data train tiap fold; sequence dibentuk
  **setelah** scaling; **tanpa shuffle**; fitur sudah di-`shift(1)` di praproses.

---

## Catatan tambahan

- File `.keras` dan `.pkl` adalah **biner** — tidak bisa dibaca sebagai teks.
  Satu-satunya file teks di kedua folder adalah `hyperparams.json`.
- Kedua folder masuk `.gitignore` (`model/saved/**/*.keras`,
  `model/optimized/**`) — tidak ikut ter-commit.
- Nama folder proyek `binance_lstm_futures/` bersifat historis: sumber data
  awalnya Binance, lalu Bybit, kini Yahoo Finance.
- Hasil numerik lengkap ada di `logs/optimization_results.csv`. **Perhatian:**
  file itu berisi 21 baris — 6 di antaranya sisa uji cepat (`search_epochs=2`)
  yang harus difilter; hanya baris dengan `search_epochs=20` yang valid.
