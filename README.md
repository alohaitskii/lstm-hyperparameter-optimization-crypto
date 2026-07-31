# Crypto LSTM: Multi-Pair Decision-Support Signal Generator (Yahoo Finance)

> **Decision-support tool** untuk prediksi arah harga crypto intraday berbasis LSTM,
> kini **multi-pair**: 22 koin, masing-masing dengan model LSTM independen.
> Sumber data: **Yahoo Finance** (`yfinance`), bebas geo-block di Indonesia dan
> merupakan sumber data yang sama dengan mayoritas paper akademik Indonesia
> tentang prediksi crypto dengan LSTM. Sentimen diambil dari **Fear & Greed Index**
> (alternative.me).
>
> Sistem ini **tidak mengeksekusi order**. Output berupa sinyal LONG/SHORT/HOLD
> terstruktur dengan entry zone, TP1/TP2, dan Stop Loss berbasis ATR, per pair
> atau sebagai **tabel scan terperingkat** lintas semua pair sekaligus.

**Disclaimer:** Bukan rekomendasi finansial. Gunakan dengan bijak.

---

## Riwayat Sumber Data

| Versi | Sumber | Alasan perubahan |
|-------|--------|------------------|
| 1.0 | Binance USDT-M FAPI | `fapi.binance.com` diblokir di Indonesia |
| 1.1 | Bybit V5 | Berfungsi, tapi diganti untuk menyesuaikan metodologi paper |
| **1.2 (sekarang)** | **Yahoo Finance** | Sesuai paper akademik; bebas blokir; tanpa API key |

Catatan: nama folder `binance_lstm_futures/` dipertahankan untuk alasan historis
(venv sudah terlanjur dibuat di dalamnya).

---

## Fitur

- **Data ingestion**: OHLCV dari Yahoo Finance (15m utama, 1h untuk konfirmasi trend) plus Fear & Greed Index harian
- **Indikator teknikal native**: RSI, MACD, Bollinger, EMA, ATR, Stochastic, dan Volume Ratio diimplementasikan langsung dengan pandas/numpy di `data/indicators.py` (tanpa pandas-ta karena bermasalah di numpy ≥ 2.0)
- **LSTM binary classifier**: memprediksi apakah harga naik minimal 0,3% dalam 4 candle ke depan (setara 1 jam)
- **Class-weight balancing**: label yang tidak seimbang ditangani otomatis saat training
- **Anti-leakage ketat**: `shift(1)` diterapkan ke semua fitur, scaler di-fit hanya pada fold training, validasi memakai `TimeSeriesSplit` walk-forward
- **Gerbang konfirmasi sinyal**: minimal 2 kondisi teknikal harus sejalan sebelum sinyal LONG/SHORT diterbitkan
- **Level risiko berbasis ATR**: TP1, TP2, dan SL dihitung dari ATR(14)
- **Log audit CSV**: semua prediksi tersimpan di `logs/predictions.csv`
- **Cache Parquet**: TTL 15 menit supaya tidak fetch berulang tanpa perlu

### Matriks Fitur (25 kolom)

```
OHLCV (5)        : open, high, low, close, volume
Teknikal (15)    : rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_lower,
                   bb_pct, bb_width, atr_14, ema_9, ema_21, ema_50,
                   stoch_k, stoch_d, vol_ratio
Konfirmasi 1h (3): rsi_1h, ema_21_1h, macd_hist_1h
Sentimen (2)     : fgi_normalized, fgi_classification_encoded
```

> Fitur derivatives (Open Interest, Funding Rate, L/S Ratio) **tidak tersedia**
> di Yahoo Finance dan telah dihapus dari sistem.

---

## Daftar Koin (22 ticker terverifikasi)

Semua ticker dikonfirmasi tersedia di Yahoo Finance dengan data intraday 15m.
Stablecoin dan token tanpa riwayat 15m yang andal sengaja dikecualikan.

```
BTC-USD   ETH-USD   BNB-USD   SOL-USD   XRP-USD   ADA-USD
DOGE-USD  TRX-USD   AVAX-USD  LINK-USD  DOT-USD   LTC-USD
BCH-USD   NEAR-USD  ATOM-USD  ETC-USD   SHIB-USD  HBAR-USD
XLM-USD   FIL-USD   ALGO-USD  VET-USD
```

Edit `data.tickers` di `config.yaml` untuk menambah/mengurangi.

---

## Struktur Proyek

```
binance_lstm_futures/
├── main.py                   # Entry point (--mode train | predict | scan | backtest)
├── train_all.py              # Orchestrator: latih semua ticker berurutan
├── config.yaml               # Semua parameter yang bisa di-tune
├── requirements.txt
│
├── data/
│   ├── fetcher.py            # Yahoo Finance OHLCV + FGI (+ rate-limit & retry)
│   ├── indicators.py         # Indikator teknikal native (pandas murni)
│   ├── preprocessor.py       # Feature engineering + anti-leakage
│   └── cache/                # Cache Parquet (dibuat otomatis)
│
├── model/
│   ├── lstm_model.py         # Build / train / save / load (+ class weights)
│   ├── validator.py          # Walk-forward validation (TimeSeriesSplit)
│   └── saved/                # SATU FOLDER PER TICKER:
│       ├── BTC-USD/          #   model.keras + scaler.pkl + fgi_encoder.pkl
│       ├── ETH-USD/
│       └── ...
│
├── signals/
│   ├── generator.py          # Probabilitas + gerbang konfirmasi → sinyal
│   └── formatter.py          # Box sinyal + tabel scan terperingkat (rich)
│
├── logs/                     # predictions.csv, validation_report_{TICKER}.csv,
│                             # scan_{ts}.csv, train_all_report_{ts}.csv
└── utils/                    # logger, helpers, ticker_utils
```

---

## Instalasi

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Tidak perlu API key**: Yahoo Finance dan alternative.me sama-sama publik.

### Verifikasi instalasi (opsional, tanpa network)

```bash
python smoke_test.py
```

Menjalankan uji offline dengan data sintetis: indikator, pipeline preprocessor
(25 fitur), pemeriksaan anti-leakage, pembentukan sequence, build + micro-train
model, dan pembuatan sinyal. Semua harus `OK`.

---

## Penggunaan

### Latih SEMUA ticker (multi-pair)

```bash
python train_all.py                                # semua 22 ticker (~6–11 jam CPU)
python train_all.py --fast                         # uji cepat: epochs=30, folds=3
python train_all.py --tickers BTC-USD ETH-USD SOL-USD   # subset tertentu
python train_all.py --skip-trained                 # resume setelah terputus
python train_all.py --from ETH-USD                 # lanjut dari posisi tertentu
```

Fitur: progress bar + ETA, ticker gagal/kurang data otomatis dilewati (SKIP/FAIL
dicatat), `Ctrl+C` aman (laporan tersimpan), ringkasan akhir + laporan CSV di
`logs/train_all_report_{timestamp}.csv`. Jeda antar fetch (`fetch_delay_seconds`)
dan jeda batch (`fetch_batch_pause`) melindungi dari rate-limit Yahoo.

### Training satu ticker (dengan walk-forward validation)

```bash
python main.py --mode train                     # ticker pertama di config (BTC-USD)
python main.py --mode train --ticker ETH-USD    # ticker tertentu
```

Yang terjadi:

1. Fetch ~5000 candle 15m + 1h dari Yahoo Finance, plus FGI 90 hari
2. Bangun matriks 25 fitur
3. Terapkan anti-leakage `shift(1)` dan buang baris NaN
4. Walk-forward validation 5 fold (`TimeSeriesSplit`); laporan per fold + agregat
5. Training model final dengan class weights pada split kronologis
6. Simpan artefak ke `model/saved/{TICKER}/` (model.keras, scaler.pkl, fgi_encoder.pkl)
7. Simpan `logs/validation_report_{TICKER}.csv`

### Scan semua pair, hasilnya tabel sinyal terperingkat

```bash
python main.py --mode scan              # semua ticker yang sudah terlatih
python main.py --mode scan --show-all   # tampilkan juga detail baris HOLD
```

Contoh output:

```
                 CRYPTO SCAN - 2026-06-11 13:30 UTC
              5 pair discan │ 2 LONG │ 1 SHORT │ 2 HOLD
┌───┬──────────┬──────────┬────────────┬───────────┬───────┬───────┬───────┬──────────────────┐
│ # │ Ticker   │ Signal   │ Confidence │     Entry │   TP1 │    SL │   R/R │ Basis            │
├───┼──────────┼──────────┼────────────┼───────────┼───────┼───────┼───────┼──────────────────┤
│ 1 │ SOL-USD  │ 🟢 LONG  │      81.2% │   $142.30 │ +1.8% │ -1.2% │ 1:1.5 │ RSI + MACD + FGI │
│ 2 │ ETH-USD  │ 🟢 LONG  │      74.5% │ $3,241.00 │ +0.9% │ -0.6% │ 1:1.5 │ BB + Stoch       │
│ 3 │ SHIB-USD │ 🔴 SHORT │      71.3% │ $0.000025 │ -1.2% │ +0.8% │ 1:1.4 │ RSI + MACD       │
└───┴──────────┴──────────┴────────────┴───────────┴───────┴───────┴───────┴──────────────────┘
             ⚠️  BUKAN REKOMENDASI FINANSIAL. GUNAKAN DENGAN BIJAK.
⚪ HOLD (2): BTC, DOGE
```

Peringkat: LONG dulu (confidence tertinggi), lalu SHORT, HOLD diringkas di bawah.
Hasil lengkap ada di `logs/scan_{timestamp}.csv`. Cache Parquet (TTL 15 menit) dipakai
ulang sehingga scan tidak menghantam Yahoo 22 kali kalau data masih segar.

### Generate sinyal satu ticker

```bash
python main.py --mode predict                   # ticker pertama di config
python main.py --mode predict --ticker SOL-USD  # ticker tertentu
```

Contoh output:

```
╭──────── CRYPTO LSTM SIGNAL GENERATOR ────────╮
│ Pair        : BTC-USD (Yahoo Finance)        │
│ Timeframe   : 15m  │  1h Trend: BULLISH      │
│ Signal      : 🟢 LONG                        │
│ Confidence  : 74.3%                          │
│ Entry Zone  : $67,420.00 – $67,490.00        │
│ TP1         : $67,822.00  (+0.60%)           │
│ TP2         : $68,240.00  (+1.22%)           │
│ Stop Loss   : $67,082.00  (-0.50%)           │
│ Risk/Reward : 1:1.80                         │
│                                              │
│ SIGNAL BASIS:                                │
│  ✅ RSI(14)=42.3 (momentum oversold)         │
│  ✅ MACD histogram positif (momentum naik)   │
│  ✅ Fear & Greed=28 (sinyal kontrarian)      │
│                                              │
│ Generated   : 2026-06-11 14:15:00 UTC        │
│ ⚠️  BUKAN REKOMENDASI FINANSIAL.              │
╰──────────────────────────────────────────────╯
```

### Backtest sinyal historis

```bash
python main.py --mode backtest --lookback 200
python main.py --mode backtest --lookback 200 --ticker ETH-USD
```

Hasil lengkap ada di `logs/backtest_{TICKER}_{timestamp}.csv`, plus ringkasan hit-rate di console.

---

## Referensi Konfigurasi

Semua parameter di `config.yaml`:

| Grup | Key | Default | Arti |
|------|-----|---------|------|
| `data` | `tickers` | 22 koin | Registry ticker Yahoo Finance (lihat daftar di atas) |
| `data` | `fetch_delay_seconds` | 3.0 | Jeda antar fetch Yahoo (rate-limit protection) |
| `data` | `fetch_batch_size` | 10 | Setelah N ticker, jeda batch (train_all.py) |
| `data` | `fetch_batch_pause` | 12.0 | Durasi jeda batch (detik) |
| `data` | `primary_tf` | 15m | Timeframe utama (batas riwayat Yahoo: 60 hari) |
| `data` | `secondary_tf` | 1h | Konfirmasi trend (batas riwayat: 730 hari) |
| `data` | `candle_limit` | 5000 | Jumlah candle (15m × 5000 ≈ 52 hari) |
| `features` | `sequence_length` | 60 | Panjang window input LSTM |
| `features` | `lookahead_n` | 4 | Prediksi 4 candle ke depan (= 1 jam di 15m) |
| `features` | `move_threshold` | 0.003 | Kenaikan minimal 0,3% untuk label BULLISH |
| `model` | `lstm_units` | [128, 64] | Lebar layer LSTM |
| `model` | `n_splits_cv` | 5 | Jumlah fold walk-forward |
| `signals` | `long_threshold` | 0.62 | Probabilitas ≥ → kandidat LONG |
| `signals` | `short_threshold` | 0.38 | Probabilitas ≤ → kandidat SHORT |
| `signals` | `min_confirmations` | 2 | Konfirmasi teknikal minimal |
| `signals` | `atr_tp1_multiplier` | 1.5 | TP1 = entry ± 1,5 × ATR |
| `signals` | `atr_sl_multiplier` | 1.0 | SL = entry ∓ 1,0 × ATR |

### Batas riwayat Yahoo Finance per interval

| Interval | Riwayat maksimal | Catatan |
|----------|------------------|---------|
| 15m | 60 hari (~5.700 candle) | Default sistem ini |
| 1h | 730 hari (~17.500 candle) | Alternatif bagus untuk riset (lebih banyak data) |
| 1d | Tak terbatas | Gaya paper akademik (harian, bertahun-tahun) |

Ganti `primary_tf`/`secondary_tf`/`candle_limit` di config untuk bereksperimen,
berguna untuk perbandingan eksperimen di skripsi/paper.

---

## Optimasi Hyperparameter (GA vs Grid Search) untuk Skripsi

Modul `optimization/` membandingkan tiga metode penyetelan hyperparameter LSTM:
**Algoritma Genetika**, **Grid Search** (ber-budget), dan **baseline manual**
(nilai `config.yaml` saat ini, yang menghasilkan model di `model/saved/`).

### Metodologi (ringkas, untuk bab metodologi)

- **Ruang pencarian identik** untuk GA & Grid (6 hyperparameter, 4.800 kombinasi
  penuh, lihat `optimization.search_space` di `config.yaml`).
  `lookahead_n` dan `move_threshold` adalah *pendefinisi tugas* dan **tidak**
  ikut dicari.
- **Fitness pencarian** = AUC pada satu split validasi kronologis
  (`split_chronological`), training dengan `search_epochs` tereduksi + early
  stopping. Anti-leakage penuh: scaler fit train-only, sequence dibentuk
  setelah scaling, tanpa shuffle.
- **Evaluasi akhir** konfigurasi terbaik tiap metode = `walk_forward_validate`
  penuh (`final_splits` fold) + backtest hit-rate, untuk mencegah overfitting pada
  satu split validasi.
- **Grid ber-budget**: subsample deterministik dengan posisi merata di seluruh
  enumerasi (bukan N kombinasi pertama, yang bias ke satu sudut ruang).
- **Reproducible**: seed (random/numpy/TF) dari config, dicatat di setiap CSV.
- **Baseline tidak tersentuh**: model hasil optimasi ditulis ke
  `model/optimized/{TICKER}/{method}/`; `model/saved/` tidak pernah ditimpa.

### Alur yang disarankan

```bash
# 1. Uji cepat end-to-end (menit-an)
python run_optimization.py --fast --tickers BTC-USD

# 2. Pilot: ukur waktu per-evaluasi + estimasi total (tidak menjalankan search)
python run_optimization.py --pilot

# 3. Eksperimen penuh: 5 koin x 3 metode (berjam-jam di CPU)
python run_optimization.py

# 4. Kalau terputus (Ctrl+C / Colab disconnect), lanjutkan:
python run_optimization.py --skip-done
```

Argumen penting: `--methods ga grid manual`, `--budget N` (budget-matched GA vs
Grid), `--final-splits N`, `--seed N`, `--output-root PATH`.

### Output

| File | Isi |
|------|-----|
| `logs/optimization_results.csv` | **Checkpoint master** (append per ticker×metode, dasar `--skip-done`) |
| `logs/optimization_summary_{ts}.csv` | Tabel utama skripsi (semua ticker × metode) |
| `logs/optimization_{TICKER}_{ts}.csv` | Tabel per koin (lampiran) |
| `logs/ga_history_{TICKER}_{ts}.csv` | Best/mean fitness per generasi (grafik konvergensi GA) |
| `model/optimized/{TICKER}/{method}/` | Model pemenang + scaler + encoder + `hyperparams.json` |

### Google Colab

Sesi Colab bisa terputus sewaktu-waktu. Mount Google Drive lalu arahkan output
ke sana agar hasil persisten:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
python run_optimization.py --output-root "/content/drive/MyDrive/skripsi-lstm" --skip-done
```

GPU terdeteksi otomatis dan dicatat di log. Untuk GPU, profil yang lebih besar
layak dipakai: `population_size: 16`, `generations: 10`, `grid.budget: 160`,
`search_epochs: 30`, `final_splits: 5` (edit `config.yaml`).

### Smoke test optimasi (offline)

```bash
python smoke_test_optimization.py
```

---

## Jaminan Anti-Leakage

1. **Feature shift**: `apply_anti_leakage()` memanggil `features.shift(1)` sehingga model hanya melihat data yang tersedia *sebelum* timestamp label.
2. **Buang baris target NaN**: `lookahead_n` baris terakhir dibuang karena label masa depan belum diketahui.
3. **Scaler fit per fold**: `MinMaxScaler` di-fit hanya pada fold training; data validasi hanya di-`.transform()`.
4. **Tanpa shuffle**: hanya `TimeSeriesSplit` dan slicing kronologis.
5. **Forward-fill**: FGI harian dan fitur 1h disebarkan ke candle 15m dengan `ffill` sehingga kejadian masa depan tidak pernah muncul di baris masa lalu.

---

## Keterbatasan

- **Tanpa eksekusi order.** Sistem hanya mencetak/mencatat sinyal.
- **Harga spot, bukan perpetual.** `BTC-USD` Yahoo adalah indeks spot; ada selisih basis kecil terhadap harga kontrak futures di exchange.
- **Volume Yahoo dalam USD** (quote currency), hanya dipakai relatif (vol_ratio), jadi tidak mempengaruhi model.
- **Riwayat 15m terbatas 60 hari.** Untuk dataset lebih panjang gunakan `1h` atau `1d`.
- **FGI bersifat harian**, disebar ke candle 15m, jadi pergerakannya lambat.
- **Tidak ada jaminan profit.** Pasar crypto non-stasioner; model bisa terdegradasi seiring waktu.

---

## Lisensi

Untuk penggunaan pribadi/akademik. Tanpa jaminan.

---

*Versi dokumentasi 1.2. Sumber data Yahoo Finance (BTC-USD), intraday 15m + konfirmasi 1h, 25 fitur.*
