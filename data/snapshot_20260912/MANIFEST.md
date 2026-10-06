# MANIFEST — data/snapshot_20260912

Dibuat: 2026-10-06 02:34 UTC  
Sumber: `data/cache/` (disalin dengan `cp -p`, bukan dipindah; mtime asli dipertahankan)  
Verifikasi: SHA256 setiap salinan identik dengan berkas sumber di `data/cache/` saat disalin.

Tujuan: membekukan jendela data yang dipakai eksperimen optimasi hyperparameter agar
seluruh metode (GA, Grid Search, Manual) dapat dijalankan dan direproduksi pada data yang
sama. Yahoo Finance membatasi riwayat 15m ke 60 hari, sehingga data ini tidak dapat
diunduh ulang di kemudian hari.

Zona waktu: kolom awal/akhir dalam **UTC**; kolom mtime dalam waktu lokal (WIB, UTC+7).

## Berkas

| Berkas | Jenis | Baris | Awal (UTC) | Akhir (UTC) | mtime asli (WIB) | Ukuran (B) |
|---|---|---:|---|---|---|---:|
| `ohlcv_yf_BTC_USD_15m_5000.parquet` | 15m | 5000 | 2026-07-22 03:45 | 2026-09-12 05:30 | 2026-09-12 12:36 | 206403 |
| `ohlcv_yf_BTC_USD_1h_5000.parquet` | 1h | 5000 | 2026-02-15 21:00 | 2026-09-12 05:00 | 2026-09-12 12:36 | 203344 |
| `ohlcv_yf_ETH_USD_15m_5000.parquet` | 15m | 5000 | 2026-07-22 04:00 | 2026-09-12 05:45 | 2026-09-12 12:57 | 195853 |
| `ohlcv_yf_ETH_USD_1h_5000.parquet` | 1h | 5000 | 2026-02-15 21:00 | 2026-09-12 05:00 | 2026-09-12 12:57 | 201939 |
| `ohlcv_yf_SOL_USD_15m_5000.parquet` | 15m | 5000 | 2026-07-22 05:00 | 2026-09-12 06:45 | 2026-09-12 13:45 | 134495 |
| `ohlcv_yf_SOL_USD_1h_5000.parquet` | 1h | 5000 | 2026-02-15 22:00 | 2026-09-12 06:00 | 2026-09-12 13:45 | 171071 |
| `ohlcv_yf_LINK_USD_15m_5000.parquet` | 15m | 5000 | 2026-07-22 06:00 | 2026-09-12 07:45 | 2026-09-12 14:49 | 147892 |
| `ohlcv_yf_LINK_USD_1h_5000.parquet` | 1h | 5000 | 2026-02-15 23:00 | 2026-09-12 07:00 | 2026-09-12 14:49 | 174653 |
| `ohlcv_yf_SHIB_USD_15m_5000.parquet` | 15m | 5000 | 2026-07-22 07:00 | 2026-09-12 08:45 | 2026-09-12 15:51 | 90188 |
| `ohlcv_yf_SHIB_USD_1h_5000.parquet` | 1h | 5000 | 2026-02-16 00:00 | 2026-09-12 08:00 | 2026-09-12 15:51 | 126312 |
| `fgi_90.parquet` | Fear & Greed (harian) | 90 | 2026-06-15 | 2026-09-12 | 2026-09-12 15:34 | 3653 |

## SHA256

```
71d1738b0a183ba55f015e2541d10e3fb46bf9c0d27aa8793007ac6ad3e5bdb0  ohlcv_yf_BTC_USD_15m_5000.parquet
5b47d4429d7c62df0f13a435ca9fc64ea304f235ad55e95577f53a57348cb8df  ohlcv_yf_BTC_USD_1h_5000.parquet
2d4930269ce644c76d8522967d832539483b598e417300d32eed524fd13ee067  ohlcv_yf_ETH_USD_15m_5000.parquet
19ae615833d6fc1be65a0da3e1568a8a1d713480643c3cb56124dc94658c8027  ohlcv_yf_ETH_USD_1h_5000.parquet
52bb4dbfb15af162dbe69a1401fadd03c33eb6352c3b84dd4172c61afc6c6a25  ohlcv_yf_SOL_USD_15m_5000.parquet
d58de05916168485334212d6ea35c4f5137d7b4fef27087c36dcfa610ba76f25  ohlcv_yf_SOL_USD_1h_5000.parquet
fb28ff34b5cbd2a96165db6c871938a9a8365b79495947469299528d67853963  ohlcv_yf_LINK_USD_15m_5000.parquet
502ec91d5ac6721d01b579aceeabaa77865d88e472a252e2405ad995666533a7  ohlcv_yf_LINK_USD_1h_5000.parquet
ab8f07607b61f9bc99940b8bf43050de21e94e3ca74430a55b50f32c01fba74d  ohlcv_yf_SHIB_USD_15m_5000.parquet
04818794752bbe7087098d0f42406afda2cb2897c6b55d5c076d8c3173077a25  ohlcv_yf_SHIB_USD_1h_5000.parquet
2a030d0773091bbe8771845ec0f6a8fe41393c1b6f616073c986bdbb76acfdec  fgi_90.parquet
```

## Berkas di `data/cache/` yang sengaja TIDAK disalin

Bukan masukan eksperimen optimasi (5 koin, candle_limit 5000, FGI 90 hari):

- `fgi_30.parquet`
- `ohlcv_yf_ADA_USD_15m_5000.parquet`
- `ohlcv_yf_ADA_USD_1h_5000.parquet`
- `ohlcv_yf_ALGO_USD_15m_5000.parquet`
- `ohlcv_yf_ALGO_USD_1h_5000.parquet`
- `ohlcv_yf_ATOM_USD_15m_5000.parquet`
- `ohlcv_yf_ATOM_USD_1h_5000.parquet`
- `ohlcv_yf_AVAX_USD_15m_5000.parquet`
- `ohlcv_yf_AVAX_USD_1h_5000.parquet`
- `ohlcv_yf_BCH_USD_15m_5000.parquet`
- `ohlcv_yf_BCH_USD_1h_5000.parquet`
- `ohlcv_yf_BNB_USD_15m_5000.parquet`
- `ohlcv_yf_BNB_USD_1h_5000.parquet`
- `ohlcv_yf_BTC_USD_15m_500.parquet`
- `ohlcv_yf_BTC_USD_1h_500.parquet`
- `ohlcv_yf_DOGE_USD_15m_5000.parquet`
- `ohlcv_yf_DOGE_USD_1h_5000.parquet`
- `ohlcv_yf_DOT_USD_15m_5000.parquet`
- `ohlcv_yf_DOT_USD_1h_5000.parquet`
- `ohlcv_yf_ETC_USD_15m_5000.parquet`
- `ohlcv_yf_ETC_USD_1h_5000.parquet`
- `ohlcv_yf_FIL_USD_15m_5000.parquet`
- `ohlcv_yf_FIL_USD_1h_5000.parquet`
- `ohlcv_yf_HBAR_USD_15m_5000.parquet`
- `ohlcv_yf_HBAR_USD_1h_5000.parquet`
- `ohlcv_yf_LTC_USD_15m_5000.parquet`
- `ohlcv_yf_LTC_USD_1h_5000.parquet`
- `ohlcv_yf_NEAR_USD_15m_5000.parquet`
- `ohlcv_yf_NEAR_USD_1h_5000.parquet`
- `ohlcv_yf_TRX_USD_15m_5000.parquet`
- `ohlcv_yf_TRX_USD_1h_5000.parquet`
- `ohlcv_yf_VET_USD_15m_5000.parquet`
- `ohlcv_yf_VET_USD_1h_5000.parquet`
- `ohlcv_yf_XLM_USD_15m_5000.parquet`
- `ohlcv_yf_XLM_USD_1h_5000.parquet`
- `ohlcv_yf_XRP_USD_15m_5000.parquet`
- `ohlcv_yf_XRP_USD_1h_5000.parquet`
