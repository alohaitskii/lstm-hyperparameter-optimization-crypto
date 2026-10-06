# Laporan Langkah 3b — Grid Search di atas snapshot

Data: `data/snapshot_20260912/` (cache-only, setiap berkas diverifikasi terhadap
MANIFEST). Grid ber-budget 50 evaluasi, stride relatif prima (97) atas ruang
4.800 kombinasi, seed 42, `search_epochs` 20. Fitness = AUC pada satu split
validasi kronologis — prosedur identik dengan GA (`fitness.evaluate`).

Run dari terminal VS Code, 6 Okt 20:59 WIB – 7 Okt 04:21 WIB (442,2 menit, kode
keluar 0, tanpa baris GAGAL). Laptop tidak dipakai, kondisi sama dengan run GA
lanjutan.

## Hasil per koin

| Koin | Evaluasi | Pencarian (menit) | Total (menit) | Best fitness | Terbaik di evaluasi ke- | best_hp |
|---|---:|---:|---:|---:|---:|---|
| BTC-USD | 50 | 65,0 | 75,5 | 0,6585 | 49 | seq 120, u 128/32, drop 0,3, lr 0,001, bs 16 |
| ETH-USD | 50 | 73,6 | 75,8 | 0,6640 | 12 | seq 45, u 32/32, drop 0,4, lr 0,01, bs 64 |
| SOL-USD | 50 | 75,7 | 80,1 | 0,5918 | 42 | seq 120, u 32/48, drop 0,2, lr 0,01, bs 64 |
| LINK-USD | 50 | 89,2 | 94,4 | 0,5824 | 26 | seq 60, u 96/16, drop 0,2, lr 0,0005, bs 32 |
| SHIB-USD | 50 | 106,8 | 115,3 | 0,6154 | 30 | seq 60, u 128/48, drop 0,4, lr 0,001, bs 64 |

Total pencarian Grid lima koin: 410,3 menit untuk 250 evaluasi.

## Kelengkapan dan konsistensi `grid_history`

Kelima koin: 50 baris berurutan; ke-50 kombinasi identik dengan
`iter_grid_budget(50)` (kombinasi yang sama untuk semua koin); `best_so_far` =
maksimum berjalan; evaluasi terbaik dan best_hp = baris hasil; jumlah
`waktu_detik` = waktu pencarian. Semua lulus.

Cakupan ke-50 kombinasi (setiap nilai kandidat setiap gen teruji):

| Gen | Frekuensi |
|---|---|
| sequence_length | 30:10, 45:10, 60:10, 90:10, 120:10 |
| lstm_units_1 | 32:15, 64:10, 96:15, 128:10 |
| lstm_units_2 | 16:12, 32:13, 48:12, 64:13 |
| dropout_rate | 0,1:13, 0,2:12, 0,3:13, 0,4:12 |
| learning_rate | 0,01:10, 0,005:9, 0,001:11, 0,0005:9, 0,0001:11 |
| batch_size | 16:17, 32:17, 64:16 |

## Jendela data

`tools/cek_jendela_data.py --semua-harus-cocok`: **15/15** backtest (10 dari 3a,
5 Grid) tepat pada snapshot.

## Catatan waktu

Evaluasi Grid BTC ke-4 (51,4 detik) berjalan bersamaan dengan regenerasi
E1_E2 selama ±1 menit. Kombinasi yang sama memakan 51,3 detik (ETH), 46,4 (LINK),
63,7 (SOL), dan 89,4 (SHIB), sehingga rasio BTC terhadap median koin lain 0,89 —
tidak ada indikasi pembengkakan.

Waktu GA BTC (53,5 menit) diukur pada run 3a pertama yang terbukti ±40% lebih
lambat untuk pekerjaan identik; GA BTC dijalankan ulang dari VS Code agar
sebanding (lihat laporan berikutnya). Angka GA BTC di bawah bersifat sementara.

## Pratinjau GA vs Grid (fitness pencarian, satu split)

| Koin | GA: fitness | GA: eval | GA: pencarian | Grid: fitness | Grid: eval | Grid: pencarian |
|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | 0,6770 | 46 | 53,5* | 0,6585 | 50 | 65,0 |
| ETH-USD | 0,6650 | 49 | 69,6 | 0,6640 | 50 | 73,6 |
| SOL-USD | 0,5806 | 45 | 108,0 | 0,5918 | 50 | 75,7 |
| LINK-USD | 0,5810 | 47 | 50,1 | 0,5824 | 50 | 89,2 |
| SHIB-USD | 0,6183 | 46 | 71,9 | 0,6154 | 50 | 106,8 |

\* Sementara, menunggu run ulang GA BTC.

Ini fitness **pencarian** pada satu split validasi, bukan metrik yang
dilaporkan. Perbandingan yang sah adalah walk-forward 3 seed pada Langkah 4.
