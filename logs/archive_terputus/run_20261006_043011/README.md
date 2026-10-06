# Run terputus — 3a, dimulai 2026-10-06 04:30 UTC (11:30 WIB)

Run `run_optimization.py --methods ga manual` dihentikan dari luar pada
2026-10-06 sekitar 06:54 UTC (13:54 WIB), di tengah GA ETH-USD (generasi 0–5
selesai, generasi 6 sedang dievaluasi).

Penyebab: aplikasi Claude desktop ter-restart pukul 13:55:15 WIB (diduga
pembaruan otomatis; Windows mencatat aktivitas paket Claude_2.19675.1.0 pukul
13:55:11) dan ikut menghentikan proses yang diluncurkan dari dalamnya.
Windows sendiri tidak restart (boot terakhir 08:56 WIB). Peluncur tidak sempat
menulis baris SELESAI, sehingga proses tidak berakhir karena error.

Yang SAH dan tetap di logs/: BTC-USD GA dan Manual (selesai lengkap,
tercatat di optimization_results.csv beserta ga_history/ga_trace/
ga_tournament/backtest BTC). Run lanjutan memakai --skip-done.

Di folder ini: berkas parsial GA ETH-USD (tidak menghasilkan baris hasil) dan
salinan log run. Karena GA deterministik, generasi 0–5 pada run lanjutan
harus identik dengan berkas parsial ini.
