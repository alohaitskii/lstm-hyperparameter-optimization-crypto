import pandas as pd
import json

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)

df = pd.read_csv('logs/optimization_results.csv')

print("=== Seluruh kolom yang tersedia di optimization_results.csv ===")
print(list(df.columns))
print()

# Filter baku yang sudah kita sepakati: buang sisa uji cepat, buang baris grid lama yang cacat
valid = df[(df['search_epochs'] == 20) & (
    (df['method'] != 'grid') | (df['n_signals_issued'].notna())
)].copy()

# Hanya 45 baris reeval snapshot (5 koin x 3 metode x seed 42/43/44). Baris
# pencarian (seed 42) lebih lama daripada baris reeval seed 42 pada kombinasi
# yang sama, jadi disisihkan dengan mengambil baris terbaru per (ticker, method, seed).
valid['timestamp'] = pd.to_datetime(valid['timestamp'])
valid = valid.sort_values('timestamp').drop_duplicates(
    subset=['ticker', 'method', 'seed'], keep='last'
)
assert len(valid) == 45, len(valid)
assert (valid.groupby(['ticker', 'method'])['seed'].nunique() == 3).all()

print(f"=== Baris valid setelah difilter: {len(valid)} dari {len(df)} total (reeval snapshot) ===")
print(valid['method'].value_counts())
print()

# Tabel lengkap per ticker x method
cols_show = ['ticker', 'method', 'seed', 'n_evals', 'search_duration_min', 'val_auc_search',
             'wf_auc', 'wf_f1', 'backtest_hit_rate', 'n_signals_issued', 'n_signals_hold']
print("=== Tabel lengkap (untuk dicek manual) ===")
print(valid[cols_show].sort_values(['ticker', 'method', 'seed']).to_string(index=False))
print()

# Rerata dari mean per koin (setiap koin berbobot sama), sama dengan laporan_4.md.
# Hit-rate NaN (run tanpa sinyal) tidak ikut dihitung.
print("=== Rata-rata lintas 5 koin per metode (bahan Bab 4) ===")
kolom = ['n_evals', 'search_duration_min', 'wf_auc', 'wf_f1', 'wf_precision', 'wf_recall',
         'backtest_hit_rate', 'n_signals_issued']
per_koin = valid.groupby(['method', 'ticker'])[kolom].mean()
ringkasan = per_koin.groupby('method').mean().round(4)
ringkasan['n_koin_hit_rate'] = per_koin['backtest_hit_rate'].notna().groupby('method').sum()
print(ringkasan)
print()

koin4 = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'LINK-USD']
print("=== Hit-rate dan sinyal pada 4 koin yang ketiga metodenya bersinyal (BTC, ETH, SOL, LINK) ===")
print(per_koin[per_koin.index.get_level_values('ticker').isin(koin4)]
      .groupby('method')[['backtest_hit_rate', 'n_signals_issued']].mean().round(4))
