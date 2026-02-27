"""Generate all figures for the paper."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.stats import entropy as shannon_entropy
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 150,
    'figure.figsize': (7, 4),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

os.makedirs("figures", exist_ok=True)

v1 = pd.read_parquet("data/contents_v1/data/train-00000-of-00001-96886cb34a7bc800.parquet")
v2 = pd.read_parquet("data/contents_v2/data/train-00000-of-00001.parquet")
v1['date_parsed'] = pd.to_datetime(v1['date'], errors='coerce')
v2['date_parsed'] = pd.to_datetime(v2['Submission Date'], errors='coerce')
v1['month'] = v1['date_parsed'].dt.to_period('M')
v2['month'] = v2['date_parsed'].dt.to_period('M')

v1_benchmarks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']
v2_benchmarks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']

# ============================================================
# Fig 1: Max score trajectories over time (v1 + v2 side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for benchmark in v1_benchmarks:
    monthly = v1.groupby('month')[benchmark].agg(['max', 'mean']).dropna()
    months = [p.to_timestamp() for p in monthly.index]
    ax1.plot(months, monthly['max'], 'o-', label=benchmark, markersize=4)

ax1.set_title('V1 Benchmarks: Maximum Scores Over Time', fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim(30, 100)
ax1.legend(fontsize=8, loc='lower right')
ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Near-ceiling (90)')

for benchmark in v2_benchmarks:
    monthly = v2.groupby('month')[benchmark].agg(['max', 'mean']).dropna()
    months = [p.to_timestamp() for p in monthly.index]
    ax2.plot(months, monthly['max'], 'o-', label=benchmark, markersize=4)

ax2.set_title('V2 Benchmarks: Maximum Scores Over Time', fontweight='bold')
ax2.set_ylabel('Score')
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('figures/fig1_max_scores.png', bbox_inches='tight')
plt.close()
print("Fig 1 saved")

# ============================================================
# Fig 2: Score distribution compression (violin plots per quarter)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, benchmark in enumerate(v1_benchmarks):
    ax = axes[idx // 3][idx % 3]
    v1['quarter'] = v1['date_parsed'].dt.to_period('Q')
    quarters = sorted(v1['quarter'].dropna().unique())
    
    data_by_q = []
    labels = []
    for q in quarters:
        vals = v1.loc[v1['quarter'] == q, benchmark].dropna()
        if len(vals) >= 20:
            # Take top 50 models per quarter
            top50 = vals.nlargest(50).values
            data_by_q.append(top50)
            labels.append(str(q))
    
    if data_by_q:
        parts = ax.violinplot(data_by_q, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_title(benchmark, fontweight='bold')
    ax.set_ylabel('Score (Top 50)')

plt.suptitle('V1: Score Distribution Compression (Top 50 Models)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig2_violin_v1.png', bbox_inches='tight')
plt.close()
print("Fig 2 saved")

# ============================================================
# Fig 3: BDI (entropy) over time for all benchmarks
# ============================================================
bdi_results = pd.read_csv("bdi_results.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for benchmark in v1_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v1') & (bdi_results['benchmark'] == benchmark)]
    ax1.plot(range(len(data)), data['bdi_entropy'].values, 'o-', label=benchmark, markersize=5)

ax1.set_title('V1: BDI (Entropy) Over Time', fontweight='bold')
ax1.set_ylabel('BDI (normalized entropy)')
ax1.set_xlabel('Month index')
ax1.legend(fontsize=8)
ax1.set_ylim(0.3, 1.0)

for benchmark in v2_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v2') & (bdi_results['benchmark'] == benchmark)]
    ax2.plot(range(len(data)), data['bdi_entropy'].values, 'o-', label=benchmark, markersize=5)

ax2.set_title('V2: BDI (Entropy) Over Time', fontweight='bold')
ax2.set_ylabel('BDI (normalized entropy)')
ax2.set_xlabel('Month index')
ax2.legend(fontsize=8)
ax2.set_ylim(0.3, 1.0)

plt.tight_layout()
plt.savefig('figures/fig3_bdi_entropy.png', bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ============================================================
# Fig 4: Top-10 gap over time (discriminability)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for benchmark in v1_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v1') & (bdi_results['benchmark'] == benchmark)]
    ax1.plot(range(len(data)), data['top10_gap'].values, 'o-', label=benchmark, markersize=5)

ax1.set_title('V1: Top-10 Mean Gap Over Time', fontweight='bold')
ax1.set_ylabel('Mean gap between adjacent top-10 models')
ax1.set_xlabel('Month index')
ax1.legend(fontsize=8)

for benchmark in v2_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v2') & (bdi_results['benchmark'] == benchmark)]
    ax2.plot(range(len(data)), data['top10_gap'].values, 'o-', label=benchmark, markersize=5)

ax2.set_title('V2: Top-10 Mean Gap Over Time', fontweight='bold')
ax2.set_ylabel('Mean gap between adjacent top-10 models')
ax2.set_xlabel('Month index')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig4_top10_gap.png', bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ============================================================
# Fig 5: Saturation curve fit (GSM8K as poster child + all v1)
# ============================================================
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, benchmark in enumerate(v1_benchmarks):
    ax = axes[idx // 3][idx % 3]
    monthly_max = v1.groupby('month')[benchmark].max().dropna()
    months = sorted(monthly_max.index)
    
    # Filter months with enough data
    filtered = [(m, monthly_max[m]) for m in months if v1[v1['month'] == m][benchmark].count() >= 10]
    if len(filtered) < 4:
        continue
    
    t = np.arange(len(filtered))
    y = np.array([f[1] for f in filtered])
    month_labels = [str(f[0]) for f in filtered]
    
    ax.plot(t, y, 'ko-', markersize=6, label='Observed max')
    
    try:
        popt, _ = curve_fit(logistic, t, y, p0=[100, 1, len(t)//2], 
                           maxfev=5000, bounds=([y.max()-1, 0.01, -5], [100, 10, len(t)+10]))
        t_fine = np.linspace(-1, len(t)+3, 100)
        ax.plot(t_fine, logistic(t_fine, *popt), 'r--', alpha=0.7, label=f'Logistic (L={popt[0]:.1f})')
        ax.axhline(y=popt[0]*0.9, color='orange', linestyle=':', alpha=0.5, label=f'90% ceiling ({popt[0]*0.9:.1f})')
    except:
        pass
    
    ax.set_xticks(t)
    ax.set_xticklabels(month_labels, rotation=45, fontsize=7)
    ax.set_title(benchmark, fontweight='bold')
    ax.set_ylabel('Max Score')
    ax.legend(fontsize=7)

plt.suptitle('V1: Logistic Saturation Curve Fits', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig5_saturation_curves_v1.png', bbox_inches='tight')
plt.close()
print("Fig 5 saved")

# ============================================================
# Fig 6: Ceiling proximity heatmap (all benchmarks, both versions)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

all_benchmarks = v1_benchmarks + v2_benchmarks
all_labels = ['v1'] * len(v1_benchmarks) + ['v2'] * len(v2_benchmarks)

# Last month stats for each benchmark
ceiling_data = []
for lb, benchmarks in [('v1', v1_benchmarks), ('v2', v2_benchmarks)]:
    data = bdi_results[bdi_results['leaderboard'] == lb]
    for b in benchmarks:
        bm = data[data['benchmark'] == b].sort_values('month')
        if len(bm) > 0:
            last = bm.iloc[-1]
            ceiling_data.append({
                'benchmark': f"{lb}/{b}",
                'ceiling_proximity': last['ceiling_proximity'],
                'bdi_entropy': last['bdi_entropy'],
                'top10_gap': last['top10_gap'],
            })

ceil_df = pd.DataFrame(ceiling_data)
colors = ['#d32f2f' if c > 0.85 else '#f57c00' if c > 0.7 else '#4caf50' for c in ceil_df['ceiling_proximity']]

bars = ax.barh(ceil_df['benchmark'], ceil_df['ceiling_proximity'] * 100, color=colors)
ax.set_xlabel('Ceiling Proximity (max score as % of 100)')
ax.set_title('Benchmark Saturation Status (Latest Month)', fontweight='bold')
ax.axvline(x=85, color='red', linestyle='--', alpha=0.5, label='Saturated threshold')
ax.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
ax.legend()

# Add BDI values as text
for i, (_, row) in enumerate(ceil_df.iterrows()):
    ax.text(row['ceiling_proximity'] * 100 + 1, i, f"BDI={row['bdi_entropy']:.2f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig6_saturation_status.png', bbox_inches='tight')
plt.close()
print("Fig 6 saved")

print("\nAll figures saved to figures/")
