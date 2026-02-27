"""Generate publication-quality figures for the benchmark saturation paper."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import entropy as shannon_entropy
import warnings
warnings.filterwarnings('ignore')

# Consistent style
COLORS_V1 = {
    'ARC': '#1f77b4',
    'HellaSwag': '#ff7f0e', 
    'MMLU': '#2ca02c',
    'TruthfulQA': '#d62728',
    'Winogrande': '#9467bd',
    'GSM8K': '#8c564b',
}
COLORS_V2 = {
    'IFEval': '#1f77b4',
    'BBH': '#ff7f0e',
    'MATH Lvl 5': '#2ca02c',
    'GPQA': '#d62728',
    'MUSR': '#9467bd',
    'MMLU-PRO': '#8c564b',
}
MARKERS_V1 = {'ARC': 'o', 'HellaSwag': 's', 'MMLU': '^', 'TruthfulQA': 'D', 'Winogrande': 'v', 'GSM8K': 'P'}
MARKERS_V2 = {'IFEval': 'o', 'BBH': 's', 'MATH Lvl 5': '^', 'GPQA': 'D', 'MUSR': 'v', 'MMLU-PRO': 'P'}

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs("paper/figures", exist_ok=True)

v1 = pd.read_parquet("data/contents_v1/data/train-00000-of-00001-96886cb34a7bc800.parquet")
v2 = pd.read_parquet("data/contents_v2/data/train-00000-of-00001.parquet")
v1['date_parsed'] = pd.to_datetime(v1['date'], errors='coerce')
v2['date_parsed'] = pd.to_datetime(v2['Submission Date'], errors='coerce')
v1['month'] = v1['date_parsed'].dt.to_period('M')
v2['month'] = v2['date_parsed'].dt.to_period('M')

v1_benchmarks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']
v2_benchmarks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']

bdi_results = pd.read_csv("bdi_results.csv")

def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# ============================================================
# Fig 1: Max score trajectories (v1 + v2 side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for b in v1_benchmarks:
    monthly = v1.groupby('month')[b].max().dropna()
    months_filtered = [(m, monthly[m]) for m in sorted(monthly.index) if v1[v1['month'] == m][b].count() >= 10]
    if not months_filtered:
        continue
    x = [p[0].to_timestamp() for p in months_filtered]
    y = [p[1] for p in months_filtered]
    ax1.plot(x, y, color=COLORS_V1[b], marker=MARKERS_V1[b], markersize=5, linewidth=1.5, label=b)

ax1.axhline(y=90, color='#888888', linestyle=':', alpha=0.6, linewidth=1)
ax1.text(ax1.get_xlim()[0], 90.5, 'Near-ceiling', fontsize=8, color='#888888')
ax1.set_title('(a) V1 Benchmarks', fontweight='bold', fontsize=12)
ax1.set_ylabel('Maximum Score', fontsize=11)
ax1.set_ylim(35, 100)
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax1.tick_params(axis='x', rotation=30)

for b in v2_benchmarks:
    monthly = v2.groupby('month')[b].max().dropna()
    months_filtered = [(m, monthly[m]) for m in sorted(monthly.index) if v2[v2['month'] == m][b].count() >= 10]
    if not months_filtered:
        continue
    x = [p[0].to_timestamp() for p in months_filtered]
    y = [p[1] for p in months_filtered]
    ax2.plot(x, y, color=COLORS_V2[b], marker=MARKERS_V2[b], markersize=5, linewidth=1.5, label=b)

ax2.set_title('(b) V2 Benchmarks', fontweight='bold', fontsize=12)
ax2.set_ylabel('Maximum Score', fontsize=11)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax2.tick_params(axis='x', rotation=30)

plt.suptitle('Figure 1: Maximum Score Trajectories Over Time', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('paper/figures/fig1_max_scores.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig1_max_scores.png', bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ============================================================
# Fig 2: BDI over time (v1 + v2)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for b in v1_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v1') & (bdi_results['benchmark'] == b)].sort_values('month')
    ax1.plot(range(len(data)), data['bdi_entropy'].values, color=COLORS_V1[b], marker=MARKERS_V1[b], 
             markersize=5, linewidth=1.5, label=b)

ax1.set_title('(a) V1: BDI Decline', fontweight='bold', fontsize=12)
ax1.set_ylabel('BDI (Normalized Entropy)', fontsize=11)
ax1.set_xlabel('Month Index (0 = Sep 2023)', fontsize=10)
ax1.legend(fontsize=8, loc='lower left', framealpha=0.9)
ax1.set_ylim(0.3, 1.0)
ax1.axhline(y=0.5, color='#d32f2f', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(0.1, 0.505, 'Low discriminability', fontsize=7, color='#d32f2f')

for b in v2_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v2') & (bdi_results['benchmark'] == b)].sort_values('month')
    ax2.plot(range(len(data)), data['bdi_entropy'].values, color=COLORS_V2[b], marker=MARKERS_V2[b],
             markersize=5, linewidth=1.5, label=b)

ax2.set_title('(b) V2: BDI Trends', fontweight='bold', fontsize=12)
ax2.set_ylabel('BDI (Normalized Entropy)', fontsize=11)
ax2.set_xlabel('Month Index (0 = Jun 2024)', fontsize=10)
ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax2.set_ylim(0.3, 1.0)

plt.suptitle('Figure 2: Benchmark Discriminability Index Over Time', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('paper/figures/fig2_bdi.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig2_bdi.png', bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ============================================================
# Fig 3: Saturation curves (v1 — 2x3 grid)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for idx, b in enumerate(v1_benchmarks):
    ax = axes[idx // 3][idx % 3]
    monthly_max = v1.groupby('month')[b].max().dropna()
    filtered = [(m, monthly_max[m]) for m in sorted(monthly_max.index) if v1[v1['month'] == m][b].count() >= 10]
    if len(filtered) < 4:
        continue
    
    t = np.arange(len(filtered))
    y = np.array([f[1] for f in filtered])
    labels = [str(f[0]) for f in filtered]
    
    ax.plot(t, y, 'ko', markersize=6, zorder=5)
    
    try:
        popt, _ = curve_fit(logistic, t, y, p0=[100, 1, len(t)//2],
                           maxfev=5000, bounds=([y.max()-1, 0.01, -5], [100, 10, len(t)+10]))
        L, k, t0 = popt
        t_fine = np.linspace(-1, len(t)+3, 200)
        r2 = 1 - np.sum((y - logistic(t, *popt))**2) / np.sum((y - np.mean(y))**2)
        
        ax.plot(t_fine, logistic(t_fine, *popt), color=COLORS_V1[b], linewidth=2, alpha=0.8)
        ax.axhline(y=L*0.9, color='#d32f2f', linestyle=':', alpha=0.5, linewidth=1)
        ax.fill_between(t_fine, logistic(t_fine, *popt), alpha=0.05, color=COLORS_V1[b])
        
        ax.text(0.95, 0.05, f'L={L:.1f}\n$R^2$={r2:.3f}', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    except:
        pass
    
    ax.set_xticks(t[::2])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=45, fontsize=8)
    ax.set_title(b, fontweight='bold', fontsize=11, color=COLORS_V1[b])
    ax.set_ylabel('Max Score', fontsize=9)

plt.suptitle('Figure 3: Logistic Saturation Curves (V1 Benchmarks)', fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('paper/figures/fig3_saturation_v1.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig3_saturation_v1.png', bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ============================================================
# Fig 4: Saturation status bar chart (both v1 and v2)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

all_data = []
for lb, benchmarks, colors in [('v1', v1_benchmarks, COLORS_V1), ('v2', v2_benchmarks, COLORS_V2)]:
    data = bdi_results[bdi_results['leaderboard'] == lb]
    for b in benchmarks:
        bm = data[data['benchmark'] == b].sort_values('month')
        if len(bm) > 0:
            last = bm.iloc[-1]
            all_data.append({
                'label': f"{b} ({lb.upper()})",
                'cp': last['ceiling_proximity'] * 100,
                'bdi': last['bdi_entropy'],
                'color': colors[b],
                'lb': lb,
            })

all_data.sort(key=lambda x: x['cp'], reverse=True)

y_pos = range(len(all_data))
bars = ax.barh(y_pos, [d['cp'] for d in all_data], color=[d['color'] for d in all_data], alpha=0.85, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([d['label'] for d in all_data], fontsize=9)
ax.set_xlabel('Ceiling Proximity (Max Score as % of 100)', fontsize=11)
ax.set_title('Figure 4: Benchmark Saturation Status', fontweight='bold', fontsize=13)

# Threshold lines
ax.axvline(x=85, color='#d32f2f', linestyle='--', alpha=0.7, linewidth=1.5)
ax.axvline(x=70, color='#f57c00', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(85.5, len(all_data)-0.5, 'Saturated', fontsize=8, color='#d32f2f', fontweight='bold')
ax.text(70.5, len(all_data)-0.5, 'Warning', fontsize=8, color='#f57c00', fontweight='bold')

# BDI annotations
for i, d in enumerate(all_data):
    ax.text(d['cp'] + 1, i, f"BDI={d['bdi']:.2f}", va='center', fontsize=7.5, color='#333333')

ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig('paper/figures/fig4_status.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig4_status.png', bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ============================================================
# Fig 5: Top-10 gap compression over time
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for b in v1_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v1') & (bdi_results['benchmark'] == b)].sort_values('month')
    if len(data) > 0:
        ax1.plot(range(len(data)), data['top10_gap'].values, color=COLORS_V1[b], marker=MARKERS_V1[b],
                markersize=5, linewidth=1.5, label=b)

ax1.set_title('(a) V1: Top-10 Mean Gap', fontweight='bold', fontsize=12)
ax1.set_ylabel('Mean Gap (points)', fontsize=11)
ax1.set_xlabel('Month Index', fontsize=10)
ax1.legend(fontsize=8, framealpha=0.9)
ax1.axhline(y=1.0, color='#d32f2f', linestyle=':', alpha=0.5)
ax1.text(0.1, 1.1, 'Indistinguishable threshold', fontsize=7, color='#d32f2f')

for b in v2_benchmarks:
    data = bdi_results[(bdi_results['leaderboard'] == 'v2') & (bdi_results['benchmark'] == b)].sort_values('month')
    if len(data) > 0:
        ax2.plot(range(len(data)), data['top10_gap'].values, color=COLORS_V2[b], marker=MARKERS_V2[b],
                markersize=5, linewidth=1.5, label=b)

ax2.set_title('(b) V2: Top-10 Mean Gap', fontweight='bold', fontsize=12)
ax2.set_ylabel('Mean Gap (points)', fontsize=11)
ax2.set_xlabel('Month Index', fontsize=10)
ax2.legend(fontsize=8, framealpha=0.9)

plt.suptitle('Figure 5: Score Gap Compression Among Top Models', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('paper/figures/fig5_top10_gap.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig5_top10_gap.png', bbox_inches='tight')
plt.close()
print("Fig 5 done")

print("\nAll figures saved to paper/figures/")
