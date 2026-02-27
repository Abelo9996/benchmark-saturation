"""
Round 1 fixes:
1. Add confidence intervals for BDI via bootstrap
2. Add R² values to saturation table, be honest about bad fits
3. Test BDI sensitivity to bin count
4. Count unique model families vs total submissions
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy
from scipy.optimize import curve_fit

v1 = pd.read_parquet("data/contents_v1/data/train-00000-of-00001-96886cb34a7bc800.parquet")
v2 = pd.read_parquet("data/contents_v2/data/train-00000-of-00001.parquet")
v1['date_parsed'] = pd.to_datetime(v1['date'], errors='coerce')
v2['date_parsed'] = pd.to_datetime(v2['Submission Date'], errors='coerce')
v1['month'] = v1['date_parsed'].dt.to_period('M')
v2['month'] = v2['date_parsed'].dt.to_period('M')

# ============================================================
# 1. Bootstrap CIs for BDI
# ============================================================
def compute_bdi(scores, n_bins=20):
    if len(scores) < 10:
        return np.nan
    hist, _ = np.histogram(scores, bins=n_bins, range=(0, 100))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    H = shannon_entropy(probs, base=2)
    return H / np.log2(n_bins)

def bootstrap_bdi(scores, n_bootstrap=1000, n_bins=20):
    bdis = []
    n = len(scores)
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bdis.append(compute_bdi(sample, n_bins))
    bdis = [b for b in bdis if not np.isnan(b)]
    return np.mean(bdis), np.percentile(bdis, 2.5), np.percentile(bdis, 97.5)

print("=== Bootstrap CIs for BDI (first and last month, V1) ===")
v1_benchmarks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']
months_v1 = sorted(v1['month'].dropna().unique())
# Find first/last months with >= 100 models
valid_months = [m for m in months_v1 if len(v1[v1['month'] == m]) >= 100]
first_m, last_m = valid_months[0], valid_months[-1]
print(f"Using months: {first_m} (n={len(v1[v1['month'] == first_m])}) and {last_m} (n={len(v1[v1['month'] == last_m])})")

for b in v1_benchmarks:
    scores_first = v1.loc[v1['month'] == first_m, b].dropna().values
    scores_last = v1.loc[v1['month'] == last_m, b].dropna().values
    
    mean_f, lo_f, hi_f = bootstrap_bdi(scores_first)
    mean_l, lo_l, hi_l = bootstrap_bdi(scores_last)
    
    # Check if CIs overlap (rough significance test)
    significant = hi_f < lo_l or hi_l < lo_f
    print(f"  {b:15s}: first={mean_f:.3f} [{lo_f:.3f}, {hi_f:.3f}]  last={mean_l:.3f} [{lo_l:.3f}, {hi_l:.3f}]  sig={'YES' if significant else 'no (overlap)'}")

# ============================================================
# 2. BDI sensitivity to bin count
# ============================================================
print("\n=== BDI Sensitivity to Bin Count ===")
# Use HellaSwag last month as test case
test_scores = v1.loc[v1['month'] == last_m, 'HellaSwag'].dropna().values
for n_bins in [5, 10, 15, 20, 25, 30, 50]:
    bdi = compute_bdi(test_scores, n_bins)
    print(f"  bins={n_bins:2d}: BDI={bdi:.4f}")

# ============================================================
# 3. Model family analysis
# ============================================================
print("\n=== Model Family Analysis ===")
# Extract base model family from fullname (rough: take org/first-part-of-name)
def get_family(name):
    if pd.isna(name):
        return 'unknown'
    # Remove HTML tags
    import re
    clean = re.sub(r'<[^>]+>', '', str(name)).strip()
    parts = clean.split('/')
    if len(parts) >= 2:
        return parts[0] + '/' + parts[1].split('-')[0]
    return clean.split('-')[0]

v1['family'] = v1['fullname'].apply(get_family)
v2['family'] = v2['fullname'].apply(get_family)

print(f"V1: {len(v1)} total models, {v1['family'].nunique()} unique families")
print(f"V2: {len(v2)} total models, {v2['family'].nunique()} unique families")

# Top families
print(f"\nV1 top 10 families:")
for fam, count in v1['family'].value_counts().head(10).items():
    print(f"  {fam}: {count}")

print(f"\nV2 top 10 families:")
for fam, count in v2['family'].value_counts().head(10).items():
    print(f"  {fam}: {count}")

# ============================================================
# 4. Saturation fit R² for all benchmarks (honest report)
# ============================================================
print("\n=== Saturation Fit Quality (all benchmarks) ===")
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

for lb_label, df_lb, benchmarks in [('v1', v1, v1_benchmarks), ('v2', v2, ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO'])]:
    for b in benchmarks:
        monthly_max = df_lb.groupby('month')[b].max().dropna()
        filtered = [(m, monthly_max[m]) for m in sorted(monthly_max.index) if df_lb[df_lb['month'] == m][b].count() >= 10]
        if len(filtered) < 4:
            continue
        t = np.arange(len(filtered))
        y = np.array([f[1] for f in filtered])
        try:
            popt, _ = curve_fit(logistic, t, y, p0=[100, 1, len(t)//2],
                               maxfev=5000, bounds=([y.max()-1, 0.01, -5], [100, 10, len(t)+10]))
            r2 = 1 - np.sum((y - logistic(t, *popt))**2) / np.sum((y - np.mean(y))**2)
            quality = "GOOD" if r2 > 0.7 else "OK" if r2 > 0.3 else "POOR"
            print(f"  {lb_label}/{b:15s}: R2={r2:.3f} L={popt[0]:.1f} [{quality}]")
        except:
            print(f"  {lb_label}/{b:15s}: FIT FAILED")
