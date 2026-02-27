"""
Benchmark Saturation Analysis Pipeline
- Compute BDI (Benchmark Discriminability Index) over time
- Fit saturation curves
- Measure rank instability
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, entropy
import json

# Load both leaderboards
v1 = pd.read_parquet("data/contents_v1/data/train-00000-of-00001-96886cb34a7bc800.parquet")
v2 = pd.read_parquet("data/contents_v2/data/train-00000-of-00001.parquet")

v1['date_parsed'] = pd.to_datetime(v1['date'], errors='coerce')
v2['date_parsed'] = pd.to_datetime(v2['Submission Date'], errors='coerce')

v1_benchmarks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']
v2_benchmarks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']

# ============================================================
# 1. BDI: Benchmark Discriminability Index
# ============================================================
# We define BDI as the effective discriminability of a benchmark at a given time:
#   BDI(t) = H(binned_scores) / H_max * (1 - ceiling_proximity)
# where:
#   H = Shannon entropy of score distribution binned into 10 equal-width bins
#   H_max = log2(10) (max entropy for 10 bins)
#   ceiling_proximity = max_score / theoretical_max (100)
#
# Simpler alternative (more interpretable):
#   BDI_gap(t) = mean pairwise gap between adjacent top-K models / theoretical_range
#   BDI_iqr(t) = IQR of top-K scores / range
#
# We'll compute all three and pick the cleanest.

def compute_bdi_entropy(scores, n_bins=20):
    """Entropy-based BDI. Higher = more discriminating."""
    if len(scores) < 10:
        return np.nan
    # Bin scores into fixed bins [0, 100]
    hist, _ = np.histogram(scores, bins=n_bins, range=(0, 100))
    hist = hist[hist > 0]  # remove empty bins for entropy
    probs = hist / hist.sum()
    H = entropy(probs, base=2)
    H_max = np.log2(n_bins)
    return H / H_max

def compute_bdi_gap(scores, top_k=20):
    """Mean gap between adjacent top-K models. Higher = more discriminating."""
    if len(scores) < top_k:
        return np.nan
    top = np.sort(scores)[-top_k:][::-1]
    gaps = np.diff(top) * -1  # positive gaps
    return np.mean(gaps)

def compute_bdi_iqr(scores, top_k=50):
    """IQR of top-K scores. Higher = more discriminating."""
    if len(scores) < top_k:
        top_k = len(scores)
    if top_k < 10:
        return np.nan
    top = np.sort(scores)[-top_k:]
    return np.percentile(top, 75) - np.percentile(top, 25)

def compute_ceiling_proximity(scores):
    """How close the max score is to 100. Higher = more saturated."""
    return np.max(scores) / 100.0

def compute_rank_stability(scores_t1, scores_t2, models_t1, models_t2, top_k=30):
    """Kendall's tau between rankings at two time points for overlapping models."""
    # Find common models (by fullname)
    common = set(models_t1) & set(models_t2)
    if len(common) < 10:
        return np.nan, len(common)
    
    # Get scores for common models
    s1 = {m: s for m, s in zip(models_t1, scores_t1) if m in common}
    s2 = {m: s for m, s in zip(models_t2, scores_t2) if m in common}
    
    models = sorted(common)
    v1_scores = [s1[m] for m in models]
    v2_scores = [s2[m] for m in models]
    
    tau, p = kendalltau(v1_scores, v2_scores)
    return tau, len(common)

# ============================================================
# 2. Compute BDI over time for all benchmarks
# ============================================================

def analyze_leaderboard(df, benchmarks, label, time_col='date_parsed'):
    """Compute BDI metrics over monthly windows."""
    df = df.copy()
    df['month'] = df[time_col].dt.to_period('M')
    
    results = []
    for benchmark in benchmarks:
        scores_col = df[benchmark].dropna()
        months = sorted(df['month'].dropna().unique())
        
        for month in months:
            mask = df['month'] == month
            scores = df.loc[mask, benchmark].dropna().values
            models = df.loc[mask & df[benchmark].notna(), 'fullname'].values if 'fullname' in df.columns else []
            
            if len(scores) < 10:
                continue
            
            results.append({
                'leaderboard': label,
                'benchmark': benchmark,
                'month': str(month),
                'n_models': len(scores),
                'max_score': np.max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'bdi_entropy': compute_bdi_entropy(scores),
                'bdi_gap_top20': compute_bdi_gap(scores, top_k=20),
                'bdi_iqr_top50': compute_bdi_iqr(scores, top_k=50),
                'ceiling_proximity': compute_ceiling_proximity(scores),
                'top10_gap': compute_bdi_gap(scores, top_k=10),
                'top10_iqr': compute_bdi_iqr(scores, top_k=10),
            })
    
    return pd.DataFrame(results)

print("Analyzing V1 leaderboard...")
v1_results = analyze_leaderboard(v1, v1_benchmarks, 'v1')
print(f"  {len(v1_results)} data points")

print("Analyzing V2 leaderboard...")
v2_results = analyze_leaderboard(v2, v2_benchmarks, 'v2', time_col='date_parsed')
print(f"  {len(v2_results)} data points")

all_results = pd.concat([v1_results, v2_results], ignore_index=True)

# ============================================================
# 3. Fit saturation curves (logistic)
# ============================================================

def logistic(t, L, k, t0):
    """Logistic function: L / (1 + exp(-k*(t-t0)))"""
    return L / (1 + np.exp(-k * (t - t0)))

print("\n=== Saturation Curve Fits ===")
saturation_fits = {}

for lb_label, benchmarks in [('v1', v1_benchmarks), ('v2', v2_benchmarks)]:
    df_lb = all_results[all_results['leaderboard'] == lb_label]
    
    for benchmark in benchmarks:
        bm_data = df_lb[df_lb['benchmark'] == benchmark].sort_values('month')
        if len(bm_data) < 4:
            continue
        
        # Use month index as time variable
        t = np.arange(len(bm_data))
        y = bm_data['max_score'].values
        
        try:
            # Fit logistic: ceiling (L), steepness (k), inflection (t0)
            popt, pcov = curve_fit(logistic, t, y, p0=[100, 1, len(t)//2], 
                                  maxfev=5000, bounds=([y.max()-1, 0.01, -5], [100, 10, len(t)+5]))
            L, k, t0 = popt
            predicted = logistic(t, *popt)
            residuals = y - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            # Time to 90% of ceiling
            t_90 = t0 + np.log(9) / k  # logistic reaches 90% at t0 + ln(9)/k
            months_list = bm_data['month'].values
            
            saturation_fits[f"{lb_label}_{benchmark}"] = {
                'ceiling': L,
                'steepness': k,
                'inflection_month_idx': t0,
                'r_squared': r_squared,
                't_90_idx': t_90,
            }
            
            print(f"\n  {lb_label}/{benchmark}:")
            print(f"    Ceiling: {L:.1f}, Steepness: {k:.2f}, R2: {r_squared:.3f}")
            print(f"    Max observed: {y.max():.1f}, Predicted ceiling: {L:.1f}")
            print(f"    Months: {months_list[0]} to {months_list[-1]}")
            if 0 <= t_90 < len(months_list):
                print(f"    90% ceiling reached at ~month index {t_90:.1f} ({months_list[min(int(t_90), len(months_list)-1)]})")
            elif t_90 >= len(months_list):
                print(f"    90% ceiling NOT YET reached (projected at index {t_90:.1f})")
            else:
                print(f"    90% ceiling was reached before tracking started")
        except Exception as e:
            print(f"\n  {lb_label}/{benchmark}: Fit failed - {e}")

# ============================================================
# 4. BDI Summary Table
# ============================================================
print("\n\n=== BDI Summary (first vs last month) ===")
for lb_label, benchmarks in [('v1', v1_benchmarks), ('v2', v2_benchmarks)]:
    df_lb = all_results[all_results['leaderboard'] == lb_label]
    print(f"\n--- {lb_label} ---")
    print(f"{'Benchmark':<15} {'BDI_ent(first)':<16} {'BDI_ent(last)':<16} {'Delta':<10} {'Gap20(first)':<14} {'Gap20(last)':<14} {'Ceil(last)':<12}")
    
    for benchmark in benchmarks:
        bm_data = df_lb[df_lb['benchmark'] == benchmark].sort_values('month')
        if len(bm_data) < 2:
            continue
        first = bm_data.iloc[0]
        last = bm_data.iloc[-1]
        delta = last['bdi_entropy'] - first['bdi_entropy']
        print(f"  {benchmark:<13} {first['bdi_entropy']:.3f}           {last['bdi_entropy']:.3f}           {delta:+.3f}     {first['bdi_gap_top20']:.2f}         {last['bdi_gap_top20']:.2f}         {last['ceiling_proximity']:.3f}")

# ============================================================
# 5. Rank Instability Analysis
# ============================================================
print("\n\n=== Rank Stability (Kendall's tau between consecutive months) ===")

for lb_label, df_lb_raw, benchmarks in [('v1', v1, v1_benchmarks), ('v2', v2, v2_benchmarks)]:
    df_lb_raw = df_lb_raw.copy()
    df_lb_raw['month'] = df_lb_raw['date_parsed'].dt.to_period('M')
    months = sorted(df_lb_raw['month'].dropna().unique())
    
    print(f"\n--- {lb_label} ---")
    for benchmark in benchmarks:
        taus = []
        for i in range(len(months)-1):
            m1, m2 = months[i], months[i+1]
            mask1 = df_lb_raw['month'] == m1
            mask2 = df_lb_raw['month'] == m2
            
            scores1 = df_lb_raw.loc[mask1, benchmark].dropna().values
            scores2 = df_lb_raw.loc[mask2, benchmark].dropna().values
            models1 = df_lb_raw.loc[mask1 & df_lb_raw[benchmark].notna(), 'fullname'].values
            models2 = df_lb_raw.loc[mask2 & df_lb_raw[benchmark].notna(), 'fullname'].values
            
            tau, n_common = compute_rank_stability(scores1, scores2, models1, models2)
            if not np.isnan(tau):
                taus.append((str(m1), str(m2), tau, n_common))
        
        if taus:
            mean_tau = np.mean([t[2] for t in taus])
            print(f"  {benchmark:<15} mean_tau={mean_tau:.3f} ({len(taus)} pairs)")
            for m1, m2, tau, n in taus:
                print(f"    {m1}->{m2}: tau={tau:.3f} (n={n})")

# Save all results
all_results.to_csv("bdi_results.csv", index=False)
with open("saturation_fits.json", "w") as f:
    json.dump(saturation_fits, f, indent=2)
print("\n\nSaved bdi_results.csv and saturation_fits.json")
