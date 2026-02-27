"""Exploratory analysis of v1 leaderboard — saturation overview."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np

df = pd.read_parquet("data/contents_v1/data/train-00000-of-00001-96886cb34a7bc800.parquet")

df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')

print(f"=== V1 Leaderboard ===")
print(f"Total models: {len(df)}")
print(f"Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")

benchmarks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']

# Overall stats
print(f"\nBenchmark stats:")
for b in benchmarks:
    vals = df[b].dropna()
    print(f"  {b}: n={len(vals)}, min={vals.min():.1f}, max={vals.max():.1f}, mean={vals.mean():.1f}, std={vals.std():.1f}")

# Monthly progression of MAX and TOP-10 MEAN scores
print(f"\n=== Monthly max scores (saturation tracker) ===")
df['month'] = df['date_parsed'].dt.to_period('M')

for b in benchmarks:
    print(f"\n--- {b} ---")
    monthly = df.groupby('month')[b].agg(['max', 'mean', 'std', 'count'])
    monthly = monthly[monthly['count'] >= 5]  # need at least 5 models
    for idx, row in monthly.iterrows():
        print(f"  {idx}: max={row['max']:.1f}, mean={row['mean']:.1f}, std={row['std']:.1f}, n={int(row['count'])}")

# Top-10 score compression over time
print(f"\n=== Top-10 IQR over time (score compression) ===")
for b in benchmarks:
    print(f"\n--- {b} ---")
    for period in sorted(df['month'].dropna().unique()):
        subset = df[df['month'] == period][b].dropna().sort_values(ascending=False)
        if len(subset) >= 10:
            top10 = subset.head(10)
            iqr = top10.quantile(0.75) - top10.quantile(0.25)
            gap = top10.iloc[0] - top10.iloc[9]
            print(f"  {period}: top1={top10.iloc[0]:.1f}, top10={top10.iloc[9]:.1f}, gap={gap:.2f}, iqr={iqr:.2f}")
