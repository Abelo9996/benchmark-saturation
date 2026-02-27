"""Analyze the v2 leaderboard data and check v1 availability."""
import pandas as pd
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

df = pd.read_parquet("data/contents_v2/data/train-00000-of-00001.parquet")

# Date range
df['sub_date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
df['upload_date'] = pd.to_datetime(df['Upload To Hub Date'], errors='coerce')

print("=== V2 Leaderboard Overview ===")
print(f"Total models: {len(df)}")
print(f"Submission date range: {df['sub_date'].min()} to {df['sub_date'].max()}")
print(f"Upload date range: {df['upload_date'].min()} to {df['upload_date'].max()}")
print(f"\nSubmissions per month:")
monthly = df.groupby(df['sub_date'].dt.to_period('M')).size()
for period, count in monthly.items():
    print(f"  {period}: {count}")

print(f"\nBenchmark columns and non-null counts:")
benchmarks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO', 'Average ⬆️']  # removed emoji
for b in benchmarks:
    if b in df.columns:
        vals = df[b].dropna()
        print(f"  {b}: n={len(vals)}, min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

print(f"\nModel types:")
print(df['T'].value_counts().to_string())

# Check if we have enough temporal spread for saturation analysis
print(f"\nModels with score > 50th percentile of Average, by quarter:")
median_avg = df['Average ⬆️'].median()  # removed emoji  
top_half = df[df['Average ⬆️'] >= median_avg]
quarterly = top_half.groupby(top_half['sub_date'].dt.to_period('Q')).agg(
    count=('Average ⬆️', 'size'),  # removed emoji
    max_avg=('Average ⬆️', 'max'),  # removed emoji
    mean_avg=('Average ⬆️', 'mean')  # removed emoji
)
print(quarterly.to_string())
