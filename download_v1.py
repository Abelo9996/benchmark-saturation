"""Download v1 contents (compiled leaderboard)."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from huggingface_hub import snapshot_download
import pandas as pd

path = snapshot_download("open-llm-leaderboard-old/contents", repo_type="dataset",
                        local_dir="data/contents_v1", ignore_patterns=["*.md"])
print(f"Downloaded to: {path}")

for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(('.parquet', '.csv', '.json', '.jsonl')):
            fp = os.path.join(root, f)
            print(f"\nFile: {fp} ({os.path.getsize(fp)} bytes)")
            if f.endswith('.parquet'):
                df = pd.read_parquet(fp)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(df.head(2).to_string())
            elif f.endswith('.csv'):
                df = pd.read_csv(fp, nrows=2)
                print(f"Columns: {list(df.columns)}")
                print(df.to_string())
