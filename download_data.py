"""Download and explore the Open LLM Leaderboard contents (v2) and old results (v1)."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import pandas as pd

# 1. Try the parquet contents dataset (v2 - current leaderboard)
print("=== open-llm-leaderboard/contents (v2) ===")
try:
    path = snapshot_download("open-llm-leaderboard/contents", repo_type="dataset", 
                            local_dir="data/contents_v2", ignore_patterns=["*.md"])
    print(f"Downloaded to: {path}")
    # Find parquet files
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.parquet'):
                fp = os.path.join(root, f)
                df = pd.read_parquet(fp)
                print(f"\n  File: {f}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample:\n{df.head(2).to_string()}")
except Exception as e:
    print(f"Error: {e}")

# 2. Try old leaderboard results (v1)
print("\n\n=== open-llm-leaderboard-old/results (v1) ===")
try:
    api = HfApi()
    # Just list files first - this repo might be huge
    files = list(api.list_repo_tree("open-llm-leaderboard-old/results", repo_type="dataset"))
    print(f"Total items: {len(files)}")
    for f in files[:20]:
        print(f"  {f.path if hasattr(f, 'path') else f}")
except Exception as e:
    print(f"Error: {e}")
