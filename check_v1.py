"""Download the v1 (old) leaderboard results and check for aggregated scores."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

api = HfApi()

# The old leaderboard has per-model directories. Let's check if there's any aggregated file
print("=== Checking open-llm-leaderboard-old/results ===")
try:
    files = list(api.list_repo_tree("open-llm-leaderboard-old/results", repo_type="dataset", path_in_repo=""))
    dirs = [f for f in files if hasattr(f, 'tree_id')]  # folders
    regular = [f for f in files if not hasattr(f, 'tree_id')]  # files
    print(f"Top-level dirs: {len(dirs)}, files: {len(regular)}")
    for f in regular[:10]:
        print(f"  FILE: {f.path if hasattr(f, 'path') else f}")
except Exception as e:
    print(f"Error: {e}")

# Also check for pre-compiled datasets
print("\n=== Searching for compiled v1 leaderboard data ===")
datasets = list(api.list_datasets(search="open llm leaderboard results", limit=15))
for ds in datasets[:15]:
    print(f"  {ds.id}")

# Try known compiled datasets
compiled_repos = [
    "Vipitis/open_llm_leaderboard_submissions",
    "lvwerra/open-llm-leaderboard-results",
    "Chenda/open-llm-leaderboard",
    "ArtificialAnalysis/OPEN_LLM_LEADERBOARD",
]
for repo in compiled_repos:
    try:
        info = api.dataset_info(repo)
        print(f"\n[OK] {repo} - modified: {info.lastModified}")
    except:
        pass

# Also try the HF leaderboard v1 snapshot
print("\n=== Trying open-llm-leaderboard/results ===")
try:
    files = list(api.list_repo_tree("open-llm-leaderboard/results", repo_type="dataset", path_in_repo=""))
    dirs = [f for f in files if hasattr(f, 'tree_id')]
    regular = [f for f in files if not hasattr(f, 'tree_id')]
    print(f"Top-level dirs: {len(dirs)}, files: {len(regular)}")
    for f in regular[:5]:
        print(f"  FILE: {f.path if hasattr(f, 'path') else f}")
    for d in dirs[:10]:
        print(f"  DIR: {d.path if hasattr(d, 'path') else d}")
except Exception as e:
    print(f"Error: {e}")
