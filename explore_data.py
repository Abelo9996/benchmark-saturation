"""Explore HuggingFace Open LLM Leaderboard data availability."""
import json
from huggingface_hub import HfApi, list_datasets

api = HfApi()

# Check known leaderboard dataset repos
repos = [
    "open-llm-leaderboard/results",
    "open-llm-leaderboard/open_llm_leaderboard",
    "open-llm-leaderboard/contents",
    "lmsys/chatbot_arena_leaderboard",
    "lmsys/lmsys-arena-human-preference-55k",
]

for repo in repos:
    try:
        info = api.dataset_info(repo)
        print(f"\n[OK] {repo}")
        print(f"   Last modified: {info.lastModified}")
        print(f"   Tags: {info.tags[:5] if info.tags else 'none'}")
        # List top-level files
        files = api.list_repo_tree(repo, repo_type="dataset")
        file_list = []
        for f in files:
            file_list.append(f.rpath)
            if len(file_list) >= 15:
                break
        print(f"   Files (first 15): {file_list}")
    except Exception as e:
        print(f"\n[MISS] {repo}: {e}")

# Also search for leaderboard-related datasets
print("\n\n--- Searching for 'open llm leaderboard' datasets ---")
datasets = list(api.list_datasets(search="open-llm-leaderboard", limit=10))
for ds in datasets:
    print(f"  {ds.id} (modified: {ds.lastModified})")

print("\n--- Searching for 'chatbot arena' datasets ---")
datasets = list(api.list_datasets(search="chatbot arena", limit=10))
for ds in datasets:
    print(f"  {ds.id} (modified: {ds.lastModified})")
