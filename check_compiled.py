"""Check if there's a compiled/contents version of v1, and also check Kaggle."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from huggingface_hub import HfApi

api = HfApi()

# Check for old contents dataset
repos_to_check = [
    "open-llm-leaderboard-old/contents",
    "albertvillanova/tmp-open-llm-leaderboard-results",
    "albertvillanova/tmp-open-llm-leaderboard-results-2",
]

for repo in repos_to_check:
    try:
        info = api.dataset_info(repo)
        print(f"[OK] {repo} - modified: {info.lastModified}")
        files = list(api.list_repo_tree(repo, repo_type="dataset"))
        for f in files[:10]:
            p = f.path if hasattr(f, 'path') else str(f)
            s = f.size if hasattr(f, 'size') else '?'
            print(f"  {p} ({s} bytes)")
    except Exception as e:
        print(f"[MISS] {repo}: {str(e)[:100]}")
