"""Check structure of one model result in v1 old leaderboard."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
from huggingface_hub import HfApi

api = HfApi()

# Check the structure of one model dir in old results
files = list(api.list_repo_tree("open-llm-leaderboard-old/results", repo_type="dataset", path_in_repo="01-ai"))
for f in files[:20]:
    p = f.path if hasattr(f, 'path') else str(f)
    print(p)

# Also check v2 results structure  
print("\n--- v2 results (open-llm-leaderboard/results) ---")
files = list(api.list_repo_tree("open-llm-leaderboard/results", repo_type="dataset", path_in_repo="01-ai"))
for f in files[:20]:
    p = f.path if hasattr(f, 'path') else str(f)
    print(p)

# Download one sample result file from old
from huggingface_hub import hf_hub_download
# Find a json file
print("\n--- Sample v1 result ---")
try:
    sub_files = list(api.list_repo_tree("open-llm-leaderboard-old/results", repo_type="dataset", path_in_repo="01-ai/Yi-34B"))
    for f in sub_files:
        p = f.path if hasattr(f, 'path') else str(f)
        print(f"  {p}")
        if p.endswith('.json'):
            local = hf_hub_download("open-llm-leaderboard-old/results", p, repo_type="dataset")
            with open(local) as fh:
                data = json.load(fh)
            print(f"  Keys: {list(data.keys())}")
            if 'results' in data:
                print(f"  Results keys: {list(data['results'].keys()) if isinstance(data['results'], dict) else 'list'}")
                if isinstance(data['results'], dict):
                    for k, v in list(data['results'].items())[:3]:
                        print(f"    {k}: {v}")
            if 'config' in data:
                print(f"  Config model: {data['config'].get('model_name', 'N/A')}")
                print(f"  Config date: {data['config'].get('submission_date', data['config'].get('start_time', 'N/A'))}")
            break
except Exception as e:
    print(f"Error: {e}")
