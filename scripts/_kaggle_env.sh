# Source this to set KAGGLE_API_TOKEN from ~/.kaggle/kaggle.json so the
# kaggle CLI uses Bearer auth (required for KGAT-format Google-account tokens).
export KAGGLE_API_TOKEN="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['key'])")"
