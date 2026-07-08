#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" || "$HF_TOKEN" == "your_huggingface_token_here" ]]; then
  echo "Set HF_TOKEN in .env (copy from .env.example and paste your Hugging Face token)." >&2
  exit 1
fi

python3 -c "
import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id='gabegaines24/ride-share-surge-forcasting',
    repo_type='space',
    token=os.environ['HF_TOKEN'],
    ignore_patterns=[
        '.venv/*', 'taxi data/*', 'processed_data/*', '__pycache__/*',
        '*.pyc', '.git/*', 'node_modules/*', '.env',
    ],
)
print('Deployed to HuggingFace!')
"
