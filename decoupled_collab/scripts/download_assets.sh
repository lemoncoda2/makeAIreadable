#!/usr/bin/env bash
# Download Qwen3-4B + prepare MBPP datasets (GOAL Step 0.3 / 0.4)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p models data/raw

echo "==> Downloading Qwen/Qwen3-4B → ./models/Qwen3-4B"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Qwen/Qwen3-4B --local-dir ./models/Qwen3-4B
else
  python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir="./models/Qwen3-4B")
print("Downloaded via huggingface_hub.snapshot_download")
PY
fi

echo "==> Preparing MBPP via datasets.load_dataset (--download)"
python3 src/prepare_data.py --download

echo "✓ download_assets.sh finished"
echo "Note: LiveCodeBench easy may be empty/placeholder; see prepare_data.prepare_lcb_easy docs."
