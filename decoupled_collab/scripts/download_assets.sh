#!/usr/bin/env bash
# Download Qwen3-4B + REAL benchmarks (MBPP full, MBPP+/EvalPlus, LCB-easy)
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

echo "==> Preparing REAL datasets (MBPP full + EvalPlus MBPP+ + LiveCodeBench-easy)"
echo "    This REQUIRES network. Synthetic/example jsonl are NOT enough for training/eval."
python3 src/prepare_data.py --download --list-benchmarks
python3 src/prepare_data.py --download

echo "==> Verifying dataset gates"
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "src")
from utils.benchmarks import require_real_benchmarks, list_benchmarks
require_real_benchmarks(Path("."), ["mbpp_train", "mbpp_plus", "lcb_easy"])
print(list_benchmarks())
print("✓ Real benchmarks present")
PY

echo "✓ download_assets.sh finished"
