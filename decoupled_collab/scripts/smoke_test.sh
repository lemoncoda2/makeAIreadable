#!/usr/bin/env bash
# Offline smoke test: pytest + dry_run pipeline phases (no GPU required)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "==> Writing tiny synthetic MBPP fixtures"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path(".")
(root / "data").mkdir(parents=True, exist_ok=True)
train = [
    {
        "task_id": "mbpp_1",
        "prompt": "Write a function to add two numbers.",
        "test_cases": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        "code_solution": "def add(a, b):\n    return a + b\n",
    },
    {
        "task_id": "mbpp_2",
        "prompt": "Write a function to return the maximum of two numbers.",
        "test_cases": ["assert maximum(1, 2) == 2"],
        "code_solution": "def maximum(a, b):\n    return a if a > b else b\n",
    },
]
eval_tasks = [
    {
        "task_id": "mbpp_101",
        "prompt": "Write a function to check if a number is even.",
        "test_cases": ["assert is_even(2) == True", "assert is_even(3) == False"],
        "entry_point": "is_even",
        "code_solution": "def is_even(n):\n    return n % 2 == 0\n",
    }
]
with open("data/mbpp_train.jsonl", "w") as f:
    for t in train:
        f.write(json.dumps(t) + "\n")
with open("data/mbpp_plus_test.jsonl", "w") as f:
    for t in eval_tasks:
        f.write(json.dumps(t) + "\n")
Path("data/lcb_easy.jsonl").write_text("")
print("fixtures ok")
PY

echo "==> pytest"
"$PYTHON" -m pytest -q tests/

# Fresh smoke outputs
rm -f ./data/traces/smoke_traces.jsonl ./data/dpo_pairs/smoke_raw.jsonl ./data/dpo_pairs/smoke_filtered.jsonl

echo "==> dry_run collect → regen → filter → evaluate"
"$PYTHON" src/collect_traces.py \
  --model ./models/Qwen3-4B \
  --base_model_path ./models/Qwen3-4B \
  --tasks ./data/mbpp_train.jsonl \
  --output ./data/traces/smoke_traces.jsonl \
  --num_tasks 2 \
  --dry_run \
  --use_vllm false

"$PYTHON" src/regen_collaboration.py \
  --base_model ./models/Qwen3-4B \
  --traces ./data/traces/smoke_traces.jsonl \
  --output ./data/dpo_pairs/smoke_raw.jsonl \
  --num_samples 2 \
  --dry_run \
  --use_vllm false

"$PYTHON" src/filter_pairs.py \
  --raw_pairs ./data/dpo_pairs/smoke_raw.jsonl \
  --output ./data/dpo_pairs/smoke_filtered.jsonl \
  --mock_judge \
  --threshold 6.0

"$PYTHON" src/evaluate.py \
  --mode full \
  --models base,rl,final \
  --base_model ./models/Qwen3-4B \
  --rl_model ./checkpoints/cycle_0/model_rl \
  --final_model ./checkpoints/cycle_0/model_rl_dpo \
  --eval_data ./data/mbpp_plus_test.jsonl \
  --num_tasks_benchmark 1 \
  --num_tasks_readability 1 \
  --dry_run \
  --mock_judge \
  --output ./results/smoke_eval.json

echo "==> dry_run pipeline single phases"
"$PYTHON" src/run_pipeline.py \
  --config configs/pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase1_grpo

"$PYTHON" src/run_pipeline.py \
  --config configs/pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase2_collect

"$PYTHON" src/run_pipeline.py \
  --config configs/pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase3_regen

"$PYTHON" src/run_pipeline.py \
  --config configs/pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase3_filter

echo "✓ smoke_test.sh finished"
