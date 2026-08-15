#!/usr/bin/env bash
# Offline smoke test: pytest + dry_run pipeline phases (no GPU required).
# Writes ONLY under data/smoke/ — never clobbers real benchmark jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

SMOKE_DIR=./data/smoke
mkdir -p "$SMOKE_DIR" ./data/traces ./data/dpo_pairs ./results

echo "==> Writing tiny SYNTHETIC fixtures under $SMOKE_DIR (never touches data/mbpp_*.jsonl)"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

smoke = Path("data/smoke")
smoke.mkdir(parents=True, exist_ok=True)
train = [
    {
        "task_id": "mbpp_1",
        "prompt": "Write a function to add two numbers.",
        "test_cases": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        "code_solution": "def add(a, b):\n    return a + b\n",
        "source": "mbpp_full",
        "benchmark": "mbpp_train",
        "synthetic": True,
    },
    {
        "task_id": "mbpp_2",
        "prompt": "Write a function to return the maximum of two numbers.",
        "test_cases": ["assert maximum(1, 2) == 2"],
        "code_solution": "def maximum(a, b):\n    return a if a > b else b\n",
        "source": "mbpp_full",
        "benchmark": "mbpp_train",
        "synthetic": True,
    },
]
eval_tasks = [
    {
        "task_id": "Mbpp/101",
        "prompt": "Write a function to check if a number is even.",
        "test_cases": ["assert is_even(2) == True", "assert is_even(3) == False"],
        "entry_point": "is_even",
        "code_solution": "def is_even(n):\n    return n % 2 == 0\n",
        "source": "evalplus_mbpp_plus",
        "benchmark": "mbpp_plus",
        "synthetic": True,
    }
]
lcb = [
    {
        "task_id": "lcb_smoke_1",
        "prompt": "Read two integers and print their sum.",
        "harness": "lcb",
        "lcb_tests": [
            {"type": "stdin", "input": "1 2\n", "output": "3\n"},
            {"type": "stdin", "input": "0 0\n", "output": "0\n"},
        ],
        "test_cases": [],
        "code_solution": "a,b=map(int,input().split())\nprint(a+b)\n",
        "source": "livecodebench_easy",
        "benchmark": "lcb_easy",
        "synthetic": True,
        "difficulty": "easy",
    }
]
with open(smoke / "mbpp_train.jsonl", "w") as f:
    for t in train:
        f.write(json.dumps(t) + "\n")
with open(smoke / "mbpp_plus_test.jsonl", "w") as f:
    for t in eval_tasks:
        f.write(json.dumps(t) + "\n")
with open(smoke / "lcb_easy.jsonl", "w") as f:
    for t in lcb:
        f.write(json.dumps(t) + "\n")
print("synthetic fixtures ok under data/smoke/ (NOT for real training)")
PY

echo "==> pytest"
"$PYTHON" -m pytest -q tests/

# Fresh smoke outputs
rm -f ./data/traces/smoke_traces.jsonl ./data/dpo_pairs/smoke_raw.jsonl ./data/dpo_pairs/smoke_filtered.jsonl

echo "==> dry_run collect → regen → filter → evaluate (smoke paths only)"
"$PYTHON" src/collect_traces.py \
  --model ./models/Qwen3-4B \
  --base_model_path ./models/Qwen3-4B \
  --tasks ./data/smoke/mbpp_train.jsonl \
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
  --threshold 6.0 \
  --min_pairs 1

"$PYTHON" src/evaluate.py \
  --mode full \
  --models base,rl,final \
  --base_model ./models/Qwen3-4B \
  --rl_model ./checkpoints/cycle_0/model_rl \
  --final_model ./checkpoints/cycle_0/model_rl_dpo \
  --eval_data ./data/smoke/mbpp_plus_test.jsonl \
  --lcb_data ./data/smoke/lcb_easy.jsonl \
  --num_tasks_benchmark 1 \
  --num_tasks_readability 1 \
  --dry_run \
  --allow_synthetic \
  --mock_judge \
  --output ./results/smoke_eval.json

echo "==> dry_run pipeline single phases"
"$PYTHON" src/run_pipeline.py \
  --config configs/smoke_pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase1_grpo

"$PYTHON" src/run_pipeline.py \
  --config configs/smoke_pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase2_collect

"$PYTHON" src/run_pipeline.py \
  --config configs/smoke_pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase3_regen

"$PYTHON" src/run_pipeline.py \
  --config configs/smoke_pipeline_config.yaml \
  --cycle_id 0 \
  --dry_run \
  --only_phase phase3_filter

echo "✓ smoke_test.sh finished (real data/mbpp_*.jsonl untouched)"
