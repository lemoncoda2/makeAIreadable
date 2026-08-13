# Decoupled Collaboration Training

Iterative **RL → separate → regenerate → DPO** experiment that keeps coding ability
while recovering human-facing collaboration readability, using Qwen3 thinking mode
to separate the work layer (`<think>` + code) from the collaboration layer.

Hardware target: **4× V100-32G** SSH server. GPU training runs on that remote machine;
local/cloud agents can use `--dry_run` / `--mock_judge` for offline pipeline checks.

**Fail-fast:** scripts refuse silent fallbacks (thinking mode, missing reward `test_cases`,
dry-run placeholder checkpoints, vLLM+LoRA adapter dirs, unparseable judge JSON, GRPO
batch/generations mismatch). Prefer a loud error over a fake success.

## Layout

```
decoupled_collab/
├── src/           # prepare / collect / regen / filter / train / evaluate / pipeline
├── configs/       # grpo / dpo / pipeline yaml
├── data/          # mbpp_*.jsonl, traces/, dpo_pairs/
├── checkpoints/   # cycle_N/model_rl, model_rl_dpo
├── scripts/       # setup_env, download_assets, smoke_test
└── GOAL.md        # full experimental protocol
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek OpenAI-compatible API for readability judging |
| `WANDB_API_KEY` | Weights & Biases logging for GRPO/DPO |
| `HF_HOME` | Hugging Face cache (models/datasets) |
| `DEEPSEEK_BASE_URL` | Optional; default `https://api.deepseek.com` |

```bash
export DEEPSEEK_API_KEY=sk-...
export WANDB_API_KEY=...
export HF_HOME=/path/to/hf_cache
```

## Phase 0 — Setup (V100×4)

```bash
cd decoupled_collab
bash scripts/setup_env.sh          # conda collab + pip (torch cu121 noted)
bash scripts/download_assets.sh    # Qwen3-4B + prepare_data --download
# or manually:
python src/prepare_data.py --download
bash scripts/smoke_test.sh         # pytest + dry_run phases (no GPU)
```

Verify GPUs: `python -c "import torch; print(torch.cuda.device_count())"` → `4`.

## Phase 1 — GRPO (work layer)

```bash
accelerate launch --num_processes 4 src/train_grpo.py --config configs/grpo_config.yaml
# Hypothesis check (base vs rl):
python src/evaluate.py --mode hypothesis_check \
  --base_model ./models/Qwen3-4B \
  --rl_model ./checkpoints/cycle_0/model_rl \
  --eval_data ./data/mbpp_plus_test.jsonl \
  --output ./results/phase1_hypothesis.json
```

## Phase 2 — Collect traces

```bash
python src/collect_traces.py \
  --model ./checkpoints/cycle_0/model_rl \
  --base_model_path ./models/Qwen3-4B \
  --tasks ./data/mbpp_train.jsonl \
  --output ./data/traces/cycle_0_rl_traces.jsonl \
  --num_tasks 2000 --temperature 0.7 --max_new_tokens 768 \
  --enable_thinking true --use_vllm true
# Offline: add --dry_run
```

## Phase 3 — Regen + filter + DPO

```bash
python src/regen_collaboration.py \
  --base_model ./models/Qwen3-4B \
  --traces ./data/traces/cycle_0_rl_traces.jsonl \
  --output ./data/dpo_pairs/cycle_0_raw_pairs.jsonl \
  --num_samples 3000 --use_vllm true

python src/filter_pairs.py \
  --raw_pairs ./data/dpo_pairs/cycle_0_raw_pairs.jsonl \
  --output ./data/dpo_pairs/cycle_0_filtered_pairs.jsonl \
  --judge_api deepseek --threshold 6.0 --batch_size 20 --max_concurrent 5

python src/train_dpo.py --config configs/dpo_config.yaml
```

## Phase 4 — Full evaluation

```bash
python src/evaluate.py --mode full \
  --models base,rl,final \
  --base_model ./models/Qwen3-4B \
  --rl_model ./checkpoints/cycle_0/model_rl \
  --final_model ./checkpoints/cycle_0/model_rl_dpo \
  --eval_data ./data/mbpp_plus_test.jsonl \
  --lcb_data ./data/lcb_easy.jsonl \
  --num_tasks_benchmark 200 --num_tasks_readability 50 \
  --output ./results/cycle_0_full_eval.json
```

Modes: `full` | `hypothesis_check` | `benchmark` | `readability`. Use `--dry_run` / `--mock_judge` offline.

## Phase 5 — Iterate via master pipeline

```bash
# Full multi-cycle run on the GPU server
python src/run_pipeline.py --config configs/pipeline_config.yaml

# Resume after interrupt (loads pipeline_state.json)
python src/run_pipeline.py --config configs/pipeline_config.yaml --resume

# Start cycle 1 from previous final model
python src/run_pipeline.py --config configs/pipeline_config.yaml \
  --cycle_id 1 --start_model ./checkpoints/cycle_0/model_rl_dpo

# Offline orchestration smoke (no GPU)
python src/run_pipeline.py --config configs/pipeline_config.yaml --dry_run --cycle_id 0
```

Pipeline state is stored in `pipeline_state.json` under the project root. Phases:
`phase1_grpo` → `phase1_eval` → `phase2_collect` → `phase3_regen` → `phase3_filter` → `phase3_dpo` → `phase4_eval`.

## Notes

- **GPU training** (`train_grpo.py` / `train_dpo.py`) is expected on the V100 server; this repo’s orchestration + data scripts support `--dry_run` for CI/agent validation without GPUs.
- LiveCodeBench easy may ship as an empty placeholder until downloaded manually — see `prepare_lcb_easy()` warnings.
- See `GOAL.md` for success criteria, timelines, and failure recovery.
