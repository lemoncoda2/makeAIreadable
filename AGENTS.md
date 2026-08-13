# AGENTS.md

## Cursor Cloud specific instructions

This repo implements the **Decoupled Collaboration** training pipeline (`decoupled_collab/`) described in `GOAL_decoupled_collaboration.md`.

### Hardware split

- **Cloud agent / laptop**: code authoring, unit/smoke tests, dry-run pipeline. No 4×V100 assumed.
- **Training server**: 4×V100-32G via SSH. Full GRPO/DPO/vLLM runs happen there after `git pull`.

Do not treat missing CUDA in the cloud VM as a product failure. Prefer:

```bash
cd decoupled_collab
pytest tests/ -q
bash scripts/smoke_test.sh
```

### Standard commands

See `decoupled_collab/README.md` and `decoupled_collab/GOAL.md` for Phase 0–5. Key entrypoints:

| Task | Command |
|------|---------|
| Smoke (no GPU) | `bash scripts/smoke_test.sh` |
| Prep data | `python src/prepare_data.py --download` |
| GRPO (4 GPU) | `accelerate launch --num_processes 4 src/train_grpo.py --config configs/grpo_config.yaml` |
| Full cycles | `python src/run_pipeline.py --config configs/pipeline_config.yaml` |
| Resume | `python src/run_pipeline.py --config configs/pipeline_config.yaml --resume` |

### Env vars (remote training)

- `DEEPSEEK_API_KEY` — readability judge / pair filter (OpenAI-compatible)
- `DEEPSEEK_BASE_URL` — default `https://api.deepseek.com`
- `DEEPSEEK_MODEL` — default `deepseek-chat` (map to deepseek-v4-flash if your gateway uses that id)
- `WANDB_API_KEY` / `WANDB_PROJECT` — optional logging
- `HF_HOME` / `HF_TOKEN` — model download cache on the GPU box

### Gotchas

- Run all training scripts with cwd = `decoupled_collab/` so relative paths in YAML resolve.
- **Fail-fast policy**: no silent fallbacks for thinking mode, missing `test_cases`, wrong PEFT args, dry-run placeholder checkpoints, vLLM+LoRA adapter dirs, or unparseable judge JSON. Fix the cause; do not “soft continue”.
- Qwen3 thinking mode requires `enable_thinking=True`; unsupported transformers → hard error (GOAL pin 4.45 may be too old).
- GRPO reward is **code-execution only**; missing `test_cases` in the reward kwargs aborts training.
- `--dry_run` and `--mock_judge` are explicit only. dry-run eval of readability requires `--mock_judge`. Placeholder dirs contain `DRY_RUN_PLACEHOLDER` and are refused by real loads.
- `inference.use_vllm` defaults to `false` (HF+PEFT). vLLM + LoRA adapter path fails fast.
- V100 is sm_70: validate torch/vLLM compatibility on the box; do not assume pins work blindly.
