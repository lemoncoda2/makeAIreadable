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
- Qwen3 thinking mode needs `enable_thinking=True` in `apply_chat_template`; if the installed transformers build rejects the kwarg, scripts fall back and log a warning.
- GRPO reward is **code-execution only** (`utils.code_executor.compute_reward`); never mix readability into the RL reward.
- Use `--dry_run` / `--mock_judge` on machines without GPU or DeepSeek credentials.
- V100 is compute capability 7.0: stick to the GOAL pin (`torch` cu121, `vllm==0.6.4`) on the server; newer vLLM may drop V100.
