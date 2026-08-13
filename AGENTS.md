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
| List benchmarks | `python src/prepare_data.py --list-benchmarks` |
| Prep **real** data | `python src/prepare_data.py --download` (EvalPlus MBPP+ + MBPP full + LCB-easy) |
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
- Qwen3 thinking mode requires `enable_thinking=True` and **`transformers>=4.51`** (official; older → `KeyError: qwen3`). Regen collaboration uses `enable_thinking=False` (must pass the kwarg; Qwen3 defaults thinking-ON if omitted).
- Recommended V100 stack (GOAL Step 0.2): `torch 2.5.1+cu121` (or `2.6.0+cu124`), `transformers>=4.51,<4.53`, optional **`vllm==0.8.5`** + `VLLM_USE_V1=0`. Do not casually use vLLM≥0.9 prebuilt on sm_70.
- FP16 only on V100 (`bf16=false`). Training path is HF+PEFT; vLLM is optional for collect/regen after merge (`src/merge_lora.py`; pipeline auto-merges when `inference.use_vllm: true`).
- GRPO reward is **code-execution only**; missing `test_cases` in the reward kwargs aborts training. TRL expands `test_cases` × `num_generations` when lengths differ by that factor.
- DPO resumes the RL LoRA only (refuses fresh LoRA on base). Reference is PEFT-implicit (`ref_model=None`) to avoid 2× weights OOM on 32G V100.
- `pipeline_state.json`: `current_phase` means **next phase to run**. `--resume` after `status=completed` is refused.
- `bash scripts/smoke_test.sh` writes only under `data/smoke/` — it must never overwrite real `data/mbpp_*.jsonl` / `data/lcb_easy.jsonl`.
- Empty `test_cases` never count as pass@1. LCB uses `harness=lcb` + `lcb_tests` (stdin/call) via `utils/lcb_executor.py` (official-style); not MBPP asserts.
- DPO prompts are re-rendered in `train_dpo.py` with the same Qwen chat template as `regen_collaboration` (`enable_thinking=False`). Fake `<system>/<user>` XML is refused.
- `--dry_run` and `--mock_judge` are explicit only. dry-run eval of readability requires `--mock_judge`. Placeholder dirs contain `DRY_RUN_PLACEHOLDER` and are refused by real loads.
- `inference.use_vllm` defaults to `false` (HF+PEFT). vLLM + LoRA adapter path fails fast unless merged.
