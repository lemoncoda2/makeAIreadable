#!/usr/bin/env python3
"""
GRPO training for decoupled_collab (Qwen3-4B + LoRA + TRL).

Target: 4×V100-32G via `accelerate launch --num_processes 4`.

Reward is code-execution only (see utils.code_executor.compute_reward).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.code_executor import compute_reward  # noqa: E402
from utils.model_utils import (  # noqa: E402
    apply_chat_template_with_thinking,
    build_lora_config,
    check_cuda_or_warn,
    find_base_model_path,
    load_causal_lm,
    load_tokenizer,
    load_yaml,
    maybe_merge_or_load_peft,
    resolve_path,
    setup_wandb_env,
)
from utils.prompts import build_coding_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training (TRL) for coding reward")
    p.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    p.add_argument("--model", type=str, default=None, help="Override model.name_or_path")
    p.add_argument("--output", type=str, default=None, help="Override output_dir")
    p.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Override data.max_tasks (limit training prompts)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config, print plan, exit without training",
    )
    return p.parse_args()


def _completion_to_text(completion: Any) -> str:
    """Normalize TRL completion (str or chat messages) to plain text."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # conversational: list[{"role","content"}]
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def load_grpo_dataset(
    train_file: Path,
    tokenizer,
    enable_thinking: bool = True,
    max_tasks: Optional[int] = None,
):
    """
    Load jsonl tasks -> HF Dataset with columns: prompt, test_cases, task_id.

    `prompt` is a chat-templated string with generation prompt (thinking enabled).
    """
    from datasets import Dataset

    rows: List[Dict[str, Any]] = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_prompt = item.get("prompt") or item.get("text") or ""
            test_cases = item.get("test_cases") or item.get("test_list") or []
            task_id = item.get("task_id", f"task_{len(rows)}")

            messages = build_coding_messages(task_prompt)
            rendered = apply_chat_template_with_thinking(
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
                tokenize=False,
            )
            rows.append(
                {
                    "prompt": rendered,
                    "test_cases": test_cases,
                    "task_id": task_id,
                    "raw_prompt": task_prompt,
                }
            )
            if max_tasks is not None and len(rows) >= max_tasks:
                break

    if not rows:
        raise ValueError(f"No training tasks found in {train_file}")

    return Dataset.from_list(rows)


def build_reward_func(timeout: int = 10, max_test_cases: int = 5):
    """
    TRL GRPO reward callable.

    Signature compatible with TRL >= 0.12:
      reward_func(completions, **kwargs) -> list[float]
    Extra dataset columns (e.g. test_cases) are passed via kwargs.
    """

    def reward_func(completions=None, test_cases=None, **kwargs):
        # Some TRL versions pass prompts as first positional; accept both.
        if completions is None:
            completions = kwargs.get("completions", [])
        if test_cases is None:
            test_cases = kwargs.get("test_cases", None)

        texts = [_completion_to_text(c) for c in completions]
        n = len(texts)

        if test_cases is None:
            raise RuntimeError(
                "GRPO reward_func did not receive dataset column 'test_cases'. "
                "Refusing to silently assign 0.0 rewards (that looks like training "
                "but learns nothing). Ensure remove_unused_columns=False and that "
                "the train dataset keeps a 'test_cases' field. kwargs keys: "
                f"{sorted(kwargs.keys())}"
            )
        tcs_batch = list(test_cases)
        if len(tcs_batch) != n:
            raise RuntimeError(
                f"GRPO reward length mismatch: len(completions)={n} but "
                f"len(test_cases)={len(tcs_batch)}. Refusing to pad/truncate "
                "silently (wrong rewards would be hard to debug)."
            )

        rewards = []
        for text, tcs in zip(texts, tcs_batch):
            if isinstance(tcs, str):
                try:
                    tcs = json.loads(tcs)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"test_cases entry is a non-JSON string: {tcs[:80]!r}"
                    ) from e
            if tcs is None:
                raise RuntimeError(
                    "Encountered test_cases=None for a completion. "
                    "Every GRPO row must include a list of assert tests."
                )
            rewards.append(
                float(
                    compute_reward(
                        text,
                        list(tcs),
                        timeout=timeout,
                        max_test_cases=max_test_cases,
                    )
                )
            )
        return rewards

    reward_func.__name__ = "code_execution_reward"
    return reward_func


def build_grpo_config(cfg: Dict[str, Any], output_dir: Path, report_to: str):
    """Map YAML -> TRL GRPOConfig (trl>=0.12)."""
    try:
        from trl import GRPOConfig
    except ImportError as e:
        raise ImportError(
            "TRL GRPOTrainer/GRPOConfig require trl>=0.12. "
            "Install with: pip install 'trl>=0.12'"
        ) from e

    grpo = cfg.get("grpo", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})

    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        num_train_epochs=float(training.get("num_epochs", 3)),
        per_device_train_batch_size=int(training.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 6)
        ),
        warmup_ratio=float(training.get("warmup_ratio", 0.05)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        fp16=bool(training.get("fp16", True)),
        bf16=False,
        logging_steps=int(logging_cfg.get("log_every_n_steps", 10)),
        save_steps=int(logging_cfg.get("save_every_n_steps", 200)),
        save_strategy="steps",
        report_to=report_to,
        remove_unused_columns=False,  # keep test_cases for reward_func
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        # GRPO-specific (names vary slightly across TRL versions)
        num_generations=int(grpo.get("num_samples_per_prompt", 6)),
        max_completion_length=int(grpo.get("max_new_tokens", 768)),
        temperature=float(grpo.get("temperature", 0.7)),
        beta=float(grpo.get("kl_coeff", 0.04)),
    )

    # Optional knobs if present in this TRL version
    optional = {
        "top_p": float(grpo.get("top_p", 0.9)),
        "epsilon": float(grpo.get("clip_range", 0.2)),
        "scale_rewards": bool(grpo.get("normalize_reward", True)),
        "run_name": logging_cfg.get("wandb_run"),
    }

    import dataclasses
    import inspect

    for k, v in optional.items():
        if v is not None:
            kwargs[k] = v

    # Prefer dataclass fields when available (TrainingArguments subclasses).
    if dataclasses.is_dataclass(GRPOConfig):
        accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    else:
        accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys()) - {
            "self"
        }

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    # Ensure required core keys always attempted even if introspection fails
    for key in ("output_dir", "learning_rate", "num_generations", "beta"):
        if key in kwargs and key not in filtered:
            filtered[key] = kwargs[key]

    try:
        return GRPOConfig(**filtered)
    except TypeError as e:
        raise TypeError(
            "Failed to construct trl.GRPOConfig — refusing to silently drop kwargs. "
            f"Accepted-by-introspection keys tried: {sorted(filtered)}. "
            "Align trl version with this script (trl>=0.12) or fix the YAML→TRL "
            f"field mapping. Underlying error: {e}"
        ) from e


def print_plan(cfg: Dict[str, Any], args: argparse.Namespace, paths: Dict[str, Path]):
    print("=" * 60)
    print("GRPO dry-run / training plan")
    print("=" * 60)
    print(f"  project_root : {ROOT}")
    print(f"  config       : {paths['config']}")
    print(f"  model        : {paths['model']}")
    print(f"  base_model   : {paths.get('base_model', paths['model'])}")
    print(f"  train_file   : {paths['train_file']}")
    print(f"  output_dir   : {paths['output']}")
    print(f"  max_tasks    : {cfg.get('data', {}).get('max_tasks')}")
    print(f"  LoRA         : rank={cfg.get('lora', {}).get('rank')}, "
          f"alpha={cfg.get('lora', {}).get('alpha')}")
    print(f"  grpo.samples : {cfg.get('grpo', {}).get('num_samples_per_prompt')}")
    print(f"  reward       : code_execution only "
          f"(timeout={cfg.get('reward', {}).get('timeout')}s, "
          f"max_tests={cfg.get('reward', {}).get('max_test_cases')})")
    print(f"  dry_run      : {args.dry_run}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)

    config_path = resolve_path(args.config, ROOT)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        return 1

    cfg = load_yaml(config_path)

    # CLI overrides
    if args.model:
        cfg.setdefault("model", {})["name_or_path"] = args.model
    if args.output:
        cfg["output_dir"] = args.output
    if args.max_tasks is not None:
        cfg.setdefault("data", {})["max_tasks"] = args.max_tasks

    model_path = resolve_path(cfg["model"]["name_or_path"], ROOT)
    output_dir = resolve_path(cfg.get("output_dir", "./checkpoints/cycle_0/model_rl"), ROOT)
    train_file = resolve_path(
        cfg.get("data", {}).get("train_file", "./data/mbpp_train.jsonl"), ROOT
    )
    try:
        base_model = find_base_model_path(
            model_path, default_base="./models/Qwen3-4B", root=ROOT
        )
    except FileNotFoundError as e:
        if args.dry_run:
            base_model = resolve_path("./models/Qwen3-4B", ROOT)
            print(f"[dry_run] {e} — using {base_model} in plan only.")
        else:
            print(f"[error] {e}")
            return 1

    paths = {
        "config": config_path,
        "model": model_path,
        "base_model": base_model,
        "train_file": train_file,
        "output": output_dir,
    }
    print_plan(cfg, args, paths)

    if args.dry_run:
        # Still validate known-broken GRPO math so dry-run surfaces config bugs.
        from utils.failfast import validate_grpo_batch_vs_generations

        try:
            nproc = int(
                os.environ.get("WORLD_SIZE")
                or os.environ.get("ACCELERATE_NUM_PROCESSES")
                or "4"
            )
            validate_grpo_batch_vs_generations(cfg, num_processes=nproc)
        except Exception as e:  # noqa: BLE001 — surface as dry-run warning exit
            print(f"[error] Config fail-fast check failed: {e}")
            return 1
        print("[dry_run] Exiting before model load / training.")
        return 0

    from utils.failfast import (
        assert_not_dry_run_placeholder,
        require_thinking_support,
        validate_grpo_batch_vs_generations,
    )

    try:
        check_cuda_or_warn(dry_run=False)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2

    assert_not_dry_run_placeholder(model_path, what="GRPO model/adapter")
    nproc = int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("ACCELERATE_NUM_PROCESSES")
        or "1"
    )
    try:
        validate_grpo_batch_vs_generations(cfg, num_processes=nproc)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}")
        return 1

    # Import TRL early with a clear message
    try:
        from trl import GRPOTrainer, GRPOConfig  # noqa: F401
    except ImportError as e:
        print(
            "[error] Cannot import GRPOTrainer/GRPOConfig. "
            "Need trl>=0.12 (`pip install 'trl>=0.12'`)."
        )
        print(f"        Detail: {e}")
        return 1

    if not train_file.exists():
        print(f"[error] Training data not found: {train_file}")
        return 1

    from utils.benchmarks import require_real_benchmark
    from utils.failfast import ConfigError

    try:
        require_real_benchmark("mbpp_train", train_file, allow_synthetic=False)
    except ConfigError as e:
        print(f"[error] {e}")
        return 1

    try:
        report_to = setup_wandb_env(cfg.get("logging"), dry_run=False)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}")
        return 1
    enable_thinking = bool(cfg.get("model", {}).get("enable_thinking", True))
    dtype = cfg.get("model", {}).get("torch_dtype", "float16")

    print(f"[info] Loading tokenizer from {base_model}")
    tokenizer = load_tokenizer(base_model)
    try:
        require_thinking_support(tokenizer, enable_thinking=enable_thinking)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1

    max_tasks = cfg.get("data", {}).get("max_tasks")
    dataset = load_grpo_dataset(
        train_file,
        tokenizer,
        enable_thinking=enable_thinking,
        max_tasks=max_tasks,
    )
    print(f"[info] Loaded {len(dataset)} GRPO tasks")

    print(f"[info] Loading model from {base_model} (dtype={dtype})")
    # Under accelerate, let the trainer/accelerator place the model; avoid device_map="auto"
    model = load_causal_lm(base_model, torch_dtype=dtype)

    lora_cfg = cfg.get("lora", {})
    peft_config = build_lora_config(
        rank=int(lora_cfg.get("rank", 32)),
        alpha=int(lora_cfg.get("alpha", 64)),
        target_modules=lora_cfg.get("target_modules"),
        dropout=float(lora_cfg.get("dropout", 0.05)),
    )

    # If model_path itself is an adapter checkpoint, resume weights
    adapter_cfg = model_path / "adapter_config.json"
    if adapter_cfg.exists() and model_path.resolve() != base_model.resolve():
        print(f"[info] Resuming LoRA weights from {model_path}")
        model = maybe_merge_or_load_peft(model, model_path, is_trainable=True)
        peft_config = None  # already wrapped

    reward_cfg = cfg.get("reward", {})
    reward_fn = build_reward_func(
        timeout=int(reward_cfg.get("timeout", 10)),
        max_test_cases=int(reward_cfg.get("max_test_cases", 5)),
    )

    grpo_args = build_grpo_config(cfg, output_dir, report_to=report_to)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    # Older TRL used `tokenizer=` instead of `processing_class=` — try once with
    # an explicit secondary call only when the error names that argument.
    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as e:
        msg = str(e)
        if "processing_class" in msg or "tokenizer" in msg:
            print(
                "[info] GRPOTrainer rejected processing_class=; retrying with "
                f"tokenizer= (trl API difference). Detail: {e}"
            )
            trainer_kwargs.pop("processing_class", None)
            trainer_kwargs["tokenizer"] = tokenizer
            trainer = GRPOTrainer(**trainer_kwargs)
        else:
            raise TypeError(
                "GRPOTrainer construction failed. Not retrying with different "
                f"kwargs (fail-fast). Detail: {e}"
            ) from e

    print("[info] Starting GRPO training...")
    trainer.train()

    print(f"[info] Saving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    # Ensure peft adapter files exist even if trainer saved full wrapper
    unwrapped = trainer.model
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(str(output_dir))

    print("[info] GRPO training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
