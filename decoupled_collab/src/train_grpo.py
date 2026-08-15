#!/usr/bin/env python3
"""
GRPO training for decoupled_collab (Qwen3-4B + LoRA + TRL).

Target: 4×V100-32G via `accelerate launch --num_processes 4`.

Reward is code-execution only (see utils.code_executor.compute_reward).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.code_executor import batch_compute_rewards  # noqa: E402
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


def get_compatible_grpo_trainer_class(base_trainer: type) -> type:
    """Bridge the TRL 0.15 sampler signature to Transformers 4.52.

    Transformers 4.52 calls ``_get_train_sampler(train_dataset)`` while TRL
    0.15.2 overrides it as ``_get_train_sampler()``.  Keep the workaround
    narrowly signature-gated so an unknown future API fails instead of being
    silently papered over.
    """
    method = base_trainer._get_train_sampler
    parameters = list(inspect.signature(method).parameters.values())
    names = [parameter.name for parameter in parameters]

    if names == ["self"]:
        class CompatibleGRPOTrainer(base_trainer):
            def _get_train_sampler(self, train_dataset=None):
                if train_dataset is None or train_dataset is self.train_dataset:
                    return super()._get_train_sampler()

                # The legacy TRL implementation reads self.train_dataset.
                # Temporarily point it at the dataset prepared by Transformers
                # so its RepeatRandomSampler is built for the correct rows.
                original_dataset = self.train_dataset
                self.train_dataset = train_dataset
                try:
                    return super()._get_train_sampler()
                finally:
                    self.train_dataset = original_dataset

        CompatibleGRPOTrainer.__name__ = f"Compatible{base_trainer.__name__}"
        return CompatibleGRPOTrainer

    if names[:2] == ["self", "train_dataset"]:
        return base_trainer

    raise RuntimeError(
        "Unsupported GRPOTrainer._get_train_sampler signature: "
        f"{inspect.signature(method)}"
    )


def get_thinking_budget_grpo_trainer_class(
    base_trainer: type,
    thinking_budget_tokens: int,
    stop_after_code_fence: bool = False,
) -> type:
    """Add forced Qwen think-close generation and mask its control token."""
    if thinking_budget_tokens <= 0:
        raise ValueError("thinking_budget_tokens must be positive")

    class ThinkingBudgetGRPOTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from utils.thinking_budget import install_thinking_budget_generate

            self._decoupled_think_end_token_id = install_thinking_budget_generate(
                self.model,
                self.processing_class,
                thinking_budget_tokens=thinking_budget_tokens,
                stop_after_code_fence=stop_after_code_fence,
            )

        def _prepare_inputs(self, inputs):
            prepared = super()._prepare_inputs(inputs)
            completion_ids = prepared["completion_ids"]
            completion_mask = prepared["completion_mask"]
            if stop_after_code_fence:
                from utils.thinking_budget import mask_completion_after_code_fence

                completion_mask = mask_completion_after_code_fence(
                    completion_ids,
                    completion_mask,
                    self.processing_class,
                )
                prepared["completion_mask"] = completion_mask
            loss_mask = completion_mask.clone()
            forced_index = thinking_budget_tokens
            if completion_ids.shape[1] > forced_index:
                forced_close = completion_ids[:, forced_index].eq(
                    self._decoupled_think_end_token_id
                )
                if bool(forced_close.any()):
                    loss_mask[forced_close, forced_index] = 0
            # Do not alter completion_mask: TRL also uses it as the attention
            # mask. The generated code must still attend to the forced </think>.
            prepared["decoupled_loss_mask"] = loss_mask
            return prepared

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            if return_outputs:
                raise ValueError(
                    "The GRPOTrainer does not support returning outputs"
                )

            import torch

            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            loss_mask = inputs["decoupled_loss_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            per_token_logps = self._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep
            )

            loss, per_token_kl = compute_masked_grpo_objective(
                per_token_logps=per_token_logps,
                ref_per_token_logps=inputs["ref_per_token_logps"],
                advantages=inputs["advantages"],
                loss_mask=loss_mask,
                beta=self.beta,
            )

            completion_length = (
                self.accelerator.gather_for_metrics(completion_mask.sum(1))
                .float()
                .mean()
                .item()
            )
            self._metrics["completion_length"].append(completion_length)
            mean_kl = (
                (per_token_kl * loss_mask).sum(dim=1)
                / loss_mask.sum(dim=1).clamp_min(1)
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
            return loss

    ThinkingBudgetGRPOTrainer.__name__ = (
        f"ThinkingBudget{base_trainer.__name__}"
    )
    return ThinkingBudgetGRPOTrainer


def compute_masked_grpo_objective(
    *, per_token_logps, ref_per_token_logps, advantages, loss_mask, beta
):
    """TRL 0.15 GRPO objective with a separate mask for forced tokens."""
    import torch

    active = loss_mask.bool()
    # Compute KL in FP32 and zero inactive log-ratios before exp. Multiplying
    # an already-overflowed value by zero would still produce NaN.
    log_ratio = (ref_per_token_logps - per_token_logps).float()
    log_ratio = torch.where(active, log_ratio, torch.zeros_like(log_ratio))
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1

    policy_ratio = torch.exp(
        per_token_logps.float() - per_token_logps.float().detach()
    )
    per_token_loss = -(
        policy_ratio * advantages.float().unsqueeze(1) - float(beta) * per_token_kl
    )
    denominator = loss_mask.sum(dim=1).clamp_min(1)
    loss = ((per_token_loss * loss_mask).sum(dim=1) / denominator).mean()
    return loss, per_token_kl


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
    p.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume from an explicit Trainer checkpoint directory, or use without "
            "a value to select the highest valid checkpoint-N under output_dir"
        ),
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

            messages = build_coding_messages(task_prompt, test_cases[:1])
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


def build_reward_func(
    timeout: int = 10,
    max_test_cases: int = 5,
    audit_dir: Optional[Path] = None,
):
    """
    TRL GRPO reward callable.

    Signature compatible with TRL >= 0.12:
      reward_func(completions, **kwargs) -> list[float]
    Extra dataset columns (e.g. test_cases) are passed via kwargs.
    """

    audit_call = 0

    def reward_func(completions=None, test_cases=None, **kwargs):
        nonlocal audit_call
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
        task_ids = list(kwargs.get("task_id") or [None] * len(tcs_batch))
        if len(tcs_batch) != n:
            # TRL GRPO expands each prompt into num_generations completions.
            # Common layout: completions grouped per prompt, so n = len(tcs) * G.
            if len(tcs_batch) > 0 and n % len(tcs_batch) == 0:
                repeat = n // len(tcs_batch)
                expanded: list = []
                for tc in tcs_batch:
                    expanded.extend([tc] * repeat)
                tcs_batch = expanded
                task_ids = [task_id for task_id in task_ids for _ in range(repeat)]
                print(
                    f"[info] Expanded test_cases x{repeat} to match "
                    f"{n} completions (GRPO num_generations alignment)"
                )
            else:
                raise RuntimeError(
                    f"GRPO reward length mismatch: len(completions)={n} but "
                    f"len(test_cases)={len(tcs_batch)}. Expected equal lengths "
                    "or completions divisible by test_cases (num_generations). "
                    "Refusing to invent alignment."
                )

        if len(task_ids) != n:
            raise RuntimeError(
                f"GRPO audit alignment mismatch: len(task_ids)={len(task_ids)} "
                f"but len(completions)={n}."
            )

        parsed_tcs: list = []
        for tcs in tcs_batch:
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
            parsed_tcs.append(list(tcs))

        rewards = [
            float(r)
            for r in batch_compute_rewards(
                texts,
                parsed_tcs,
                timeout=timeout,
                max_test_cases=max_test_cases,
            )
        ]

        if audit_dir is not None:
            audit_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            audit_path = audit_dir / f"reward_rank{rank}.jsonl"
            with audit_path.open("a", encoding="utf-8") as audit_file:
                for task_id, text, reward in zip(task_ids, texts, rewards):
                    audit_file.write(
                        json.dumps(
                            {
                                "call": audit_call,
                                "task_id": task_id,
                                "reward": reward,
                                "has_think_end": "</think>" in text,
                                "has_python_fence": "```python" in text.lower(),
                                "completion_chars": len(text),
                                "completion": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            audit_call += 1
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
        ddp_find_unused_parameters=False,
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

    from utils.failfast import ConfigError, resolve_trainer_resume_checkpoint

    resume_requested = args.resume_from_checkpoint
    if resume_requested is None:
        resume_requested = cfg.get("training", {}).get("resume_from_checkpoint")
    try:
        resume_checkpoint = resolve_trainer_resume_checkpoint(
            resume_requested, output_dir, root=ROOT
        )
    except ConfigError as e:
        print(f"[error] {e}")
        return 1
    if resume_checkpoint is not None:
        print(f"[info] Will resume Trainer state from {resume_checkpoint}")

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
        from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig  # noqa: F401
    except ImportError as e:
        print(
            "[error] Cannot import GRPOTrainer/GRPOConfig. "
            "Need trl>=0.12 (`pip install 'trl>=0.12'`)."
        )
        print(f"        Detail: {e}")
        return 1

    try:
        GRPOTrainer = get_compatible_grpo_trainer_class(TRLGRPOTrainer)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1

    thinking_budget_tokens = cfg.get("grpo", {}).get("thinking_budget_tokens")
    if thinking_budget_tokens is not None:
        thinking_budget_tokens = int(thinking_budget_tokens)
        max_completion_length = int(
            cfg.get("grpo", {}).get("max_new_tokens", 768)
        )
        if not bool(cfg.get("model", {}).get("enable_thinking", True)):
            print("[error] thinking_budget_tokens requires enable_thinking=True")
            return 1
        if not 0 < thinking_budget_tokens < max_completion_length:
            print(
                "[error] thinking_budget_tokens must be positive and smaller "
                f"than max_new_tokens; got {thinking_budget_tokens} and "
                f"{max_completion_length}"
            )
            return 1
        GRPOTrainer = get_thinking_budget_grpo_trainer_class(
            GRPOTrainer,
            thinking_budget_tokens,
            stop_after_code_fence=bool(
                cfg.get("grpo", {}).get("stop_after_code_fence", False)
            ),
        )

    if not train_file.exists():
        print(f"[error] Training data not found: {train_file}")
        return 1

    from utils.benchmarks import require_real_benchmark
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
        audit_dir=output_dir / "reward_audit",
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
    trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        print(f"[info] Saving LoRA adapter to {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        required_adapter_files = (
            output_dir / "adapter_config.json",
            output_dir / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required_adapter_files if not path.is_file()]
        if missing:
            raise RuntimeError(
                "GRPO completed but the LoRA adapter save is incomplete: "
                + ", ".join(missing)
            )

    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()
    if trainer.accelerator.is_main_process:
        print("[info] GRPO training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
