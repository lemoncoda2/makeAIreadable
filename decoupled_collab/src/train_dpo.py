#!/usr/bin/env python3
"""
DPO training for decoupled_collab (Qwen3-4B + LoRA + TRL).

Continues from the GRPO (Model_RL) LoRA adapter. Reference model = Model_RL
(reference_free: false). Target: 4×V100-32G, fp16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.model_utils import (  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO training (TRL) on preference pairs")
    p.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="Path to YAML config",
    )
    p.add_argument("--model", type=str, default=None, help="Override model.name_or_path")
    p.add_argument(
        "--dpo_data",
        type=str,
        default=None,
        help="Override data.train_file (preference jsonl)",
    )
    p.add_argument("--output", type=str, default=None, help="Override output_dir")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config, print plan, exit without training",
    )
    return p.parse_args()


def load_preference_dataset(train_file: Path, eval_split: float = 0.05):
    """
    Load jsonl with fields: prompt, chosen, rejected.
    Returns (train_dataset, eval_dataset|None).
    """
    from datasets import Dataset

    rows = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            chosen = item.get("chosen")
            rejected = item.get("rejected")
            if prompt is None or chosen is None or rejected is None:
                raise ValueError(
                    f"DPO jsonl rows must have prompt/chosen/rejected; got keys={list(item.keys())}"
                )
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    if not rows:
        raise ValueError(f"No DPO pairs found in {train_file}")

    ds = Dataset.from_list(rows)
    eval_ds = None
    if eval_split and 0.0 < float(eval_split) < 1.0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=float(eval_split), seed=42)
        return split["train"], split["test"]
    return ds, eval_ds


def build_dpo_config(cfg: Dict[str, Any], output_dir: Path, report_to: str):
    """Map YAML -> TRL DPOConfig (trl>=0.12) with V100-friendly fp16 settings."""
    try:
        from trl import DPOConfig
    except ImportError:
        # Older TRL folded DPO args into TrainingArguments / DPOTrainer kwargs
        DPOConfig = None  # type: ignore

    dpo = cfg.get("dpo", {})
    training = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})

    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_train_epochs=float(training.get("num_epochs", 3)),
        per_device_train_batch_size=int(training.get("per_device_batch_size", 2)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 4)
        ),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        fp16=bool(training.get("fp16", True)),
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to=report_to,
        remove_unused_columns=False,
        beta=float(dpo.get("beta", 0.1)),
        loss_type=str(dpo.get("loss_type", "sigmoid")),
        label_smoothing=float(dpo.get("label_smoothing", 0.0)),
        max_length=int(training.get("max_length", 1536)),
        max_prompt_length=int(training.get("max_prompt_length", 1024)),
    )
    if logging_cfg.get("wandb_run"):
        kwargs["run_name"] = logging_cfg["wandb_run"]

    if DPOConfig is None:
        return kwargs

    try:
        return DPOConfig(**kwargs)
    except TypeError:
        # Drop keys not accepted by this version
        import dataclasses
        import inspect

        if dataclasses.is_dataclass(DPOConfig):
            fields = {f.name for f in dataclasses.fields(DPOConfig)}
        else:
            fields = set(inspect.signature(DPOConfig.__init__).parameters) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in fields}
        return DPOConfig(**filtered)


def print_plan(cfg: Dict[str, Any], args: argparse.Namespace, paths: Dict[str, Path]):
    print("=" * 60)
    print("DPO dry-run / training plan")
    print("=" * 60)
    print(f"  project_root : {ROOT}")
    print(f"  config       : {paths['config']}")
    print(f"  model/RL     : {paths['model']}")
    print(f"  base_model   : {paths['base_model']}")
    print(f"  resume_lora  : {paths.get('resume_from')}")
    print(f"  dpo_data     : {paths['dpo_data']}")
    print(f"  output_dir   : {paths['output']}")
    print(f"  dpo.beta     : {cfg.get('dpo', {}).get('beta')}")
    print(f"  loss_type    : {cfg.get('dpo', {}).get('loss_type')}")
    print(f"  reference_free: {cfg.get('dpo', {}).get('reference_free')}")
    print(f"  dry_run      : {args.dry_run}")
    print("=" * 60)


def _freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)

    config_path = resolve_path(args.config, ROOT)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        return 1

    cfg = load_yaml(config_path)

    if args.model:
        cfg.setdefault("model", {})["name_or_path"] = args.model
        # CLI --model must win over yaml lora.resume_from (multi-cycle correctness).
        cfg.setdefault("lora", {})["resume_from"] = args.model
    if args.output:
        cfg["output_dir"] = args.output
    if args.dpo_data:
        cfg.setdefault("data", {})["train_file"] = args.dpo_data

    model_path = resolve_path(cfg["model"]["name_or_path"], ROOT)
    output_dir = resolve_path(
        cfg.get("output_dir", "./checkpoints/cycle_0/model_rl_dpo"), ROOT
    )
    dpo_data = resolve_path(
        cfg.get("data", {}).get(
            "train_file", "./data/dpo_pairs/cycle_0_filtered.jsonl"
        ),
        ROOT,
    )

    lora_cfg = cfg.get("lora", {})
    resume_from_raw = lora_cfg.get("resume_from") or str(model_path)
    resume_from = resolve_path(resume_from_raw, ROOT)
    try:
        base_model = find_base_model_path(
            model_path, default_base="./models/Qwen3-4B", root=ROOT
        )
        if (resume_from / "adapter_config.json").exists():
            base_model = find_base_model_path(
                resume_from, default_base=base_model, root=ROOT
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
        "resume_from": resume_from,
        "dpo_data": dpo_data,
        "output": output_dir,
    }
    print_plan(cfg, args, paths)

    if args.dry_run:
        print("[dry_run] Exiting before model load / training.")
        return 0

    from utils.failfast import assert_not_dry_run_placeholder

    try:
        check_cuda_or_warn(dry_run=False)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2

    assert_not_dry_run_placeholder(resume_from, what="DPO resume adapter")
    assert_not_dry_run_placeholder(model_path, what="DPO model path")

    try:
        from trl import DPOTrainer
    except ImportError as e:
        print(f"[error] Cannot import DPOTrainer from trl: {e}")
        return 1

    if not dpo_data.exists():
        print(
            f"[error] DPO data not found: {dpo_data}. "
            "Pipeline writes cycle_N_filtered.jsonl; yaml default must match."
        )
        return 1

    try:
        report_to = setup_wandb_env(cfg.get("logging"), dry_run=False)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}")
        return 1
    dtype = cfg.get("model", {}).get("torch_dtype", "float16")
    reference_free = bool(cfg.get("dpo", {}).get("reference_free", False))

    print(f"[info] Loading tokenizer from {base_model}")
    tokenizer = load_tokenizer(base_model)

    eval_split = float(cfg.get("data", {}).get("eval_split", 0.05))
    train_ds, eval_ds = load_preference_dataset(dpo_data, eval_split=eval_split)
    print(
        f"[info] DPO pairs: train={len(train_ds)}"
        + (f", eval={len(eval_ds)}" if eval_ds is not None else "")
    )

    print(f"[info] Loading policy base from {base_model} (dtype={dtype})")
    policy = load_causal_lm(base_model, torch_dtype=dtype)

    if resume_from.exists() and (
        (resume_from / "adapter_config.json").exists()
        or (resume_from / "adapter_model.safetensors").exists()
        or (resume_from / "adapter_model.bin").exists()
    ):
        print(f"[info] Resuming LoRA from RL checkpoint: {resume_from}")
        policy = maybe_merge_or_load_peft(policy, resume_from, is_trainable=True)
        peft_config = None
    else:
        print(
            f"[warn] No LoRA adapter at {resume_from}; "
            "initializing fresh LoRA on base model."
        )
        peft_config = build_lora_config(
            rank=int(lora_cfg.get("rank", 32)),
            alpha=int(lora_cfg.get("alpha", 64)),
            target_modules=lora_cfg.get("target_modules"),
            dropout=float(lora_cfg.get("dropout", 0.05)),
        )

    ref_model = None
    if not reference_free:
        # Reference = Model_RL (frozen). Load a separate copy with the same adapter.
        print(f"[info] Loading reference model (Model_RL) from {resume_from}")
        ref_base = load_causal_lm(base_model, torch_dtype=dtype)
        if resume_from.exists() and (resume_from / "adapter_config.json").exists():
            ref_model = maybe_merge_or_load_peft(
                ref_base, resume_from, is_trainable=False
            )
        else:
            # If resume path is a full merged checkpoint directory, load it directly
            try:
                ref_model = load_causal_lm(resume_from, torch_dtype=dtype)
            except Exception:
                ref_model = ref_base
        ref_model = _freeze_model(ref_model)
    else:
        print("[info] reference_free=true → no explicit reference model")

    dpo_args = build_dpo_config(cfg, output_dir, report_to=report_to)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs: Dict[str, Any] = dict(
        model=policy,
        ref_model=ref_model,
        args=dpo_args if not isinstance(dpo_args, dict) else None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    # TRL API variants across 0.12–0.14+
    def _make_trainer(kwargs: Dict[str, Any]):
        local = dict(kwargs)
        # If args is a plain dict (no DPOConfig class), merge into trainer kwargs
        if local.get("args") is None and isinstance(dpo_args, dict):
            # Pass hyperparams that DPOTrainer accepts directly (older API)
            for k in (
                "beta",
                "loss_type",
                "label_smoothing",
                "max_length",
                "max_prompt_length",
            ):
                if k in dpo_args:
                    local[k] = dpo_args[k]
            from transformers import TrainingArguments

            ta_keys = {
                "output_dir",
                "learning_rate",
                "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "warmup_ratio",
                "fp16",
                "bf16",
                "logging_steps",
                "save_strategy",
                "report_to",
                "remove_unused_columns",
                "run_name",
            }
            ta_kwargs = {k: dpo_args[k] for k in ta_keys if k in dpo_args}
            local["args"] = TrainingArguments(**ta_kwargs)
        try:
            return DPOTrainer(**local)
        except TypeError:
            # processing_class -> tokenizer
            if "processing_class" in local:
                local["tokenizer"] = local.pop("processing_class")
            return DPOTrainer(**local)

    trainer = _make_trainer(trainer_kwargs)

    print("[info] Starting DPO training...")
    trainer.train()

    print(f"[info] Saving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(output_dir))

    print("[info] DPO training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
