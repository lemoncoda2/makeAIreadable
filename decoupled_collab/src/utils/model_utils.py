"""
Model / tokenizer helpers shared by GRPO and DPO training scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def resolve_path(path: Union[str, Path], root: Path) -> Path:
    """Resolve relative paths against project root; leave absolute paths as-is."""
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[key]


def load_tokenizer(model_path: Union[str, Path], trust_remote_code: bool = True):
    """Load AutoTokenizer with left padding (generation / GRPO friendly)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_path: Union[str, Path],
    torch_dtype: Union[str, Any] = "float16",
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
    trust_remote_code: bool = True,
    **kwargs,
):
    """
    Load AutoModelForCausalLM.

    Does not download if path is local and missing — raises a clear error.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model_path = Path(model_path)
    if model_path.exists() and model_path.is_dir() and not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"Model directory exists but is empty: {model_path}. "
            "Download the model before training (this script will not download)."
        )
    # Treat absolute / relative filesystem paths as local (no auto-download).
    looks_local = model_path.is_absolute() or str(model_path).startswith(
        ("./", "../", "/")
    )
    if looks_local and not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            "Download the model before training (this script will not download)."
        )

    dtype = (
        torch_dtype
        if isinstance(torch_dtype, torch.dtype)
        else _dtype_from_name(torch_dtype)
    )

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    return AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)


def apply_lora(
    model,
    rank: int = 32,
    alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
    task_type: str = "CAUSAL_LM",
):
    """Wrap a causal LM with LoRA via peft."""
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=getattr(TaskType, task_type, TaskType.CAUSAL_LM),
    )
    return get_peft_model(model, peft_config), peft_config


def build_lora_config(
    rank: int = 32,
    alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
):
    """Return a peft.LoraConfig (for passing peft_config into TRL trainers)."""
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_chat_template_with_thinking(
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
):
    """
    Apply chat template with Qwen3 thinking mode when supported.

    Tries `enable_thinking=...`, then `thinking=...`, then plain apply_chat_template.
    """
    base_kwargs = dict(
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **base_kwargs
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            messages, thinking=enable_thinking, **base_kwargs
        )
    except TypeError:
        pass
    return tokenizer.apply_chat_template(messages, **base_kwargs)


def maybe_merge_or_load_peft(
    model,
    adapter_path: Optional[Union[str, Path]] = None,
    is_trainable: bool = True,
    merge_and_unload: bool = False,
):
    """
    Attach or resume a PEFT adapter.

    - If `adapter_path` is set and exists, load it onto `model` (base or already Peft).
    - If `merge_and_unload`, merge LoRA weights into base and return a plain model.
    """
    from peft import PeftModel

    if adapter_path is None:
        return model

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter path not found: {adapter_path}. "
            "Train GRPO first or point resume_from to an existing adapter."
        )

    if isinstance(model, PeftModel):
        model.load_adapter(str(adapter_path), adapter_name="default")
        if is_trainable:
            model.set_adapter("default")
            model.train()
        else:
            model.eval()
    else:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=is_trainable,
        )

    if merge_and_unload:
        model = model.merge_and_unload()
    return model


def find_base_model_path(
    name_or_path: Union[str, Path],
    default_base: Union[str, Path] = "./models/Qwen3-4B",
    root: Optional[Path] = None,
) -> Path:
    """
    Resolve the underlying base model directory.

    If `name_or_path` looks like a LoRA/checkpoint dir (has adapter_config.json),
    fall back to `default_base` (or adapter_config's base_model_name_or_path).
    """
    path = Path(name_or_path)
    if root is not None and not path.is_absolute():
        path = resolve_path(path, root)

    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        with adapter_cfg.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path") or default_base
        base_path = Path(base)
        if root is not None and not base_path.is_absolute():
            base_path = resolve_path(base_path, root)
        return base_path

    if path.exists():
        return path

    default = Path(default_base)
    if root is not None and not default.is_absolute():
        default = resolve_path(default, root)
    return default


def setup_wandb_env(logging_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Configure wandb report_to based on env / config.

    Returns report_to string: "wandb" or "none".
    """
    logging_cfg = logging_cfg or {}
    project = os.environ.get("WANDB_PROJECT") or logging_cfg.get("wandb_project")
    run_name = logging_cfg.get("wandb_run")
    if project:
        os.environ.setdefault("WANDB_PROJECT", str(project))
        if run_name and "WANDB_RUN_NAME" not in os.environ:
            os.environ["WANDB_RUN_NAME"] = str(run_name)
        return "wandb"
    return "none"


def check_cuda_or_warn(dry_run: bool = False) -> bool:
    """
    Return True if CUDA is available.
    Print a graceful message when unavailable; allow dry_run without failing hard.
    """
    try:
        import torch
    except ImportError:
        print("[warn] PyTorch is not installed.")
        if dry_run:
            print("        Continuing dry-run without training.")
            return False
        print("        Use --dry_run to load config and print the plan without training.")
        return False

    available = torch.cuda.is_available()
    if available:
        print(f"[info] CUDA available: {torch.cuda.device_count()} device(s)")
        return True
    msg = (
        "[warn] CUDA is not available. Training requires GPU(s) "
        "(target hardware: 4×V100-32G)."
    )
    if dry_run:
        print(msg + " Continuing dry-run without training.")
        return False
    print(msg)
    print("        Use --dry_run to load config and print the plan without training.")
    return False
