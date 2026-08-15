"""Fail-fast helpers: raise clear errors instead of silent fallbacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigError(RuntimeError):
    """Invalid or incompatible configuration for a real run."""


def resolve_trainer_resume_checkpoint(
    requested: object,
    output_dir: Path,
    *,
    root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve an explicit or ``auto`` Transformers Trainer checkpoint.

    Auto-resume only accepts directories containing ``trainer_state.json`` and
    chooses the highest numeric ``checkpoint-N``. A requested resume never
    silently falls back to a fresh run.
    """
    if requested in (None, False, "", "false", "False", "none", "None"):
        return None

    if requested is True or str(requested).lower() == "auto":
        candidates: list[tuple[int, Path]] = []
        if output_dir.is_dir():
            for path in output_dir.glob("checkpoint-*"):
                if not path.is_dir() or not (path / "trainer_state.json").is_file():
                    continue
                try:
                    step = int(path.name.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                candidates.append((step, path.resolve()))
        if not candidates:
            raise ConfigError(
                f"Resume requested but no valid checkpoint-N/trainer_state.json "
                f"exists under {output_dir}. Refusing to silently start over."
            )
        return max(candidates, key=lambda item: item[0])[1]

    path = Path(str(requested))
    if not path.is_absolute():
        path = ((root or output_dir.parent) / path).resolve()
    if not path.is_dir() or not (path / "trainer_state.json").is_file():
        raise ConfigError(
            f"Invalid resume checkpoint {path}: expected a directory containing "
            "trainer_state.json."
        )
    return path


def require_exists(path: Union[str, Path], what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found: {p}. Refusing to continue with a missing path."
        )
    return p


def assert_not_dry_run_placeholder(path: Union[str, Path], what: str = "checkpoint") -> None:
    """Refuse to treat dry-run stub directories as real model weights."""
    p = Path(path)
    marker = p / "DRY_RUN_PLACEHOLDER"
    if marker.exists():
        raise ConfigError(
            f"{what} at {p} is a dry-run placeholder ({marker.name}), not a trained "
            "adapter/model. Re-run the real training phase without --dry_run, or point "
            "to a real checkpoint. Refusing to load fake weights."
        )


def require_cuda(*, dry_run: bool = False) -> None:
    """Require CUDA for non-dry-run training/inference. No soft continue."""
    if dry_run:
        return
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is not installed. Install torch (CUDA build) before running "
            "without --dry_run. Example: "
            "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available but this command was started without --dry_run. "
            "Target hardware is 4×V100-32G. Fix the GPU/driver/torch CUDA install, "
            "or pass --dry_run for offline planning only."
        )
    n = torch.cuda.device_count()
    print(f"[info] CUDA available: {n} device(s)")


def model_input_device(model) -> "torch.device":  # type: ignore[name-defined]
    """Resolve device for inputs. Fail if model has no parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration as e:
        raise RuntimeError(
            "Model has no parameters; cannot place inputs on a device."
        ) from e


def require_thinking_support(tokenizer, *, enable_thinking: bool) -> None:
    """
    Fail fast if thinking mode is required but the tokenizer rejects the kwarg.

    We probe with a minimal call so train/eval do not silently drop <think>.
    """
    if not enable_thinking:
        return
    messages = [{"role": "user", "content": "ping"}]
    try:
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as e:
        raise RuntimeError(
            "enable_thinking=True is required for this experiment, but "
            "tokenizer.apply_chat_template() rejects that keyword. "
            "Install a transformers build that supports Qwen3 thinking mode "
            "(need transformers>=4.51; see GOAL Step 0.2 / requirements.txt). "
            f"Underlying error: {e}"
        ) from e


def apply_chat_template_thinking_strict(
    tokenizer,
    messages,
    *,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
):
    """apply_chat_template with thinking; never silently drop the flag."""
    base = dict(
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not enable_thinking:
        # Qwen3 defaults to thinking-ON when the kwarg is omitted — must pass False.
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **base
            )
        except TypeError as e:
            raise RuntimeError(
                "Refusing to omit enable_thinking=False. Qwen3 chat templates default "
                "to thinking mode when the flag is absent, which would pollute "
                "collaboration-only regen with <think> text. "
                f"Underlying error: {e}"
            ) from e
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=True, **base
        )
    except TypeError as e:
        raise RuntimeError(
            "Refusing to fall back to a non-thinking chat template. "
            "Qwen3 thinking mode (enable_thinking=True) is required so work/collaboration "
            "layers stay separable via <think> tags. "
            "Upgrade transformers for Qwen3 support, or pass enable_thinking=False "
            "only if you intentionally disable the methodology. "
            f"Underlying error: {e}"
        ) from e


def assert_not_adapter_for_vllm(model_path: Union[str, Path]) -> None:
    p = Path(model_path)
    if (p / "adapter_config.json").exists():
        raise ConfigError(
            f"use_vllm=True but model_path is a PEFT/LoRA adapter directory: {p}. "
            "vLLM's LLM(model=...) expects a full model, not adapter_config.json. "
            "Merge the adapter into the base model first, or set --use_vllm false "
            "to generate with HuggingFace+PEFT."
        )


def validate_grpo_batch_vs_generations(
    cfg: Dict[str, Any],
    *,
    num_processes: Optional[int] = None,
) -> None:
    """
    TRL GRPO requires (num_processes * per_device_batch_size) % num_generations == 0.
    Fail before launching training with a fix hint.
    """
    training = cfg.get("training", {})
    grpo = cfg.get("grpo", {})
    per_device = int(training.get("per_device_batch_size", 1))
    num_gen = int(grpo.get("num_samples_per_prompt", grpo.get("num_generations", 6)))
    if num_processes is None:
        num_processes = int(
            os.environ.get("WORLD_SIZE")
            or os.environ.get("ACCELERATE_NUM_PROCESSES")
            or "1"
        )
    global_batch = num_processes * per_device
    if num_gen <= 0:
        raise ConfigError(f"num_samples_per_prompt must be > 0, got {num_gen}")
    if global_batch % num_gen != 0:
        raise ConfigError(
            "GRPO batch/generations mismatch (TRL will fail or behave incorrectly): "
            f"num_processes({num_processes}) * per_device_batch_size({per_device}) "
            f"= {global_batch}, which is not divisible by "
            f"num_samples_per_prompt/num_generations({num_gen}). "
            "Fix options: set grpo.num_samples_per_prompt to a divisor of the global "
            "batch (e.g. 4 for 4×V100 with batch_size=1), or raise "
            "per_device_batch_size / change accelerate --num_processes. "
            "Refusing to start with a known-broken config."
        )


def validate_wandb_if_enabled(logging_cfg: Optional[Dict[str, Any]], *, dry_run: bool) -> None:
    if dry_run:
        return
    logging_cfg = logging_cfg or {}
    project = os.environ.get("WANDB_PROJECT") or logging_cfg.get("wandb_project")
    if not project:
        return
    if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE") == "disabled":
        raise ConfigError(
            f"logging.wandb_project={project!r} is set but WANDB_API_KEY is missing "
            "and WANDB_MODE is not 'disabled'. Either export WANDB_API_KEY, "
            "set WANDB_MODE=disabled, or clear wandb_project in the YAML. "
            "Refusing to hang on an interactive wandb login."
        )
