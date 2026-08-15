#!/usr/bin/env python3
"""Collect agent traces with thinking-mode separation (GOAL Step 2.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.code_executor import compute_reward, separate_output  # noqa: E402
from utils.harvest_rollouts import (  # noqa: E402
    find_reward_audit_dir,
    harvest_rollouts,
    write_traces,
)
from utils.prompts import build_coding_messages  # noqa: E402
from utils.thinking_budget import (  # noqa: E402
    build_thinking_budget_logits_processor,
)


def _progress(iterable, total: Optional[int] = None, desc: str = ""):
    try:
        from rich.progress import track

        return track(iterable, total=total, description=desc or "Working...")
    except ImportError:
        try:
            from tqdm import tqdm

            return tqdm(iterable, total=total, desc=desc)
        except ImportError:
            if desc:
                print(desc)
            return iterable


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    with open(output, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _build_messages(
    task_prompt: str, public_test_cases: Optional[list[str]] = None
) -> list[dict[str, str]]:
    return build_coding_messages(task_prompt, public_test_cases)


def _fake_trace_from_gt(task: dict[str, Any]) -> dict[str, Any]:
    """Dry-run: synthesize a realistic thinking-mode output from ground truth."""
    code = task.get("code_solution") or task.get("code") or "def solution():\n    pass\n"
    thinking = (
        f"Analyze the problem: {task.get('prompt', '')[:120]}. "
        "I will implement a direct solution matching the reference."
    )
    collab = (
        "I understood the request and implemented a straightforward solution. "
        "Here is the working code with the expected interface."
    )
    full_output = (
        f"<think>\n{thinking}\n</think>\n\n"
        f"{collab}\n\n"
        f"```python\n{code}\n```\n"
    )
    separated = separate_output(full_output)
    reward = compute_reward(full_output, task.get("test_cases") or [])
    return {
        "task_id": task["task_id"],
        "task_prompt": task.get("prompt", ""),
        "full_output": full_output,
        "thinking": separated["thinking"],
        "code": separated["code"],
        "collaboration": separated["collaboration"],
        "reward": reward,
        "work_trace": separated["thinking"] + "\n[CODE]\n" + separated["code"],
        "dry_run": True,
    }


class TraceGenerator:
    """PEFT/base HF generator or optional vLLM backend."""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str],
        *,
        use_vllm: bool,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
        thinking_budget_tokens: Optional[int] = None,
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        if thinking_budget_tokens is not None and not enable_thinking:
            raise ValueError(
                "thinking_budget_tokens requires enable_thinking=True"
            )
        if thinking_budget_tokens is not None and use_vllm:
            raise ValueError(
                "thinking_budget_tokens is not implemented for vLLM; refusing "
                "to silently ignore the budget"
            )
        self.use_vllm = use_vllm
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.sampling_params = None

        from utils.failfast import (
            assert_not_adapter_for_vllm,
            assert_not_dry_run_placeholder,
            model_input_device,
            require_cuda,
            require_thinking_support,
        )
        from utils.failfast import apply_chat_template_thinking_strict

        self._apply_chat = apply_chat_template_thinking_strict
        assert_not_dry_run_placeholder(model_path, what="collect_traces model")
        require_cuda(dry_run=False)

        if use_vllm:
            assert_not_adapter_for_vllm(model_path)
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.9,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path or model_path, trust_remote_code=True
            )
            require_thinking_support(self.tokenizer, enable_thinking=enable_thinking)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            adapter_cfg = Path(model_path) / "adapter_config.json"
            if adapter_cfg.exists():
                if not base_model_path:
                    raise ValueError(
                        f"{model_path} is a PEFT adapter but --base_model_path was "
                        "not provided. Refusing to guess the base model."
                    )
                tok_src = base_model_path
            else:
                tok_src = base_model_path or model_path

            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_src, trust_remote_code=True
            )
            require_thinking_support(self.tokenizer, enable_thinking=enable_thinking)
            base = AutoModelForCausalLM.from_pretrained(
                tok_src,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            if adapter_cfg.exists():
                self.model = PeftModel.from_pretrained(base, model_path)
            elif Path(model_path).resolve() != Path(tok_src).resolve():
                raise ValueError(
                    f"model_path={model_path} is not a PEFT adapter (no "
                    f"adapter_config.json) but differs from tokenizer/base "
                    f"source {tok_src}. Refusing ambiguous load."
                )
            else:
                self.model = base
            self.model.eval()
            self._device = model_input_device(self.model)

    def generate(
        self, task_prompt: str, public_test_cases: Optional[list[str]] = None
    ) -> str:
        messages = _build_messages(task_prompt, public_test_cases)
        text = self._apply_chat(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.use_vllm:
            outputs = self.llm.generate([text], self.sampling_params)
            return outputs[0].outputs[0].text

        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        generation_kwargs = {}
        if self.thinking_budget_tokens is not None:
            generation_kwargs["logits_processor"] = (
                build_thinking_budget_logits_processor(
                    self.tokenizer,
                    prompt_length=inputs["input_ids"].shape[1],
                    thinking_budget_tokens=self.thinking_budget_tokens,
                )
            )
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                **generation_kwargs,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=False)


def collect_single_trace(generator: TraceGenerator, task: dict[str, Any]) -> dict[str, Any]:
    output = generator.generate(task["prompt"], (task.get("test_cases") or [])[:1])
    separated = separate_output(output)
    reward = compute_reward(output, task.get("test_cases") or [])
    return {
        "task_id": task["task_id"],
        "task_prompt": task.get("prompt", ""),
        "full_output": output,
        "thinking": separated["thinking"],
        "code": separated["code"],
        "collaboration": separated["collaboration"],
        "reward": reward,
        "work_trace": separated["thinking"] + "\n[CODE]\n" + separated["code"],
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Collect RL traces (GOAL Step 2.2)")
    parser.add_argument("--model", required=True, help="RL / PEFT model path")
    parser.add_argument("--base_model_path", default=None, help="Base model for PEFT / tokenizer")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_tasks", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--enable_thinking", type=_str2bool, default=True)
    parser.add_argument(
        "--thinking_budget_tokens",
        type=int,
        default=None,
        help="Force </think> after this many thinking tokens, then continue output",
    )
    parser.add_argument("--use_vllm", type=_str2bool, default=False)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate fake traces from ground-truth (no GPU)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Ignore reward_audit and sample the model (old collect path)",
    )
    parser.add_argument(
        "--min_reward",
        type=float,
        default=0.0,
        help="Keep GRPO rollouts with reward strictly greater than this",
    )
    parser.add_argument(
        "--later_frac",
        type=float,
        default=1.0 / 3.0,
        help="Prefer this last fraction of GRPO audit calls (fallback if too few)",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        _write_live_or_dry_traces(args, dry_run=True)
        return

    audit_dir = None if args.live else find_reward_audit_dir(args.model)
    if audit_dir is not None:
        traces, stats = harvest_rollouts(
            audit_dir,
            args.tasks,
            min_reward=args.min_reward,
            later_frac=args.later_frac,
            max_traces=args.num_tasks,
        )
        write_traces(traces, args.output)
        print(f"[harvest] {json.dumps(stats, ensure_ascii=False)}")
        print(f"Wrote {len(traces)} harvested traces → {args.output}")
        return

    if not args.live:
        raise SystemExit(
            f"[error] No reward_audit under {args.model}. "
            "DPO traces now come from GRPO rollouts. Re-run GRPO so "
            "reward_audit/reward_rank*.jsonl exists, or pass --live to sample."
        )
    _write_live_or_dry_traces(args, dry_run=False)


def _write_live_or_dry_traces(args: argparse.Namespace, *, dry_run: bool) -> None:
    tasks = load_jsonl(args.tasks)[: args.num_tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.output)
    remaining = [t for t in tasks if t.get("task_id") not in done]
    print(f"Tasks: {len(tasks)} total, {len(done)} done, {len(remaining)} remaining")

    generator = None
    if not dry_run:
        generator = TraceGenerator(
            args.model,
            args.base_model_path,
            use_vllm=args.use_vllm,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
        )

    with open(args.output, "a", encoding="utf-8") as fout:
        for task in _progress(remaining, total=len(remaining), desc="Collecting traces"):
            if dry_run:
                record = _fake_trace_from_gt(task)
            else:
                record = collect_single_trace(generator, task)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Wrote traces → {args.output}")


if __name__ == "__main__":
    main()
