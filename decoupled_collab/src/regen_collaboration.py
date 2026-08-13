#!/usr/bin/env python3
"""Regenerate collaboration text with the base model (GOAL Step 3.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.prompts import build_regen_messages  # noqa: E402


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


def messages_for_trace(trace: dict[str, Any]) -> list[dict[str, str]]:
    return build_regen_messages(
        trace.get("task_prompt", ""),
        trace.get("thinking", ""),
        trace.get("code", ""),
    )


def dry_run_regen(trace: dict[str, Any]) -> str:
    """Copy collaboration with a slight paraphrase marker for offline tests."""
    original = (trace.get("collaboration") or "").strip()
    if not original:
        original = "Here is a clear explanation of the solution approach."
    return f"[regen-paraphrase] {original}"


class RegenGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        use_vllm: bool,
        temperature: float,
        max_new_tokens: int,
    ):
        self.use_vllm = use_vllm
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.sampling_params = None

        from utils.failfast import (
            assert_not_adapter_for_vllm,
            assert_not_dry_run_placeholder,
            model_input_device,
            require_cuda,
        )
        from transformers import AutoTokenizer

        assert_not_dry_run_placeholder(model_path, what="regen base_model")
        require_cuda(dry_run=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if use_vllm:
            assert_not_adapter_for_vllm(model_path)
            from vllm import LLM, SamplingParams

            self.llm = LLM(model=model_path, trust_remote_code=True)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.9,
            )
        else:
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self._device = model_input_device(self.model)

    def generate(self, trace: dict[str, Any]) -> str:
        messages = messages_for_trace(trace)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.use_vllm:
            outputs = self.llm.generate([text], self.sampling_params)
            return outputs[0].outputs[0].text.strip()

        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def make_raw_pair(trace: dict[str, Any], regen_collaboration: str) -> dict[str, Any]:
    return {
        "task_id": trace.get("task_id"),
        "task_prompt": trace.get("task_prompt", ""),
        "thinking": trace.get("thinking", ""),
        "code": trace.get("code", ""),
        "rl_collaboration": trace.get("collaboration", ""),
        "regen_collaboration": regen_collaboration,
        "reward": trace.get("reward", 0.0),
        "full_output": trace.get("full_output"),
        "work_trace": trace.get("work_trace"),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate collaboration (GOAL Step 3.1)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_vllm", type=_str2bool, default=False)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Copy collaboration with paraphrase marker (no GPU)",
    )
    args = parser.parse_args(argv)

    traces = load_jsonl(args.traces)[: args.num_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.output)
    remaining = [t for t in traces if t.get("task_id") not in done]
    print(f"Traces: {len(traces)} total, {len(done)} done, {len(remaining)} remaining")

    generator = None
    if not args.dry_run:
        generator = RegenGenerator(
            args.base_model,
            use_vllm=args.use_vllm,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    with open(args.output, "a", encoding="utf-8") as fout:
        for trace in _progress(remaining, total=len(remaining), desc="Regen collaboration"):
            if args.dry_run:
                regen = dry_run_regen(trace)
            else:
                regen = generator.generate(trace)
            pair = make_raw_pair(trace, regen)
            if args.dry_run:
                pair["dry_run"] = True
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Wrote raw DPO pairs → {args.output}")


if __name__ == "__main__":
    main()
