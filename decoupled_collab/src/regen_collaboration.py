#!/usr/bin/env python3
"""Regenerate collaboration text with the base model (GOAL Step 3.1)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.prompts import build_regen_messages  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def _cuda_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def _resolve_num_gpus(requested: Optional[int]) -> int:
    available = _cuda_count()
    if available <= 0:
        return 1
    if requested is None or int(requested) <= 0:
        return available
    return min(int(requested), available)


def _split_contiguous(items: list[Any], parts: int) -> list[list[Any]]:
    parts = max(1, min(parts, len(items)))
    chunks: list[list[Any]] = []
    start = 0
    for i in range(parts):
        extra = 1 if i < (len(items) % parts) else 0
        size = len(items) // parts + extra
        chunks.append(items[start : start + size])
        start += size
    return [c for c in chunks if c]


def _release_generator(generator: Any) -> None:
    del generator
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


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


def _reject_think(raw: str) -> str:
    if "<think>" in raw or "</think>" in raw:
        raise RuntimeError(
            "Regen output still contains <think> tags despite enable_thinking=False. "
            "Refusing to write polluted collaboration into DPO pairs. "
            f"Snippet: {raw[:200]!r}"
        )
    return raw


class RegenGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        use_vllm: bool,
        temperature: float,
        max_new_tokens: int,
        device: Optional[int] = 0,
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
        from utils.model_utils import load_tokenizer

        assert_not_dry_run_placeholder(model_path, what="regen base_model")
        require_cuda(dry_run=False)
        # Left padding so batched HF generate aligns new tokens.
        self.tokenizer = load_tokenizer(model_path)

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
            from utils.model_utils import load_causal_lm

            if device is None or device == "auto":
                device_map: Any = "auto"
            else:
                device_map = {"": int(device)}
            self.model = load_causal_lm(
                model_path, torch_dtype="float16", device_map=device_map
            )
            self.model.eval()
            self._device = model_input_device(self.model)

    def _render(self, trace: dict[str, Any]) -> str:
        from utils.model_utils import apply_chat_template_with_thinking

        return apply_chat_template_with_thinking(
            self.tokenizer,
            messages_for_trace(trace),
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, trace: dict[str, Any]) -> str:
        return self.generate_many([trace], batch_size=1)[0]

    def generate_many(
        self,
        traces: list[dict[str, Any]],
        *,
        batch_size: int = 8,
    ) -> list[str]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not traces:
            return []
        texts = [self._render(t) for t in traces]
        if self.use_vllm:
            raws: list[str] = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                outputs = self.llm.generate(chunk, self.sampling_params)
                for item in outputs:
                    raws.append(_reject_think(item.outputs[0].text.strip()))
                _log(f"[regen] {min(start + batch_size, len(texts))}/{len(texts)}")
            return raws

        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        raws = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                    pad_token_id=pad_id,
                )
            for row in out:
                gen = row[prompt_len:]
                raws.append(
                    _reject_think(
                        self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                    )
                )
            _log(f"[regen] {min(start + batch_size, len(texts))}/{len(texts)}")
        return raws


def _regen_shard(payload: dict[str, Any]) -> list[str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu"])
    generator = RegenGenerator(
        payload["model_path"],
        use_vllm=False,
        temperature=payload["temperature"],
        max_new_tokens=payload["max_new_tokens"],
        device=0,
    )
    try:
        return generator.generate_many(
            payload["traces"], batch_size=payload["batch_size"]
        )
    finally:
        _release_generator(generator)


def regen_traces(
    traces: list[dict[str, Any]],
    *,
    model_path: str,
    use_vllm: bool,
    temperature: float,
    max_new_tokens: int,
    gen_batch_size: int,
    num_gpus: Optional[int],
) -> list[str]:
    if not traces:
        return []
    if gen_batch_size < 1:
        raise SystemExit(f"[error] --gen_batch_size must be >= 1, got {gen_batch_size}")
    if use_vllm:
        generator = RegenGenerator(
            model_path,
            use_vllm=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        try:
            _log(f"[regen] traces={len(traces)} backend=vllm batch={gen_batch_size}")
            return generator.generate_many(traces, batch_size=gen_batch_size)
        finally:
            _release_generator(generator)

    n_gpu = _resolve_num_gpus(num_gpus)
    if n_gpu > 1 and len(traces) > 1:
        import multiprocessing as mp

        chunks = _split_contiguous(traces, n_gpu)
        _log(
            f"[regen] traces={len(traces)} gpus={len(chunks)} batch={gen_batch_size}"
        )
        payloads = [
            {
                "gpu": gpu,
                "model_path": model_path,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "batch_size": gen_batch_size,
                "traces": chunk,
            }
            for gpu, chunk in enumerate(chunks)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            parts = pool.map(_regen_shard, payloads)
        merged: list[str] = []
        for part in parts:
            merged.extend(part)
        if len(merged) != len(traces):
            raise RuntimeError(
                f"multi-GPU regen returned {len(merged)} rows for {len(traces)} traces"
            )
        return merged

    generator = RegenGenerator(
        model_path,
        use_vllm=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=0,
    )
    try:
        _log(f"[regen] traces={len(traces)} gpus=1 batch={gen_batch_size}")
        return generator.generate_many(traces, batch_size=gen_batch_size)
    finally:
        _release_generator(generator)


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
        "--gen_batch_size",
        type=int,
        default=8,
        help="HF/vLLM generate batch size per GPU",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="HF data-parallel GPU workers; 0 = all visible devices",
    )
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
    _log(f"Traces: {len(traces)} total, {len(done)} done, {len(remaining)} remaining")

    if args.dry_run:
        texts = [dry_run_regen(t) for t in remaining]
    else:
        texts = regen_traces(
            remaining,
            model_path=args.base_model,
            use_vllm=args.use_vllm,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            num_gpus=args.num_gpus,
        )

    with open(args.output, "a", encoding="utf-8") as fout:
        for trace, regen in zip(remaining, texts):
            pair = make_raw_pair(trace, regen)
            if args.dry_run:
                pair["dry_run"] = True
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")

    _log(f"Wrote raw DPO pairs → {args.output}")


if __name__ == "__main__":
    main()
