#!/usr/bin/env python3
"""Evaluate base / RL / final models (GOAL Step 4).

Modes: full | hypothesis_check | benchmark | readability
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.api_judge import judge_batch, mock_judge_scores  # noqa: E402
from utils.code_executor import extract_code, separate_output  # noqa: E402
from utils.metrics import (  # noqa: E402
    aggregate_readability_scores,
    avg_length_tokens,
    summarize_eval_results,
    syntax_error_rate,
    think_leak_rate,
)
from utils.prompts import build_coding_messages  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def pass_at_1(model_output: str, test_cases: list[str], timeout: int = 10) -> bool:
    """True only when code extracts and every provided test executes successfully.

    Empty/missing test_cases never count as a pass (that was a fake-success footgun).
    """
    if not test_cases:
        return False
    from utils.code_executor import execute_tests

    code = extract_code(model_output)
    if not code:
        return False
    return all(execute_tests(code, list(test_cases), timeout))


def dry_run_generation(task: dict[str, Any], model_tag: str) -> str:
    """Mock generation from ground-truth solution / trivial stub."""
    code = (
        task.get("code_solution")
        or task.get("code")
        or "def solution(*args, **kwargs):\n    return None\n"
    )
    if model_tag == "rl":
        collab = (
            "ok code done. used some approach. whatever. "
            "long chain of thought leak: " + (task.get("prompt", "")[:80])
        )
        thinking = "optimize hard; skip explanations for the user response"
    elif model_tag == "final":
        collab = (
            "I understood your request and implemented a clear solution. "
            "The core idea is a direct implementation of the specified behavior."
        )
        thinking = "Recover readable explanation after DPO."
    else:
        collab = (
            "I understood the problem and wrote a straightforward implementation. "
            "The approach follows the problem statement directly."
        )
        thinking = f"Solve: {task.get('prompt', '')[:100]}"

    return (
        f"<think>\n{thinking}\n</think>\n\n"
        f"{collab}\n\n"
        f"```python\n{code}\n```\n"
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _release_runner(runner: Any) -> None:
    del runner
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        thinking_budget_tokens: int = 256,
        device: Optional[int] = 0,
    ):
        from utils.failfast import (
            assert_not_dry_run_placeholder,
            model_input_device,
            require_cuda,
            require_thinking_support,
        )
        from utils.model_utils import (
            apply_chat_template_with_thinking,
            load_causal_lm,
            load_tokenizer,
            maybe_merge_or_load_peft,
        )

        if not model_path:
            raise ValueError("model_path is required for ModelRunner (got empty/None)")

        assert_not_dry_run_placeholder(model_path, what="eval model")
        require_cuda(dry_run=False)

        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            if not base_model_path:
                raise ValueError(
                    f"{model_path} looks like a PEFT adapter (adapter_config.json) "
                    "but --base_model was not provided. Pass the base model path "
                    "explicitly; refusing to guess."
                )
            tok_src = base_model_path
        else:
            tok_src = base_model_path or model_path

        self.tokenizer = load_tokenizer(tok_src)
        require_thinking_support(self.tokenizer, enable_thinking=True)
        # Put the full 4B on one GPU. device_map="auto" shards one sequence
        # across cards and serializes decode; a V100-32G holds Qwen3-4B FP16.
        if device is None or device == "auto":
            device_map: Any = "auto"
        else:
            device_map = {"": int(device)}
        base = load_causal_lm(tok_src, torch_dtype="float16", device_map=device_map)
        if adapter_cfg.exists():
            self.model = maybe_merge_or_load_peft(
                base, model_path, is_trainable=False
            )
        else:
            if base_model_path and Path(model_path).resolve() != Path(tok_src).resolve():
                raise ValueError(
                    f"model_path={model_path} is not a PEFT adapter and differs from "
                    f"base={tok_src}. Refusing ambiguous load."
                )
            self.model = base
        self.model.eval()
        from utils.thinking_budget import install_thinking_budget_generate

        install_thinking_budget_generate(
            self.model,
            self.tokenizer,
            thinking_budget_tokens=thinking_budget_tokens,
            stop_after_code_fence=True,
        )
        self._apply = apply_chat_template_with_thinking
        self._device = model_input_device(self.model)

    def generate(
        self,
        prompt: str,
        public_test_cases: Optional[list[str]] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
    ) -> str:
        return self.generate_many(
            [{"prompt": prompt, "test_cases": list(public_test_cases or [])}],
            max_new_tokens=max_new_tokens,
            batch_size=1,
            temperature=temperature,
        )[0]

    def generate_many(
        self,
        tasks: list[dict[str, Any]],
        *,
        max_new_tokens: int = 768,
        batch_size: int = 4,
        temperature: float = 0.2,
    ) -> list[str]:
        import torch

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        texts = []
        for task in tasks:
            messages = build_coding_messages(
                task["prompt"], (task.get("test_cases") or [])[:1]
            )
            texts.append(self._apply(self.tokenizer, messages, enable_thinking=True))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=pad_id,
                )
            for row in out:
                gen = row[prompt_len:]
                outputs.append(self.tokenizer.decode(gen, skip_special_tokens=False))
            _log(f"[gen] {min(start + batch_size, len(texts))}/{len(texts)}")
        return outputs


def _is_executable_assert(tc: Any) -> bool:
    return isinstance(tc, str) and tc.strip().startswith("assert")


def fraction_executable_assert_tasks(tasks: list[dict[str, Any]]) -> float:
    """Share of tasks whose test_cases are assert-style strings (code_executor-ready)."""
    if not tasks:
        return 0.0
    ok = 0
    for t in tasks:
        tcs = t.get("test_cases") or []
        if tcs and all(_is_executable_assert(x) for x in tcs):
            ok += 1
    return ok / len(tasks)


def pass_task(model_output: str, task: dict[str, Any], *, timeout: int = 10) -> bool:
    """Dispatch MBPP assert harness vs LiveCodeBench stdin/call harness."""
    from utils.lcb_executor import pass_lcb, task_lcb_cases

    if task.get("harness") == "lcb" or task_lcb_cases(task):
        code = extract_code(model_output)
        return pass_lcb(code, task, timeout=timeout)
    return pass_at_1(model_output, task.get("test_cases") or [], timeout=timeout)


def evaluate_benchmark(tasks: list[dict[str, Any]], outputs: list[str]) -> dict[str, float]:
    from utils.lcb_executor import fraction_lcb_ready

    passes = [pass_task(o, t) for o, t in zip(outputs, tasks)]
    codes = [extract_code(o) for o in outputs]
    return {
        "pass_at_1": mean(float(p) for p in passes),
        "avg_code_length": mean(float(len(c)) for c in codes),
        "syntax_error_rate": syntax_error_rate(codes),
        "n": float(len(tasks)),
        "executable_assert_task_frac": fraction_executable_assert_tasks(tasks),
        "lcb_ready_frac": fraction_lcb_ready(tasks),
    }


def evaluate_readability(
    tasks: list[dict[str, Any]],
    outputs: list[str],
    *,
    mock_judge: bool,
    model_tag: str,
    judge_max_concurrent: int = 5,
) -> dict[str, Any]:
    scores: list[dict[str, float]] = []
    collabs: list[str] = []
    leaks: list[float] = []

    for task, output in zip(tasks, outputs):
        sep = separate_output(output)
        collab = sep["collaboration"]
        collabs.append(collab)
        leaks.append(think_leak_rate(sep["thinking"], collab))

        if mock_judge:
            regen_like = collab if model_tag != "rl" else collab + " [clearer]"
            rl_like = collab if model_tag == "rl" else "ok done."
            good, bad = mock_judge_scores(regen_like, rl_like)
            scores.append(good if model_tag != "rl" else bad)

    if not mock_judge:
        items = [
            {
                "task_prompt": task.get("prompt", ""),
                "text": collab or "(empty)",
            }
            for task, collab in zip(tasks, collabs)
        ]
        judged = asyncio.run(
            judge_batch(items, max_concurrent=judge_max_concurrent)
        )
        scores = [row["score"] for row in judged]

    detail = aggregate_readability_scores(scores)
    return {
        **detail,
        "avg_collab_length": avg_length_tokens(collabs),
        "think_leak_rate": mean(leaks),
        "n": float(len(tasks)),
    }


def _split_contiguous(tasks: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    parts = max(1, min(parts, len(tasks)))
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    for i in range(parts):
        extra = 1 if i < (len(tasks) % parts) else 0
        size = len(tasks) // parts + extra
        chunks.append(tasks[start : start + size])
        start += size
    return [c for c in chunks if c]


def _generate_shard(payload: dict[str, Any]) -> list[str]:
    """Spawn worker: pin one visible GPU, then batch-generate a task shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu"])
    runner = ModelRunner(
        payload["model_path"],
        base_model_path=payload["base_model_path"],
        thinking_budget_tokens=payload["thinking_budget_tokens"],
        device=0,
    )
    try:
        return runner.generate_many(
            payload["tasks"],
            max_new_tokens=payload["max_new_tokens"],
            batch_size=payload["batch_size"],
        )
    finally:
        _release_runner(runner)


def generate_for_model(
    model_tag: str,
    model_path: Optional[str],
    base_model_path: Optional[str],
    tasks: list[dict[str, Any]],
    *,
    dry_run: bool,
    max_new_tokens: int = 768,
    thinking_budget_tokens: int = 256,
    gen_batch_size: int = 4,
    num_gpus: Optional[int] = None,
) -> list[str]:
    if dry_run:
        return [dry_run_generation(t, model_tag) for t in tasks]
    if not model_path:
        raise SystemExit(
            f"[error] evaluate: model path for tag={model_tag!r} is missing. "
            "Refusing to inject ground-truth dry_run_generation stubs outside --dry_run "
            "(that would fake pass@1 / readability)."
        )
    if gen_batch_size < 1:
        raise SystemExit(f"[error] --gen_batch_size must be >= 1, got {gen_batch_size}")
    if not tasks:
        return []

    n_gpu = _resolve_num_gpus(num_gpus)
    if n_gpu > 1 and len(tasks) > 1:
        import multiprocessing as mp

        chunks = _split_contiguous(tasks, n_gpu)
        _log(
            f"[gen] tag={model_tag} tasks={len(tasks)} gpus={len(chunks)} "
            f"batch={gen_batch_size}"
        )
        payloads = [
            {
                "gpu": gpu,
                "model_path": model_path,
                "base_model_path": base_model_path,
                "thinking_budget_tokens": thinking_budget_tokens,
                "max_new_tokens": max_new_tokens,
                "batch_size": gen_batch_size,
                "tasks": chunk,
            }
            for gpu, chunk in enumerate(chunks)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            parts = pool.map(_generate_shard, payloads)
        merged: list[str] = []
        for part in parts:
            merged.extend(part)
        if len(merged) != len(tasks):
            raise RuntimeError(
                f"multi-GPU generate returned {len(merged)} rows for {len(tasks)} tasks"
            )
        return merged

    runner = ModelRunner(
        model_path,
        base_model_path=base_model_path,
        thinking_budget_tokens=thinking_budget_tokens,
        device=0,
    )
    try:
        if len(tasks) == 1:
            return [
                runner.generate(
                    tasks[0]["prompt"],
                    (tasks[0].get("test_cases") or [])[:1],
                    max_new_tokens=max_new_tokens,
                )
            ]
        _log(
            f"[gen] tag={model_tag} tasks={len(tasks)} gpus=1 batch={gen_batch_size}"
        )
        return runner.generate_many(
            tasks,
            max_new_tokens=max_new_tokens,
            batch_size=gen_batch_size,
        )
    finally:
        _release_runner(runner)


def evaluate_model(
    model_tag: str,
    model_path: Optional[str],
    base_model_path: Optional[str],
    mbpp_tasks: list[dict[str, Any]],
    lcb_tasks: list[dict[str, Any]],
    *,
    mode: str,
    dry_run: bool,
    mock_judge: bool,
    num_tasks_benchmark: int,
    num_tasks_readability: int,
    max_new_tokens: int = 768,
    thinking_budget_tokens: int = 256,
    gen_batch_size: int = 4,
    num_gpus: Optional[int] = None,
    judge_max_concurrent: int = 5,
) -> dict[str, Any]:
    bench_tasks = mbpp_tasks[:num_tasks_benchmark]
    read_tasks = mbpp_tasks[:num_tasks_readability]
    lcb_eval = lcb_tasks[:num_tasks_benchmark] if lcb_tasks else []

    result: dict[str, Any] = {}
    need_bench = mode in ("full", "hypothesis_check", "benchmark")
    need_read = mode in ("full", "hypothesis_check", "readability")
    gen_kwargs = {
        "dry_run": dry_run,
        "max_new_tokens": max_new_tokens,
        "thinking_budget_tokens": thinking_budget_tokens,
        "gen_batch_size": gen_batch_size,
        "num_gpus": num_gpus,
    }

    bench_outs: Optional[list[str]] = None
    if need_bench:
        if lcb_eval:
            from utils.lcb_executor import fraction_lcb_ready

            lcb_frac = fraction_lcb_ready(lcb_eval)
            if lcb_frac < 0.5 and not dry_run:
                raise SystemExit(
                    f"[error] LiveCodeBench tasks have structured lcb_tests coverage "
                    f"{lcb_frac:.0%} (<50%). Re-run: python src/prepare_data.py --download "
                    "(stores harness=lcb + stdin/call cases). "
                    "Do not use assert-only fixtures for LCB."
                )
        else:
            lcb_frac = None
        gen_tasks = list(bench_tasks) + list(lcb_eval)
        all_outs = generate_for_model(
            model_tag,
            model_path,
            base_model_path,
            gen_tasks,
            **gen_kwargs,
        )
        if len(all_outs) != len(gen_tasks):
            raise RuntimeError(
                f"generate_for_model returned {len(all_outs)} rows for {len(gen_tasks)} tasks"
            )
        bench_outs = all_outs[: len(bench_tasks)]
        mbpp = evaluate_benchmark(bench_tasks, bench_outs)
        result["mbpp_plus_pass1"] = mbpp["pass_at_1"]
        result["avg_code_length"] = mbpp["avg_code_length"]
        result["syntax_error_rate"] = mbpp["syntax_error_rate"]
        if lcb_eval:
            lcb_outs = all_outs[len(bench_tasks) :]
            lcb = evaluate_benchmark(lcb_eval, lcb_outs)
            result["lcb_easy_pass1"] = lcb["pass_at_1"]
            result["lcb_ready_frac"] = lcb_frac
        else:
            result["lcb_easy_pass1"] = None

    if need_read:
        read_is_bench_prefix = (
            bench_outs is not None
            and len(read_tasks) <= len(bench_tasks)
            and read_tasks == bench_tasks[: len(read_tasks)]
        )
        if read_is_bench_prefix:
            _log(
                f"[info] reusing first {len(read_tasks)} {model_tag} bench "
                "completions for readability (same generation contract)"
            )
            outs = bench_outs[: len(read_tasks)]
        else:
            outs = generate_for_model(
                model_tag,
                model_path,
                base_model_path,
                read_tasks,
                **gen_kwargs,
            )
        read = evaluate_readability(
            read_tasks,
            outs,
            mock_judge=mock_judge,
            model_tag=model_tag,
            judge_max_concurrent=judge_max_concurrent,
        )
        result["readability_overall"] = read["overall"]
        result["readability_detail"] = {
            "clarity": read["clarity"],
            "conciseness": read["conciseness"],
            "informativeness": read["informativeness"],
            "naturalness": read["naturalness"],
        }
        result["avg_collab_length"] = read["avg_collab_length"]
        result["think_leak_rate"] = read["think_leak_rate"]

    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate models (GOAL Step 4)")
    parser.add_argument(
        "--mode",
        choices=["full", "hypothesis_check", "benchmark", "readability"],
        default="full",
    )
    parser.add_argument("--models", default="base,rl,final", help="Comma-separated model tags")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--rl_model", default=None)
    parser.add_argument("--final_model", default=None)
    parser.add_argument("--eval_data", type=Path, default=ROOT / "data" / "mbpp_plus_test.jsonl")
    parser.add_argument("--lcb_data", type=Path, default=ROOT / "data" / "lcb_easy.jsonl")
    parser.add_argument("--num_tasks", type=int, default=None, help="Alias for both caps")
    parser.add_argument("--num_tasks_benchmark", type=int, default=200)
    parser.add_argument("--num_tasks_readability", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--thinking_budget_tokens", type=int, default=256)
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=4,
        help="HF generate batch size per GPU (left-padded)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Data-parallel GPU workers; 0 = all visible devices",
    )
    parser.add_argument("--judge_max_concurrent", type=int, default=5)
    parser.add_argument("--judge_api", default="deepseek")
    parser.add_argument("--mock_judge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--allow_synthetic",
        action="store_true",
        help="Allow synthetic/example fixtures (smoke only). Implied by --dry_run.",
    )
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.num_tasks is not None:
        args.num_tasks_benchmark = args.num_tasks
        args.num_tasks_readability = min(args.num_tasks, args.num_tasks_readability)
    if not 0 < args.thinking_budget_tokens < args.max_new_tokens:
        raise SystemExit(
            "[error] --thinking_budget_tokens must be positive and smaller than "
            f"--max_new_tokens; got {args.thinking_budget_tokens} and "
            f"{args.max_new_tokens}"
        )

    # mock_judge only when explicitly requested — dry_run alone must not imply
    # fake readability scores unless --mock_judge / --judge_api mock is set.
    mock_judge = bool(args.mock_judge or args.judge_api == "mock")
    if args.dry_run and not mock_judge and args.mode in ("full", "hypothesis_check", "readability"):
        raise SystemExit(
            "[error] --dry_run evaluation of readability still needs an explicit "
            "--mock_judge (or --judge_api mock). Refusing to silently fabricate "
            "judge scores."
        )

    allow_synthetic = bool(args.allow_synthetic or args.dry_run)

    from utils.benchmarks import require_real_benchmark
    from utils.failfast import ConfigError

    try:
        require_real_benchmark(
            "mbpp_plus", Path(args.eval_data), allow_synthetic=allow_synthetic
        )
    except ConfigError as e:
        if allow_synthetic and Path(args.eval_data).exists():
            print(f"[warn] eval_data failed real-benchmark gate (allowed for dry_run): {e}")
        else:
            raise SystemExit(f"[error] {e}") from e

    mbpp_tasks = load_jsonl(args.eval_data)
    lcb_tasks = load_jsonl(args.lcb_data) if args.lcb_data else []
    if not mbpp_tasks:
        raise SystemExit(
            f"[error] eval_data is empty/missing: {args.eval_data}. "
            "Run: python src/prepare_data.py --download"
        )
    if args.mode == "full":
        if not args.lcb_data:
            raise SystemExit(
                "[error] --mode full requires --lcb_data pointing at real "
                "LiveCodeBench-easy jsonl (GOAL secondary benchmark)."
            )
        try:
            require_real_benchmark(
                "lcb_easy", Path(args.lcb_data), allow_synthetic=allow_synthetic
            )
        except ConfigError as e:
            if allow_synthetic and Path(args.lcb_data).exists() and lcb_tasks:
                print(f"[warn] lcb_data failed real-benchmark gate (allowed for dry_run): {e}")
            elif allow_synthetic:
                print(
                    "[warn] dry_run full mode without real LCB — "
                    "lcb_easy_pass1 will be null; not a valid GOAL check."
                )
            else:
                raise SystemExit(f"[error] {e}") from e
        lcb_tasks = load_jsonl(args.lcb_data)

    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    if args.mode == "hypothesis_check":
        tags = [t for t in tags if t in ("base", "rl")] or ["base", "rl"]

    path_map = {
        "base": args.base_model,
        "rl": args.rl_model,
        "final": args.final_model,
    }

    models_out: dict[str, Any] = {}
    for tag in tags:
        _log(f"Evaluating model={tag} path={path_map.get(tag)}")
        models_out[tag] = evaluate_model(
            tag,
            path_map.get(tag),
            args.base_model,
            mbpp_tasks,
            lcb_tasks,
            mode=args.mode,
            dry_run=args.dry_run,
            mock_judge=mock_judge,
            num_tasks_benchmark=args.num_tasks_benchmark,
            num_tasks_readability=args.num_tasks_readability,
            max_new_tokens=args.max_new_tokens,
            thinking_budget_tokens=args.thinking_budget_tokens,
            gen_batch_size=args.gen_batch_size,
            num_gpus=args.num_gpus,
            judge_max_concurrent=args.judge_max_concurrent,
        )

    payload = summarize_eval_results(cycle=args.cycle, models=models_out)
    payload["mode"] = args.mode

    hypothesis = payload.get("hypothesis_results") or {}
    if args.mode == "hypothesis_check" and "base" in models_out and "rl" in models_out:
        payload["benchmark"] = {
            "base_pass_rate": models_out["base"].get("mbpp_plus_pass1"),
            "rl_pass_rate": models_out["rl"].get("mbpp_plus_pass1"),
            "delta": (hypothesis.get("H1_rl_improves_coding") or {}).get("delta"),
        }
        payload["readability"] = {
            "base_score": models_out["base"].get("readability_overall"),
            "rl_score": models_out["rl"].get("readability_overall"),
            "delta": (hypothesis.get("H2_rl_hurts_readability") or {}).get("delta"),
        }
        payload["hypothesis_1_verified"] = (hypothesis.get("H1_rl_improves_coding") or {}).get(
            "verified"
        )
        payload["comment"] = (
            "RL improves coding but hurts readability"
            if (hypothesis.get("H1_rl_improves_coding") or {}).get("verified")
            and (hypothesis.get("H2_rl_hurts_readability") or {}).get("verified")
            else "See hypothesis_results"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _log(f"Wrote evaluation → {args.output}")


if __name__ == "__main__":
    main()
