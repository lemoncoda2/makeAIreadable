#!/usr/bin/env python3
"""Evaluate base / RL / final models (GOAL Step 4).

Modes: full | hypothesis_check | benchmark | readability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.api_judge import judge_collaboration, mock_judge_scores  # noqa: E402
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
    """True if reward == 1.0 (all capped tests pass) or no tests with extractable code."""
    if not test_cases:
        return bool(extract_code(model_output))
    # Use full test list for eval (not capped to 5)
    from utils.code_executor import execute_test

    code = extract_code(model_output)
    if not code:
        return False
    return all(execute_test(code, tc, timeout) for tc in test_cases)


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


class ModelRunner:
    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
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
        base = load_causal_lm(tok_src, torch_dtype="float16", device_map="auto")
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
        self._apply = apply_chat_template_with_thinking
        self._device = model_input_device(self.model)

    def generate(self, prompt: str, max_new_tokens: int = 768, temperature: float = 0.2) -> str:
        import torch

        messages = build_coding_messages(prompt)
        text = self._apply(self.tokenizer, messages, enable_thinking=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=False)


def evaluate_benchmark(tasks: list[dict[str, Any]], outputs: list[str]) -> dict[str, float]:
    passes = [pass_at_1(o, t.get("test_cases") or []) for o, t in zip(outputs, tasks)]
    codes = [extract_code(o) for o in outputs]
    return {
        "pass_at_1": mean(float(p) for p in passes),
        "avg_code_length": mean(float(len(c)) for c in codes),
        "syntax_error_rate": syntax_error_rate(codes),
        "n": float(len(tasks)),
    }


def evaluate_readability(
    tasks: list[dict[str, Any]],
    outputs: list[str],
    *,
    mock_judge: bool,
    model_tag: str,
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
        else:
            scores.append(judge_collaboration(task.get("prompt", ""), collab or "(empty)"))

    detail = aggregate_readability_scores(scores)
    return {
        **detail,
        "avg_collab_length": avg_length_tokens(collabs),
        "think_leak_rate": mean(leaks),
        "n": float(len(tasks)),
    }


def generate_for_model(
    model_tag: str,
    model_path: Optional[str],
    base_model_path: Optional[str],
    tasks: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> list[str]:
    if dry_run or not model_path:
        return [dry_run_generation(t, model_tag) for t in tasks]
    runner = ModelRunner(model_path, base_model_path=base_model_path)
    return [runner.generate(t["prompt"]) for t in tasks]


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
) -> dict[str, Any]:
    bench_tasks = mbpp_tasks[:num_tasks_benchmark]
    read_tasks = mbpp_tasks[:num_tasks_readability]
    lcb_eval = lcb_tasks[:num_tasks_benchmark] if lcb_tasks else []

    result: dict[str, Any] = {}
    need_bench = mode in ("full", "hypothesis_check", "benchmark")
    need_read = mode in ("full", "hypothesis_check", "readability")

    if need_bench:
        outs = generate_for_model(
            model_tag, model_path, base_model_path, bench_tasks, dry_run=dry_run
        )
        mbpp = evaluate_benchmark(bench_tasks, outs)
        result["mbpp_plus_pass1"] = mbpp["pass_at_1"]
        result["avg_code_length"] = mbpp["avg_code_length"]
        result["syntax_error_rate"] = mbpp["syntax_error_rate"]
        if lcb_eval:
            lcb_outs = generate_for_model(
                model_tag, model_path, base_model_path, lcb_eval, dry_run=dry_run
            )
            lcb = evaluate_benchmark(lcb_eval, lcb_outs)
            result["lcb_easy_pass1"] = lcb["pass_at_1"]
        else:
            result["lcb_easy_pass1"] = None

    if need_read:
        outs = generate_for_model(
            model_tag, model_path, base_model_path, read_tasks, dry_run=dry_run
        )
        read = evaluate_readability(
            read_tasks, outs, mock_judge=mock_judge, model_tag=model_tag
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
        print(f"Evaluating model={tag} path={path_map.get(tag)}")
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
    print(f"Wrote evaluation → {args.output}")


if __name__ == "__main__":
    main()
