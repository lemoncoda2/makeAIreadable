#!/usr/bin/env python3
"""Filter DPO pairs with DeepSeek readability judge (GOAL Step 3.2)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.api_judge import (  # noqa: E402
    filter_pair,
    get_deepseek_async_client,
    judge_collaboration_async,
    mock_judge_scores,
)
from utils.prompts import DPO_PROMPT_FORMAT, build_dpo_messages  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_dpo_record(
    pair: dict[str, Any],
    regen_score: dict[str, float],
    rl_score: dict[str, float],
) -> dict[str, Any]:
    """Store work-trace fields + messages; train_dpo renders Qwen chat template.

    ``prompt`` is intentionally omitted as a final string here — baking fake XML
    caused train≠inference. train_dpo re-renders from metadata / messages.
    """
    task_prompt = pair.get("task_prompt", "")
    thinking = pair.get("thinking", "")
    code = pair.get("code", "")
    return {
        "chosen": pair.get("regen_collaboration", ""),
        "rejected": pair.get("rl_collaboration", ""),
        "messages": build_dpo_messages(task_prompt, thinking, code),
        "metadata": {
            "task_id": pair.get("task_id"),
            "task_prompt": task_prompt,
            "thinking": thinking,
            "code": code,
            "prompt_format": DPO_PROMPT_FORMAT,
            "regen_score": regen_score,
            "rl_score": rl_score,
            "reward": pair.get("reward"),
            "score_gap": float(regen_score.get("overall", 0))
            - float(rl_score.get("overall", 0)),
        },
    }


async def _score_pair_async(
    client: Any,
    pair: dict[str, Any],
    sem: asyncio.Semaphore,
    *,
    model: Optional[str],
    max_retries: int,
) -> tuple[dict[str, float], dict[str, float]]:
    async with sem:
        regen_score, rl_score = await asyncio.gather(
            judge_collaboration_async(
                pair.get("task_prompt", ""),
                pair.get("regen_collaboration", ""),
                client=client,
                model=model,
                max_retries=max_retries,
            ),
            judge_collaboration_async(
                pair.get("task_prompt", ""),
                pair.get("rl_collaboration", ""),
                client=client,
                model=model,
                max_retries=max_retries,
            ),
        )
        return regen_score, rl_score


async def filter_async(
    pairs: list[dict[str, Any]],
    *,
    threshold: float,
    max_concurrent: int,
    model: Optional[str],
    max_retries: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return (kept_records, score_error_count)."""
    client = get_deepseek_async_client()
    sem = asyncio.Semaphore(max_concurrent)
    kept: list[dict[str, Any]] = []
    score_errors = 0

    try:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            results = await asyncio.gather(
                *[
                    _score_pair_async(
                        client, p, sem, model=model, max_retries=max_retries
                    )
                    for p in batch
                ],
                return_exceptions=True,
            )
            for pair, result in zip(batch, results):
                if isinstance(result, Exception):
                    score_errors += 1
                    print(f"[warn] scoring failed for {pair.get('task_id')}: {result}")
                    continue
                regen_score, rl_score = result
                if filter_pair(regen_score, rl_score, threshold=threshold):
                    kept.append(to_dpo_record(pair, regen_score, rl_score))
            print(
                f"Processed {min(start + batch_size, len(pairs))}/{len(pairs)}; "
                f"kept {len(kept)}; score_errors={score_errors}"
            )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    return kept, score_errors


def filter_mock(pairs: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        regen_score, rl_score = mock_judge_scores(
            pair.get("regen_collaboration", ""),
            pair.get("rl_collaboration", ""),
        )
        if filter_pair(regen_score, rl_score, threshold=threshold):
            kept.append(to_dpo_record(pair, regen_score, rl_score))
    return kept


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Filter DPO pairs (GOAL Step 3.2)")
    parser.add_argument("--raw_pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge_api", default="deepseek")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument(
        "--mock_judge",
        action="store_true",
        help="Offline: synthetic scores so regen > rl when texts differ",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=1,
        help="Fail if fewer pairs kept (GOAL real runs use ~1500 via pipeline)",
    )
    parser.add_argument(
        "--max_score_error_rate",
        type=float,
        default=0.25,
        help="Fail if scoring exceptions exceed this fraction of pairs (non-mock)",
    )
    args = parser.parse_args(argv)

    pairs = load_jsonl(args.raw_pairs)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    print(f"Loaded {len(pairs)} raw pairs from {args.raw_pairs}")

    score_errors = 0
    if args.mock_judge or args.judge_api == "mock":
        kept = filter_mock(pairs, threshold=args.threshold)
    else:
        kept, score_errors = asyncio.run(
            filter_async(
                pairs,
                threshold=args.threshold,
                max_concurrent=args.max_concurrent,
                model=args.judge_model,
                max_retries=args.max_retries,
                batch_size=args.batch_size,
            )
        )
        if pairs:
            err_rate = score_errors / len(pairs)
            if err_rate > args.max_score_error_rate:
                raise SystemExit(
                    f"[error] Judge scoring error rate {err_rate:.2%} "
                    f"({score_errors}/{len(pairs)}) exceeds "
                    f"--max_score_error_rate={args.max_score_error_rate}. "
                    "Fix API/model before training on a thin filtered set."
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Kept {len(kept)}/{len(pairs)} pairs → {args.output}")
    if len(kept) < args.min_pairs:
        raise SystemExit(
            f"[error] Kept only {len(kept)} pairs < --min_pairs={args.min_pairs}. "
            "Refusing empty/thin DPO data (looks like success but trains on almost nothing)."
        )


if __name__ == "__main__":
    main()
