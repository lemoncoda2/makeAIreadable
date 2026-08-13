"""
Safe code extraction and execution for GRPO reward and eval.

Reward evaluates only code correctness — collaboration text is ignored.
This keeps RL focused on the work layer; collaboration drift is a side effect.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence


def extract_code(output: str) -> str:
    """Extract executable Python from model output.

    Strips ``<think>...</think>``, prefers fenced ``python`` blocks,
    then falls back to a ``def ...`` span.
    """
    if not output:
        return ""

    output_no_think = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

    code_blocks = re.findall(
        r"```(?:python)?\s*\n(.*?)```", output_no_think, re.DOTALL
    )
    if code_blocks:
        return "\n".join(code_blocks).strip()

    func_match = re.search(
        r"(def \w+.*?)(?:\n\n|\Z)", output_no_think, re.DOTALL
    )
    if func_match:
        return func_match.group(1).strip()

    return ""


def execute_test(code: str, test_case: str, timeout: int = 10) -> bool:
    """Run ``code`` plus one assert-style ``test_case`` in a subprocess.

    Returns True only if the process exits with code 0 within ``timeout``.
    """
    full_code = code + "\n\n" + test_case
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, Exception):
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def safe_execute(code: str, test_case: str, timeout: int = 5) -> bool:
    """Alias for :func:`execute_test` used by Phase 0.7 sandbox checks."""
    return execute_test(code, test_case, timeout=timeout)


def compute_reward(
    model_output: str,
    test_cases: Sequence[str],
    timeout: int = 10,
    max_test_cases: int = 5,
) -> float:
    """Extract code from ``model_output`` and score against up to 5 tests.

    Returns a float in ``[0.0, 1.0]`` equal to
    ``passed / min(len(test_cases), max_test_cases)``.
    Empty extraction yields ``0.0``.
    """
    if not test_cases:
        return 0.0

    code = extract_code(model_output)
    if not code:
        return 0.0

    limit = min(len(test_cases), max_test_cases)
    passed = sum(
        1 for tc in list(test_cases)[:limit] if execute_test(code, tc, timeout)
    )
    return passed / float(limit)


def separate_output(output: str) -> dict:
    """Deterministically split Qwen3 thinking-mode output into layers.

    Rules (GOAL Phase 2):
    - Content inside ``<think>...</think>`` → ``thinking`` (work)
    - Content inside fenced code blocks → ``code`` (work)
    - Everything else → ``collaboration`` (user-facing)
    """
    if not output:
        return {"thinking": "", "code": "", "collaboration": ""}

    think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    without_think = re.sub(
        r"<think>.*?</think>", "", output, flags=re.DOTALL
    ).strip()

    code_blocks = re.findall(
        r"```(?:python)?\s*\n(.*?)```", without_think, re.DOTALL
    )
    code = "\n".join(code_blocks).strip()

    collaboration = re.sub(
        r"```(?:python)?\s*\n.*?```", "", without_think, flags=re.DOTALL
    ).strip()

    return {
        "thinking": thinking,
        "code": code,
        "collaboration": collaboration,
    }


def batch_compute_rewards(
    outputs: List[str],
    test_cases_batch: List[list],
    timeout: int = 10,
    max_test_cases: int = 5,
) -> List[float]:
    """Vector helper used by GRPO reward_func."""
    rewards: List[float] = []
    for output, tcs in zip(outputs, test_cases_batch):
        rewards.append(
            compute_reward(
                output,
                tcs or [],
                timeout=timeout,
                max_test_cases=max_test_cases,
            )
        )
    return rewards
