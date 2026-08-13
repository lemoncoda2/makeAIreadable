"""Smoke tests for utils — no GPU or external API required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `pytest` from project root without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.code_executor import (  # noqa: E402
    compute_reward,
    extract_code,
    safe_execute,
    separate_output,
)
from utils.api_judge import filter_pair, parse_json_score  # noqa: E402
from utils.metrics import syntax_error_rate, think_leak_rate  # noqa: E402


# ---------------------------------------------------------------------------
# code_executor
# ---------------------------------------------------------------------------


def test_extract_code_from_python_fence():
    output = (
        "<think>plan</think>\n"
        "Here is the solution:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    code = extract_code(output)
    assert "def add" in code
    assert "return a + b" in code
    assert "<think>" not in code


def test_extract_code_fallback_def():
    output = "<think>x</think>\ndef mul(a, b):\n    return a * b\n\nDone."
    code = extract_code(output)
    assert code.startswith("def mul")
    assert "return a * b" in code


def test_compute_reward_good_and_bad():
    good = (
        "<think>add</think>\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    bad = (
        "<think>wrong</think>\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "```\n"
    )
    tests = ["assert add(1, 2) == 3", "assert add(2, 3) == 5"]

    assert compute_reward(good, tests) == 1.0
    assert compute_reward(bad, tests) == 0.0
    assert compute_reward("", tests) == 0.0


def test_safe_execute_alias():
    assert safe_execute("def add(a, b): return a + b", "assert add(1, 2) == 3") is True
    assert safe_execute("def add(a, b): return a - b", "assert add(1, 2) == 3") is False


def test_separate_output_think_code_collab():
    output = (
        "<think>\n"
        "Use a simple loop over the list.\n"
        "</think>\n"
        "I'll solve this with a linear scan:\n"
        "```python\n"
        "def longest_common_prefix(strs):\n"
        "    if not strs:\n"
        "        return ''\n"
        "    prefix = strs[0]\n"
        "    for s in strs[1:]:\n"
        "        while not s.startswith(prefix):\n"
        "            prefix = prefix[:-1]\n"
        "    return prefix\n"
        "```\n"
        "This returns an empty string for empty input."
    )
    parts = separate_output(output)
    assert "linear" in parts["thinking"] or "loop" in parts["thinking"]
    assert "def longest_common_prefix" in parts["code"]
    assert "```" not in parts["collaboration"]
    assert "<think>" not in parts["collaboration"]
    assert "linear scan" in parts["collaboration"]
    assert "empty string" in parts["collaboration"]


# ---------------------------------------------------------------------------
# api_judge helpers (offline)
# ---------------------------------------------------------------------------


def test_parse_json_score_clean():
    text = (
        '{"clarity": 8, "conciseness": 7, "informativeness": 8, '
        '"naturalness": 9, "overall": 8}'
    )
    score = parse_json_score(text)
    assert score["overall"] == 8.0
    assert score["clarity"] == 8.0


def test_parse_json_score_fenced_and_prose():
    text = (
        "Here is my rating:\n"
        "```json\n"
        '{"clarity": 6.5, "conciseness": 7, "informativeness": 6, '
        '"naturalness": 7, "overall": 6.5}\n'
        "```\n"
        "Thanks."
    )
    score = parse_json_score(text)
    assert score["overall"] == 6.5
    assert score["conciseness"] == 7.0


def test_parse_json_score_missing_overall_averages():
    text = '{"clarity": 8, "conciseness": 6, "informativeness": 8, "naturalness": 6}'
    score = parse_json_score(text)
    assert score["overall"] == pytest.approx(7.0)


def test_filter_pair_logic():
    regen = {"overall": 7.5}
    rl_ok = {"overall": 5.0}
    rl_close = {"overall": 7.2}
    rl_better = {"overall": 8.0}
    regen_low = {"overall": 5.5}

    assert filter_pair(regen, rl_ok) is True
    assert filter_pair(regen_low, rl_ok) is False  # below threshold 6.0
    assert filter_pair(regen, rl_better) is False  # regen not better
    assert filter_pair(regen, rl_close) is False  # gap < 0.5
    assert filter_pair(regen, rl_ok, threshold=8.0) is False
    assert filter_pair(regen, {"overall": 6.9}, min_gap=1.0) is False


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_think_leak_rate():
    thinking = "dynamic programming table optimal substructure recurrence"
    # Heavy leak: many shared ngrams
    collab_leak = (
        "I used dynamic programming table with optimal substructure recurrence"
    )
    collab_clean = "Here is a short explanation of the approach for the user."

    leak = think_leak_rate(thinking, collab_leak, n=3)
    clean = think_leak_rate(thinking, collab_clean, n=3)
    assert leak > clean
    assert leak > 0.0
    assert clean == 0.0
    assert think_leak_rate("", "anything") == 0.0


def test_syntax_error_rate():
    good = ["def f():\n    return 1\n", "x = 1 + 2\n"]
    mixed = ["def f():\n    return 1\n", "def broken(\n", "y = [\n"]
    assert syntax_error_rate(good) == 0.0
    assert syntax_error_rate(mixed) == pytest.approx(2 / 3)
    assert syntax_error_rate([]) == 0.0
