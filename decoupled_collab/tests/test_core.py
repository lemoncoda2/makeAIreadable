"""Unit tests for decoupled_collab utilities and data prep (no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_data import (  # noqa: E402
    extract_function_name,
    prepare_eval_tasks,
    prepare_grpo_tasks,
    prepare_lcb_easy,
)
from utils.api_judge import filter_pair, mock_judge_scores, parse_json_score  # noqa: E402
from utils.code_executor import compute_reward, execute_test, separate_output  # noqa: E402
from utils.metrics import hypothesis_check  # noqa: E402
from utils.prompts import build_dpo_prompt  # noqa: E402


def test_extract_function_name_ast():
    assert extract_function_name("def foo(x):\n    return x\n") == "foo"


def test_extract_function_name_regex_fallback():
    assert extract_function_name("def bar(a):\n  return a") == "bar"


def test_separate_output():
    out = "<think>plan</think>\nHello user.\n```python\ndef f():\n    return 1\n```\nDone."
    sep = separate_output(out)
    assert sep["thinking"] == "plan"
    assert "def f()" in sep["code"]
    assert "Hello user" in sep["collaboration"]
    assert "def f()" not in sep["collaboration"]


def test_compute_reward_pass_fail():
    good = "```python\ndef add(a,b):\n    return a+b\n```"
    bad = "```python\ndef add(a,b):\n    return a-b\n```"
    tests = ["assert add(1,2)==3"]
    assert compute_reward(good, tests) == 1.0
    assert compute_reward(bad, tests) == 0.0


def test_execute_test():
    assert execute_test("def add(a,b): return a+b", "assert add(1,2)==3") is True


def test_parse_json_score_and_filter():
    scores = parse_json_score(
        '{"clarity": 8, "conciseness": 7, "informativeness": 8, "naturalness": 7, "overall": 7.5}'
    )
    assert scores["overall"] == 7.5
    regen, rl = mock_judge_scores("clear text", "ok done")
    assert filter_pair(regen, rl, threshold=6.0) is True
    same_a, same_b = mock_judge_scores("same", "same")
    assert filter_pair(same_a, same_b, threshold=6.0) is False


def test_build_dpo_prompt():
    p = build_dpo_prompt("add numbers", "think", "def add(a,b): return a+b")
    assert "add numbers" in p and "think" in p and "def add" in p


def test_hypothesis_results():
    base = {"mbpp_plus_pass1": 0.74, "readability_overall": 7.2}
    rl = {"mbpp_plus_pass1": 0.82, "readability_overall": 5.5}
    final = {"mbpp_plus_pass1": 0.81, "readability_overall": 7.6}
    h = hypothesis_check(base, rl, final)
    assert h["H1_rl_improves_coding"]["verified"] is True
    assert h["H2_rl_hurts_readability"]["verified"] is True
    assert h["H3_dpo_recovers_readability"]["verified"] is True
    assert h["H4_dpo_preserves_coding"]["verified"] is True


def test_prepare_data_from_fake_disk(tmp_path, monkeypatch):
    import prepare_data as pd

    fake_full = {
        "train": [
            {
                "task_id": 1,
                "text": "add",
                "test_list": ["assert add(1,2)==3"],
                "code": "def add(a,b): return a+b",
            }
        ]
    }
    fake_san = {
        "test": [
            {
                "task_id": 2,
                "text": "sub",
                "test_list": ["assert sub(3,1)==2"],
                "code": "def sub(a,b): return a-b",
            }
        ]
    }

    def fake_load(path, download, config_name):
        return fake_full if config_name == "full" else fake_san

    monkeypatch.setattr(pd, "_load_mbpp", fake_load)
    train_out = tmp_path / "train.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    prepare_grpo_tasks(tmp_path / "full", train_out, download=False)
    prepare_eval_tasks(tmp_path / "san", eval_out, download=False)
    assert train_out.read_text().strip()
    assert "mbpp_2" in eval_out.read_text()
    assert json.loads(eval_out.read_text().strip())["entry_point"] == "sub"


def test_prepare_lcb_easy_placeholder(tmp_path):
    out = tmp_path / "lcb.jsonl"
    with pytest.warns(UserWarning):
        tasks = prepare_lcb_easy(out)
    assert out.exists()
    assert isinstance(tasks, list)


def test_run_pipeline_phases_from():
    from run_pipeline import PHASES, phases_from

    assert phases_from("phase3_filter")[0] == "phase3_filter"
    assert phases_from("not_a_real_phase") == PHASES
