"""Regression tests for 上机必炸 / 假成功 / resume logic bugs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_grpo_reward_expands_test_cases_for_num_generations():
    from train_grpo import build_reward_func

    reward_fn = build_reward_func(timeout=1, max_test_cases=2)
    # Two prompts × 2 generations → 4 completions, 2 test_cases rows
    completions = [
        "```python\ndef add(a,b):\n    return a+b\n```",
        "```python\ndef add(a,b):\n    return a+b\n```",
        "```python\ndef add(a,b):\n    return 0\n```",
        "```python\ndef add(a,b):\n    return 0\n```",
    ]
    test_cases = [
        ["assert add(1,2)==3"],
        ["assert add(1,2)==3"],
    ]
    rewards = reward_fn(completions=completions, test_cases=test_cases)
    assert len(rewards) == 4
    assert rewards[0] == 1.0 and rewards[1] == 1.0
    assert rewards[2] == 0.0 and rewards[3] == 0.0


def test_grpo_reward_mismatch_not_divisible_fails():
    from train_grpo import build_reward_func

    reward_fn = build_reward_func(timeout=1, max_test_cases=2)
    with pytest.raises(RuntimeError, match="length mismatch"):
        reward_fn(
            completions=["a", "b", "c"],
            test_cases=[["assert True"], ["assert True"]],
        )


def test_pass_at_1_empty_tests_is_false():
    from evaluate import pass_at_1

    assert pass_at_1("```python\ndef f():\n    return 1\n```", []) is False


def test_generate_for_model_refuses_missing_path_outside_dry_run():
    from evaluate import generate_for_model

    with pytest.raises(SystemExit, match="missing"):
        generate_for_model("rl", None, None, [{"prompt": "x"}], dry_run=False)


def test_pipeline_advance_state_skips_completed_phase():
    from run_pipeline import PHASES, advance_state_after_phase, next_phase_after

    assert next_phase_after("phase1_grpo") == "phase1_eval"
    assert next_phase_after(PHASES[-1]) is None

    state: dict = {"current_cycle": 0, "current_phase": "phase1_grpo", "history": []}
    advance_state_after_phase(state, "phase1_grpo", 0)
    assert state["current_phase"] == "phase1_eval"
    assert state["current_cycle"] == 0

    advance_state_after_phase(state, PHASES[-1], 0)
    assert state["current_cycle"] == 1
    assert state["current_phase"] == PHASES[0]
    assert len(state["history"]) == 1


def test_enable_thinking_false_must_pass_kwarg():
    from utils.failfast import apply_chat_template_thinking_strict

    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            if "enable_thinking" not in kwargs:
                raise AssertionError("must pass enable_thinking explicitly")
            return "ok"

    out = apply_chat_template_thinking_strict(
        Tok(), [{"role": "user", "content": "hi"}], enable_thinking=False
    )
    assert out == "ok"
    assert seen.get("enable_thinking") is False


def test_fraction_executable_assert_tasks():
    from evaluate import fraction_executable_assert_tasks

    tasks = [
        {"test_cases": ["assert foo()"]},
        {"test_cases": ['{"input": "1", "output": "2"}']},
        {"test_cases": []},
    ]
    assert fraction_executable_assert_tasks(tasks) == pytest.approx(1 / 3)


def test_filter_min_pairs_fails(tmp_path):
    from filter_pairs import main

    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "filt.jsonl"
    raw.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "task_prompt": "p",
                "thinking": "t",
                "code": "c",
                "rl_collaboration": "same",
                "regen_collaboration": "same",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="min_pairs"):
        main(
            [
                "--raw_pairs",
                str(raw),
                "--output",
                str(out),
                "--mock_judge",
                "--min_pairs",
                "5",
            ]
        )


def test_smoke_dir_isolation_contract():
    """smoke_test.sh must not write canonical benchmark paths."""
    text = (_ROOT / "scripts" / "smoke_test.sh").read_text(encoding="utf-8")
    assert "data/smoke/" in text
    assert 'with open("data/mbpp_train.jsonl"' not in text
    assert "data/mbpp_train.jsonl" not in text or "data/smoke/mbpp_train.jsonl" in text
    # Ensure production paths are not opened for write in the fixture block.
    assert "open(\"data/mbpp_plus_test.jsonl\"" not in text
    assert "open(\"data/lcb_easy.jsonl\"" not in text
