"""LCB stdin/call harness + DPO chat-template alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_parse_public_test_cases_stdin_and_call():
    from utils.lcb_executor import parse_public_test_cases

    raw = json.dumps(
        [
            {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
            {
                "input": "[1,2]\n3",
                "output": "6",
                "testtype": "functional",
                "fn_name": "add",
            },
        ]
    )
    cases = parse_public_test_cases(raw)
    assert cases[0]["type"] == "stdin"
    assert cases[1]["type"] == "call"
    assert cases[1]["fn_name"] == "add"


def test_lcb_stdin_pass_and_fail():
    from utils.lcb_executor import pass_lcb

    task = {
        "harness": "lcb",
        "lcb_tests": [
            {"type": "stdin", "input": "1 2\n", "output": "3\n"},
            {"type": "stdin", "input": "10 5\n", "output": "15\n"},
        ],
    }
    good = "a,b=map(int,input().split())\nprint(a+b)\n"
    bad = "print(0)\n"
    assert pass_lcb(good, task) is True
    assert pass_lcb(bad, task) is False


def test_lcb_call_pass():
    from utils.lcb_executor import pass_lcb

    task = {
        "harness": "lcb",
        "lcb_tests": [
            {
                "type": "call",
                "fn_name": "add",
                "input": "1\n2",
                "output": "3",
            }
        ],
    }
    code = "def add(a, b):\n    return a + b\n"
    assert pass_lcb(code, task) is True


def test_evaluate_dispatches_lcb_harness():
    from evaluate import pass_task

    task = {
        "harness": "lcb",
        "lcb_tests": [{"type": "stdin", "input": "2 3\n", "output": "5\n"}],
        "test_cases": [],
    }
    out = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    assert pass_task(out, task) is True


def test_build_dpo_prompt_refuses_fake_xml():
    from utils.prompts import build_dpo_prompt

    with pytest.raises(RuntimeError, match="fake"):
        build_dpo_prompt("p", "t", "c")


def test_dpo_messages_match_regen():
    from utils.prompts import build_dpo_messages, build_regen_messages

    assert build_dpo_messages("p", "t", "c") == build_regen_messages("p", "t", "c")


def test_render_preference_row_uses_chat_template():
    from train_dpo import _render_preference_row
    from utils.prompts import DPO_PROMPT_FORMAT

    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return "<|im_start|>user\nRENDERED<|im_end|>\n<|im_start|>assistant\n"

    row = _render_preference_row(
        {
            "chosen": "good collab",
            "rejected": "bad collab",
            "metadata": {
                "task_prompt": "add two numbers",
                "thinking": "use +",
                "code": "def add(a,b): return a+b",
                "prompt_format": DPO_PROMPT_FORMAT,
            },
        },
        Tok(),
    )
    assert row["prompt"].startswith("<|im_start|>")
    assert "RENDERED" in row["prompt"]
    assert seen["kwargs"].get("enable_thinking") is False
    assert seen["kwargs"].get("add_generation_prompt") is True
    assert row["chosen"] == "good collab"
    assert "<system>" not in row["prompt"]


def test_legacy_xml_without_metadata_fails():
    from train_dpo import _render_preference_row

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "x"

    with pytest.raises(ValueError, match="Legacy fake-XML"):
        _render_preference_row(
            {
                "prompt": "<system>x</system>\n<user>y</user>\n",
                "chosen": "a",
                "rejected": "b",
                "metadata": {},
            },
            Tok(),
        )


def test_filter_to_dpo_record_stores_work_trace():
    from filter_pairs import to_dpo_record
    from utils.prompts import DPO_PROMPT_FORMAT

    rec = to_dpo_record(
        {
            "task_id": "t1",
            "task_prompt": "p",
            "thinking": "th",
            "code": "print(1)",
            "regen_collaboration": "clear",
            "rl_collaboration": "ok done",
        },
        {"overall": 8.0, "clarity": 8, "conciseness": 8, "informativeness": 8, "naturalness": 8},
        {"overall": 4.0, "clarity": 4, "conciseness": 4, "informativeness": 4, "naturalness": 4},
    )
    assert "prompt" not in rec or rec.get("prompt") in (None, "")
    assert rec["metadata"]["prompt_format"] == DPO_PROMPT_FORMAT
    assert rec["metadata"]["thinking"] == "th"
    assert rec["messages"][0]["role"] == "system"
