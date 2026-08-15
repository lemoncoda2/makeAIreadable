import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filter_pairs import filter_structural, main as filter_main, structural_keep  # noqa: E402
from utils.harvest_rollouts import (  # noqa: E402
    find_reward_audit_dir,
    harvest_rollouts,
)


def _audit_line(task_id, reward, call, completion):
    return {
        "call": call,
        "task_id": task_id,
        "reward": reward,
        "completion": completion,
    }


def _fenced(think, collab, code):
    return f"<think>\n{think}\n</think>\n\n{collab}\n\n```python\n{code}\n```\n"


def _write_audit(tmp_path: Path, rows):
    audit = tmp_path / "model_rl" / "reward_audit"
    audit.mkdir(parents=True)
    (audit / "reward_rank0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    return audit


def _write_tasks(tmp_path: Path, ids=("mbpp_1", "mbpp_2")):
    tasks = tmp_path / "mbpp_train.jsonl"
    tasks.write_text(
        "".join(
            json.dumps({"task_id": tid, "prompt": f"prompt for {tid}"}) + "\n"
            for tid in ids
        ),
        encoding="utf-8",
    )
    return tasks


def test_harvest_keeps_positive_later_rollouts(tmp_path):
    rows = [
        _audit_line(
            "mbpp_1",
            0.0,
            0,
            _fenced("early fail", "bad", "def a():\n    return 0"),
        ),
        _audit_line(
            "mbpp_1",
            1.0,
            1,
            _fenced("early ok", "early collab", "def a():\n    return 1"),
        ),
        _audit_line(
            "mbpp_1",
            1.0,
            8,
            _fenced("late ok", "late collab", "def a():\n    return 2"),
        ),
        _audit_line(
            "mbpp_2",
            0.5,
            9,
            _fenced("late ok 2", "late collab 2", "def b():\n    return 3"),
        ),
    ]
    audit = _write_audit(tmp_path, rows)
    tasks = _write_tasks(tmp_path)
    traces, stats = harvest_rollouts(
        audit, tasks, later_frac=1.0 / 3.0, min_later_traces=1
    )
    assert stats["used_later_window"] is True
    ids_codes = {(t["task_id"], t["code"]) for t in traces}
    assert ("mbpp_1", "def a():\n    return 2") in ids_codes
    assert ("mbpp_2", "def b():\n    return 3") in ids_codes
    assert all(t["reward"] > 0 for t in traces)
    assert all(t["source"] == "grpo_reward_audit" for t in traces)
    assert all(t["task_prompt"].startswith("prompt for") for t in traces)


def test_harvest_falls_back_if_later_window_thin(tmp_path):
    rows = [
        _audit_line(
            "mbpp_1",
            1.0,
            0,
            _fenced("early", "collab", "def a():\n    return 1"),
        ),
        _audit_line(
            "mbpp_1",
            0.0,
            9,
            _fenced("late fail", "x", "def a():\n    return 0"),
        ),
    ]
    audit = _write_audit(tmp_path, rows)
    tasks = _write_tasks(tmp_path)
    traces, stats = harvest_rollouts(
        audit, tasks, later_frac=1.0 / 3.0, min_later_traces=8
    )
    assert stats["fallback_all_positive"] is True
    assert len(traces) == 1
    assert traces[0]["code"] == "def a():\n    return 1"


def test_harvest_dedups_same_code_keeps_later_call(tmp_path):
    code = "def a():\n    return 1"
    rows = [
        _audit_line("mbpp_1", 1.0, 1, _fenced("t1", "old", code)),
        _audit_line("mbpp_1", 1.0, 5, _fenced("t2", "new", code)),
    ]
    audit = _write_audit(tmp_path, rows)
    tasks = _write_tasks(tmp_path)
    traces, _ = harvest_rollouts(audit, tasks, later_frac=1.0, min_later_traces=1)
    assert len(traces) == 1
    assert traces[0]["collaboration"] == "new"
    assert traces[0]["audit_call"] == 5


def test_find_reward_audit_from_adapter_dir(tmp_path):
    _write_audit(tmp_path, [_audit_line("mbpp_1", 1.0, 0, _fenced("t", "c", "def a():\n  return 1"))])
    found = find_reward_audit_dir(tmp_path / "model_rl")
    assert found is not None
    assert found.name == "reward_audit"


def test_structural_keep_and_filter():
    good = {
        "code": "def a():\n    return 1",
        "regen_collaboration": "rewritten clearly",
        "rl_collaboration": "ok done",
        "task_id": "t1",
        "task_prompt": "p",
        "thinking": "th",
    }
    assert structural_keep(good) is True
    same = dict(good, regen_collaboration="ok done")
    assert structural_keep(same) is False
    empty = dict(good, code="")
    assert structural_keep(empty) is False
    kept = filter_structural([good, same, empty])
    assert len(kept) == 1
    assert kept[0]["chosen"] == "rewritten clearly"
    assert kept[0]["rejected"] == "ok done"
    assert kept[0]["metadata"]["regen_score"] is None


def test_judge_api_deepseek_is_structural(tmp_path):
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "filt.jsonl"
    raw.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "task_prompt": "p",
                "thinking": "t",
                "code": "def a():\n    return 1",
                "rl_collaboration": "short",
                "regen_collaboration": "a clearer rewrite",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    filter_main(
        [
            "--raw_pairs",
            str(raw),
            "--output",
            str(out),
            "--judge_api",
            "deepseek",
            "--min_pairs",
            "1",
        ]
    )
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(rows) == 1
    assert rows[0]["metadata"]["score_gap"] is None
