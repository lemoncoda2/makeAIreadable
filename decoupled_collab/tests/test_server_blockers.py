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


def test_grpo_reward_audit_records_expanded_task_ids(tmp_path):
    from train_grpo import build_reward_func

    reward_fn = build_reward_func(
        timeout=1, max_test_cases=1, audit_dir=tmp_path / "audit"
    )
    rewards = reward_fn(
        completions=[
            "</think>\n```python\ndef f(): return 1\n```",
            "</think>\n```python\ndef f(): return 0\n```",
        ],
        test_cases=[["assert f() == 1"]],
        task_id=["mbpp_1"],
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "audit" / "reward_rank0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rewards == [1.0, 0.0]
    assert [row["task_id"] for row in rows] == ["mbpp_1", "mbpp_1"]
    assert [row["reward"] for row in rows] == rewards
    assert all(row["has_think_end"] for row in rows)
    assert all(row["has_python_fence"] for row in rows)


def test_pass_at_1_empty_tests_is_false():
    from evaluate import pass_at_1

    assert pass_at_1("```python\ndef f():\n    return 1\n```", []) is False


def test_generate_for_model_refuses_missing_path_outside_dry_run():
    from evaluate import generate_for_model

    with pytest.raises(SystemExit, match="missing"):
        generate_for_model("rl", None, None, [{"prompt": "x"}], dry_run=False)


def test_generate_for_model_passes_public_interface_and_budgets(monkeypatch):
    import evaluate

    seen = {}

    class Runner:
        def __init__(self, model_path, base_model_path=None, thinking_budget_tokens=0):
            seen["init"] = (model_path, base_model_path, thinking_budget_tokens)

        def generate(self, prompt, public_test_cases, max_new_tokens):
            seen["generate"] = (prompt, public_test_cases, max_new_tokens)
            return "ok"

    monkeypatch.setattr(evaluate, "ModelRunner", Runner)
    outputs = evaluate.generate_for_model(
        "base",
        "base-model",
        None,
        [{"prompt": "write f", "test_cases": ["assert f() == 1", "hidden"]}],
        dry_run=False,
        max_new_tokens=704,
        thinking_budget_tokens=256,
    )

    assert outputs == ["ok"]
    assert seen["init"] == ("base-model", None, 256)
    assert seen["generate"] == ("write f", ["assert f() == 1"], 704)


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


def test_pipeline_grpo_train_file_override(tmp_path):
    from run_pipeline import resolve_grpo_train_file

    cfg = {"grpo": {"train_file": "./data/smoke/mbpp_train.jsonl"}}
    assert resolve_grpo_train_file(cfg, tmp_path) == (
        tmp_path / "data" / "smoke" / "mbpp_train.jsonl"
    ).resolve()


def test_prepare_lcb_from_local_jsonl_drops_private_tests(tmp_path):
    from prepare_data import prepare_lcb_easy

    source = tmp_path / "test.jsonl"
    output = tmp_path / "lcb_easy.jsonl"
    rows = []
    for index in range(20):
        rows.append(
            {
                "question_id": str(index),
                "question_content": f"Solve task {index}",
                "difficulty": "easy",
                "public_test_cases": json.dumps(
                    [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
                ),
                "private_test_cases": "x" * 1000,
                "metadata": "{}",
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    prepared = prepare_lcb_easy(output, source_jsonl=[source])

    assert len(prepared) == 20
    assert all("private_test_cases" not in row for row in prepared)
    assert all(row["source"] == "livecodebench_easy" for row in prepared)
    assert output.stat().st_size < source.stat().st_size


def test_legacy_grpo_sampler_signature_is_adapted():
    from train_grpo import get_compatible_grpo_trainer_class

    class LegacyTrainer:
        def __init__(self):
            self.train_dataset = ["original"]

        def _get_train_sampler(self):
            return tuple(self.train_dataset)

    compatible = get_compatible_grpo_trainer_class(LegacyTrainer)
    trainer = compatible()

    assert trainer._get_train_sampler() == ("original",)
    assert trainer._get_train_sampler(["prepared"]) == ("prepared",)
    assert trainer.train_dataset == ["original"]


def test_modern_grpo_sampler_signature_is_unchanged():
    from train_grpo import get_compatible_grpo_trainer_class

    class ModernTrainer:
        def _get_train_sampler(self, train_dataset=None):
            return train_dataset

    assert get_compatible_grpo_trainer_class(ModernTrainer) is ModernTrainer


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


def test_qwen3_coding_prompt_uses_single_user_turn():
    from utils.prompts import build_coding_messages

    messages = build_coding_messages(
        "Write function f.", ["assert f(2) == 4", "assert f(3) == 6"]
    )

    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"].startswith("Write function f.")
    assert "fenced Python code block" in messages[0]["content"]
    assert "assert f(2) == 4" in messages[0]["content"]
    assert "assert f(3) == 6" not in messages[0]["content"]


def test_qwen_think_end_token_must_be_single_token():
    from utils.thinking_budget import qwen_think_end_token_id

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "</think>"
            assert add_special_tokens is False
            return [151668]

    assert qwen_think_end_token_id(Tokenizer()) == 151668

    class BrokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(RuntimeError, match="exactly one token"):
        qwen_think_end_token_id(BrokenTokenizer())


def test_thinking_budget_processor_forces_only_open_rows():
    torch = pytest.importorskip("torch")
    from utils.thinking_budget import build_thinking_budget_logits_processor

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [7]

    processor = build_thinking_budget_logits_processor(
        Tokenizer(), prompt_length=2, thinking_budget_tokens=3
    )
    input_ids = torch.tensor(
        [
            [90, 91, 1, 2, 3],
            [90, 91, 1, 7, 3],
        ]
    )
    scores = torch.zeros((2, 10))

    result = processor(input_ids, scores)

    assert torch.isneginf(result[0]).sum().item() == 9
    assert result[0, 7].item() == 0.0
    assert torch.equal(result[1], scores[1])


def test_code_fence_stopping_waits_for_post_think_closing_fence():
    torch = pytest.importorskip("torch")
    from utils.thinking_budget import build_code_fence_stopping_criteria

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"</think>": [7], "```": [8]}[text]

    stopping = build_code_fence_stopping_criteria(Tokenizer(), prompt_length=2)
    input_ids = torch.tensor(
        [
            [90, 91, 7, 8, 1, 8, 0],  # post-think open + close fence
            [90, 91, 8, 7, 8, 1, 0],  # fence in thought + opening fence only
            [90, 91, 7, 8, 1, 2, 0],  # opening fence only
        ]
    )

    assert stopping(input_ids, torch.zeros((3, 10))).tolist() == [True, False, False]


def test_completion_mask_ends_at_token_id_closing_fence():
    torch = pytest.importorskip("torch")
    from utils.thinking_budget import mask_completion_after_code_fence

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"</think>": [7], "```": [8]}[text]

    completion_ids = torch.tensor([[7, 8, 1, 8, 99, 99, 99], [7, 8, 1, 2, 3, 4, 5]])
    mask = mask_completion_after_code_fence(
        completion_ids, torch.ones_like(completion_ids), Tokenizer()
    )

    assert mask.tolist() == [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]


def test_masked_grpo_objective_ignores_forced_token_before_exp():
    torch = pytest.importorskip("torch")
    from train_grpo import compute_masked_grpo_objective

    policy = torch.zeros((1, 3), requires_grad=True)
    reference = torch.tensor([[0.0, 1000.0, 0.0]])
    loss_mask = torch.tensor([[1, 0, 1]])
    loss, kl = compute_masked_grpo_objective(
        per_token_logps=policy,
        ref_per_token_logps=reference,
        advantages=torch.tensor([0.0]),
        loss_mask=loss_mask,
        beta=0.04,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)
    assert kl[0, 1].item() == 0.0


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
                "code": "def a():\n    return 1",
                "rl_collaboration": "short",
                "regen_collaboration": "a clearer rewrite",
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
                "--strict_min_pairs",
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

    import yaml

    cfg = yaml.safe_load(
        (_ROOT / "configs" / "smoke_pipeline_config.yaml").read_text(encoding="utf-8")
    )
    general = cfg["general"]
    for key in (
        "checkpoint_root",
        "log_root",
        "trace_root",
        "dpo_pairs_root",
        "results_root",
        "state_file",
    ):
        assert "smoke" in general[key], f"{key} must be isolated for smoke runs"


def test_pipeline_model_paths_respect_checkpoint_root(tmp_path):
    from run_pipeline import get_model_path

    cfg = {
        "_project_root": str(tmp_path),
        "_start_cycle": 0,
        "general": {
            "base_model": "./models/base",
            "checkpoint_root": "./checkpoints/smoke",
        },
    }

    assert get_model_path(0, "phase3_dpo", cfg, start_model=None) == str(
        (tmp_path / "checkpoints" / "smoke" / "cycle_0" / "model_rl").resolve()
    )
