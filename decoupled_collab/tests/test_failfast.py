"""Fail-fast policy tests — no GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

from utils.api_judge import parse_json_score  # noqa: E402
from utils.failfast import (  # noqa: E402
    ConfigError,
    assert_not_adapter_for_vllm,
    assert_not_dry_run_placeholder,
    validate_grpo_batch_vs_generations,
)
from utils.model_utils import apply_chat_template_with_thinking  # noqa: E402


def test_grpo_batch_not_divisible_fails():
    cfg = {
        "training": {"per_device_batch_size": 1},
        "grpo": {"num_samples_per_prompt": 6},
    }
    with pytest.raises(ConfigError, match="not divisible"):
        validate_grpo_batch_vs_generations(cfg, num_processes=4)


def test_grpo_batch_divisible_ok():
    cfg = {
        "training": {"per_device_batch_size": 1},
        "grpo": {"num_samples_per_prompt": 4},
    }
    validate_grpo_batch_vs_generations(cfg, num_processes=4)


def test_dry_run_placeholder_refused(tmp_path):
    d = tmp_path / "model_rl"
    d.mkdir()
    (d / "DRY_RUN_PLACEHOLDER").write_text("fake\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="dry-run placeholder"):
        assert_not_dry_run_placeholder(d)


def test_vllm_adapter_refused(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="PEFT/LoRA"):
        assert_not_adapter_for_vllm(d)


def test_parse_json_score_empty_fails():
    with pytest.raises(ValueError, match="empty"):
        parse_json_score("")
    with pytest.raises(ValueError, match="Could not parse"):
        parse_json_score("not json at all")


def test_parse_json_score_missing_keys_fails():
    with pytest.raises(ValueError, match="missing required"):
        parse_json_score('{"clarity": 8, "overall": 8}')


class _TokNoThinking:
    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "ok"


def test_thinking_template_no_silent_fallback():
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        apply_chat_template_with_thinking(
            _TokNoThinking(),
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )


def test_list_benchmarks_mentions_three():
    from utils.benchmarks import BENCHMARKS, list_benchmarks

    text = list_benchmarks()
    assert "mbpp_plus" in text and "lcb_easy" in text and "mbpp_train" in text
    assert set(BENCHMARKS) == {"mbpp_train", "mbpp_plus", "lcb_easy"}
