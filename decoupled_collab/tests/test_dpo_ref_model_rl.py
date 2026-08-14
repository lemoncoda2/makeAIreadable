"""DPO reference = merged Model_RL (not pretrained base)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def test_default_merged_rl_dir_naming():
    from train_dpo import default_merged_rl_dir

    assert default_merged_rl_dir(Path("/ckpt/cycle_0/model_rl")) == Path(
        "/ckpt/cycle_0/model_rl_merged"
    )


def test_fresh_dpo_lora_config_is_new_adapter():
    pytest.importorskip("peft")
    from train_dpo import build_fresh_dpo_lora_config

    cfg = {"lora": {"rank": 16, "alpha": 32, "dropout": 0.1}}
    peft_cfg = build_fresh_dpo_lora_config(cfg)
    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 32


def test_ensure_merged_reuses_existing(tmp_path):
    from train_dpo import ensure_merged_model_rl

    merged = tmp_path / "model_rl_merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"x")
    out = ensure_merged_model_rl(
        base_model=tmp_path / "base",
        rl_adapter=tmp_path / "adapter",
        merged_dir=merged,
        dtype="float16",
        tokenizer_src=tmp_path / "base",
    )
    assert out == merged


def test_docstring_states_ref_is_model_rl():
    import train_dpo as td

    doc = td.__doc__ or ""
    assert "Model_RL" in doc
    assert "pretrained base" in doc or "original base" in doc
