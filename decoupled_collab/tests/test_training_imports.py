"""Training-stack import regression tests (no model load or GPU work)."""

from __future__ import annotations

import pytest


def test_trl_grpo_trainer_imports():
    """Catch broken optional-integration imports before a distributed launch."""
    pytest.importorskip("trl")
    from trl import GRPOConfig, GRPOTrainer

    assert GRPOConfig is not None
    assert GRPOTrainer is not None


def test_default_requirements_omit_unused_native_accelerators():
    from pathlib import Path

    requirements = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text(encoding="utf-8")
    active = {
        line.strip().split("==", 1)[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "deepspeed" not in active
    assert "bitsandbytes" not in active
