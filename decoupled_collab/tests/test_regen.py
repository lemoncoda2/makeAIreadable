"""CPU-only tests for batched regen_collaboration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _trace(i: int) -> dict:
    return {
        "task_id": f"t{i}",
        "task_prompt": f"prompt {i}",
        "thinking": "plan",
        "code": f"def f{i}():\n    return {i}\n",
        "collaboration": f"ok {i}",
        "reward": 1.0,
    }


def test_dry_run_main_writes_pairs(tmp_path):
    import regen_collaboration as regen

    traces = tmp_path / "traces.jsonl"
    out = tmp_path / "raw.jsonl"
    with traces.open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps(_trace(i)) + "\n")

    regen.main(
        [
            "--base_model",
            "unused",
            "--traces",
            str(traces),
            "--output",
            str(out),
            "--dry_run",
        ]
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["regen_collaboration"].startswith("[regen-paraphrase]")
    assert rows[0]["rl_collaboration"] == "ok 0"
    assert rows[0]["dry_run"] is True


def test_regen_traces_batches_on_one_gpu(monkeypatch):
    import regen_collaboration as regen

    seen = {}

    class FakeGen:
        def __init__(self, model_path, **kwargs):
            seen["init"] = (model_path, kwargs.get("use_vllm"), kwargs.get("device"))

        def generate_many(self, traces, *, batch_size):
            seen["many"] = (len(traces), batch_size)
            return [f"regen-{t['task_id']}" for t in traces]

    monkeypatch.setattr(regen, "RegenGenerator", FakeGen)
    monkeypatch.setattr(regen, "_resolve_num_gpus", lambda requested: 1)
    monkeypatch.setattr(regen, "_release_generator", lambda gen: None)

    outs = regen.regen_traces(
        [_trace(0), _trace(1), _trace(2)],
        model_path="Qwen3-4B",
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=256,
        gen_batch_size=8,
        num_gpus=1,
    )
    assert outs == ["regen-t0", "regen-t1", "regen-t2"]
    assert seen["many"] == (3, 8)
    assert seen["init"][1] is False


def test_regen_traces_vllm_is_single_process_batched(monkeypatch):
    import regen_collaboration as regen

    seen = {}

    class FakeGen:
        def __init__(self, model_path, **kwargs):
            seen["vllm"] = kwargs.get("use_vllm")

        def generate_many(self, traces, *, batch_size):
            seen["n"] = len(traces)
            seen["batch"] = batch_size
            return ["x"] * len(traces)

    monkeypatch.setattr(regen, "RegenGenerator", FakeGen)
    monkeypatch.setattr(regen, "_release_generator", lambda gen: None)

    def boom(*args, **kwargs):
        raise AssertionError("vLLM path must not spawn HF shards")

    monkeypatch.setattr(regen, "_regen_shard", boom)
    outs = regen.regen_traces(
        [_trace(0), _trace(1)],
        model_path="Qwen3-4B",
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=256,
        gen_batch_size=16,
        num_gpus=4,
    )
    assert outs == ["x", "x"]
    assert seen["vllm"] is True
    assert seen["batch"] == 16


def test_reject_think_fails_fast():
    import regen_collaboration as regen

    with pytest.raises(RuntimeError, match="think"):
        regen._reject_think("hello <think>secret</think>")


def test_generate_many_hf_rejects_think_tags():
    torch = pytest.importorskip("torch")
    import regen_collaboration as regen

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, chunk, return_tensors="pt", padding=True):
            return {"input_ids": torch.ones((len(chunk), 3), dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return "<think>leaked</think> explanation"

    class Model:
        def generate(self, **kwargs):
            batch, prompt = kwargs["input_ids"].shape
            return torch.zeros((batch, prompt + 2), dtype=torch.long)

    gen = object.__new__(regen.RegenGenerator)
    gen.use_vllm = False
    gen.temperature = 0.0
    gen.max_new_tokens = 8
    gen.tokenizer = Tok()
    gen.model = Model()
    gen._device = torch.device("cpu")
    gen._render = lambda trace: "prompt"
    with pytest.raises(RuntimeError, match="think"):
        gen.generate_many([_trace(0)], batch_size=1)
