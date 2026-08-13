#!/usr/bin/env python3
"""Merge a PEFT/LoRA adapter into the base model for vLLM or HF full-model serving.

vLLM cannot load adapter_config.json directories. Use this before --use_vllm true.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.failfast import assert_not_dry_run_placeholder  # noqa: E402
from utils.model_utils import (  # noqa: E402
    find_base_model_path,
    load_causal_lm,
    load_tokenizer,
    maybe_merge_or_load_peft,
    resolve_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    p.add_argument("--base_model", default="./models/Qwen3-4B")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--output", required=True, help="Output directory for merged model")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    os_chdir = ROOT
    import os

    os.chdir(os_chdir)

    base = resolve_path(args.base_model, ROOT)
    adapter = resolve_path(args.adapter, ROOT)
    out = resolve_path(args.output, ROOT)

    assert_not_dry_run_placeholder(adapter, what="adapter")
    if not (adapter / "adapter_config.json").exists():
        print(f"[error] Not a PEFT adapter (missing adapter_config.json): {adapter}")
        return 1

    # Merge runs on CPU; CUDA not required (useful for offline prep / low VRAM hosts).
    base_resolved = find_base_model_path(adapter, default_base=base, root=ROOT)
    print(f"[info] base={base_resolved} adapter={adapter} → {out} (device_map=cpu)")

    tok = load_tokenizer(base_resolved)
    model = load_causal_lm(base_resolved, torch_dtype=args.dtype, device_map="cpu")
    model = maybe_merge_or_load_peft(
        model, adapter, is_trainable=False, merge_and_unload=True
    )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"[info] Merged model saved to {out}")
    print("Use this directory with collect_traces/regen --use_vllm true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
