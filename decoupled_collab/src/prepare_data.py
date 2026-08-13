#!/usr/bin/env python3
"""Prepare MBPP / LiveCodeBench data for GRPO training and evaluation.

GOAL Step 0.4 — converts raw MBPP into jsonl task formats.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def extract_function_name(code: str) -> Optional[str]:
    """Extract the first top-level function name via AST, with regex fallback."""
    code = (code or "").strip()
    if not code:
        return None
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                return node.name
    except SyntaxError:
        pass
    match = re.search(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", code, re.MULTILINE)
    return match.group(1) if match else None


def _iter_split_items(ds: Any, preferred_splits: Iterable[str]) -> list[Any]:
    """Return rows from the first available split (dict-like or DatasetDict)."""
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        for split in preferred_splits:
            if split in keys:
                return list(ds[split])
        # Fallback: first split
        if keys:
            return list(ds[keys[0]])
        return []
    # Plain list / Dataset
    return list(ds)


def _load_mbpp(path: Path, download: bool, config_name: str) -> Any:
    """Load MBPP from disk or optionally download via HuggingFace datasets."""
    if path.exists():
        try:
            from datasets import load_from_disk

            return load_from_disk(str(path))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] load_from_disk failed for {path}: {e}")

    if download:
        from datasets import load_dataset

        print(f"[info] Downloading google-research-datasets/mbpp config={config_name}")
        ds = load_dataset("google-research-datasets/mbpp", config_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ds.save_to_disk(str(path))
            print(f"[info] Saved raw dataset to {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Could not save_to_disk ({e}); continuing with in-memory data")
        return ds

    raise FileNotFoundError(
        f"MBPP data not found at {path}. "
        "Pass --download to fetch via datasets.load_dataset, "
        "or place a datasets.save_to_disk directory there."
    )


def _normalize_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "keys"):
        return {k: item[k] for k in item.keys()}
    return dict(item)


def prepare_grpo_tasks(
    raw_mbpp_full: Path,
    output: Path,
    *,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Prepare GRPO training tasks: prompt + test_cases (+ ground-truth code)."""
    ds = _load_mbpp(raw_mbpp_full, download=download, config_name="full")
    rows = _iter_split_items(ds, preferred_splits=("train", "test", "prompt", "validation"))

    tasks: list[dict[str, Any]] = []
    for item in rows:
        item = _normalize_item(item)
        task_id = item.get("task_id", len(tasks))
        prompt = item.get("text") or item.get("prompt") or item.get("description") or ""
        test_cases = item.get("test_list") or item.get("test_cases") or []
        if isinstance(test_cases, str):
            test_cases = [test_cases]
        code = item.get("code") or item.get("code_solution") or ""
        tasks.append(
            {
                "task_id": f"mbpp_{task_id}",
                "prompt": prompt,
                "test_cases": list(test_cases),
                "code_solution": code,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Prepared {len(tasks)} GRPO training tasks → {output}")
    return tasks


def prepare_eval_tasks(
    raw_mbpp_sanitized: Path,
    output: Path,
    *,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Prepare evaluation tasks from MBPP sanitized / MBPP+ test split."""
    ds = _load_mbpp(raw_mbpp_sanitized, download=download, config_name="sanitized")
    rows = _iter_split_items(ds, preferred_splits=("test", "train", "prompt", "validation"))

    tasks: list[dict[str, Any]] = []
    for item in rows:
        item = _normalize_item(item)
        task_id = item.get("task_id", len(tasks))
        prompt = item.get("text") or item.get("prompt") or item.get("description") or ""
        test_cases = item.get("test_list") or item.get("test_cases") or []
        if isinstance(test_cases, str):
            test_cases = [test_cases]
        code = item.get("code") or item.get("code_solution") or ""
        tasks.append(
            {
                "task_id": f"mbpp_{task_id}",
                "prompt": prompt,
                "test_cases": list(test_cases),
                "entry_point": extract_function_name(code),
                "code_solution": code,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Prepared {len(tasks)} evaluation tasks → {output}")
    return tasks


def prepare_lcb_easy(output: Path) -> list[dict[str, Any]]:
    """
    Stub: try to load LiveCodeBench easy split if available; otherwise write
    an empty/placeholder file with a clear warning and manual download notes.
    """
    tasks: list[dict[str, Any]] = []
    loaded = False

    # Attempt 1: livecodebench package / datasets hub
    try:
        from datasets import load_dataset

        for name in (
            "livecodebench/code_generation_lite",
            "livecodebench/code_generation",
            "livecodebench/lcb",
        ):
            try:
                ds = load_dataset(name, split="test", trust_remote_code=True)
                for i, item in enumerate(ds):
                    item = _normalize_item(item)
                    difficulty = str(item.get("difficulty", item.get("level", ""))).lower()
                    if difficulty and difficulty not in ("easy", "e"):
                        continue
                    tasks.append(
                        {
                            "task_id": f"lcb_{item.get('question_id', item.get('task_id', i))}",
                            "prompt": item.get("question_content")
                            or item.get("prompt")
                            or item.get("problem")
                            or "",
                            "test_cases": item.get("public_test_cases")
                            or item.get("test_cases")
                            or [],
                            "difficulty": "easy",
                            "source": name,
                        }
                    )
                if tasks:
                    loaded = True
                    break
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    if not loaded or not tasks:
        msg = (
            "WARNING: LiveCodeBench easy split could not be loaded automatically. "
            f"Wrote placeholder/empty file at {output}.\n"
            "Manual download:\n"
            "  1. Clone https://github.com/LiveCodeBench/LiveCodeBench\n"
            "  2. Follow their data download instructions for the easy split\n"
            "  3. Convert problems to jsonl with fields: "
            "task_id, prompt, test_cases, difficulty\n"
            "  4. Place the file at data/lcb_easy.jsonl\n"
            "Or: pip install livecodebench / use datasets hub if a public split exists."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        print(msg)
    else:
        print(f"Prepared {len(tasks)} LiveCodeBench-easy tasks → {output}")

    return tasks


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare MBPP / LCB datasets (GOAL Step 0.4)")
    parser.add_argument(
        "--raw-mbpp-full",
        type=Path,
        default=ROOT / "data" / "raw" / "mbpp_full",
        help="Path to MBPP full (save_to_disk dir or download target)",
    )
    parser.add_argument(
        "--raw-mbpp-sanitized",
        type=Path,
        default=ROOT / "data" / "raw" / "mbpp_sanitized",
        help="Path to MBPP sanitized (save_to_disk dir or download target)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "data" / "mbpp_train.jsonl",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=ROOT / "data" / "mbpp_plus_test.jsonl",
    )
    parser.add_argument(
        "--lcb-output",
        type=Path,
        default=ROOT / "data" / "lcb_easy.jsonl",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MBPP via datasets.load_dataset if missing on disk",
    )
    parser.add_argument(
        "--skip-lcb",
        action="store_true",
        help="Skip LiveCodeBench stub preparation",
    )
    args = parser.parse_args(argv)

    prepare_grpo_tasks(args.raw_mbpp_full, args.train_output, download=args.download)
    prepare_eval_tasks(args.raw_mbpp_sanitized, args.eval_output, download=args.download)
    if not args.skip_lcb:
        prepare_lcb_easy(args.lcb_output)


if __name__ == "__main__":
    main()
