#!/usr/bin/env python3
"""Prepare REAL benchmark/training data for decoupled_collab (GOAL Step 0.4).

Datasets produced
-----------------
1. ``data/mbpp_train.jsonl`` — MBPP **full** (HF) for GRPO *training prompts*
2. ``data/mbpp_plus_test.jsonl`` — **MBPP+ via EvalPlus** (primary eval)  ← not HF sanitized
3. ``data/lcb_easy.jsonl`` — LiveCodeBench **easy** (secondary eval)

Real runs refuse missing/empty/synthetic files (see ``utils.benchmarks``).
Use ``--download`` on a machine with network access.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.benchmarks import BENCHMARKS, list_benchmarks  # noqa: E402


def extract_function_name(code: str) -> Optional[str]:
    """Extract the first top-level function name via AST, with regex fallback."""
    code = (code or "").strip()
    if not code:
        return None
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        pass
    match = re.search(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", code, re.MULTILINE)
    return match.group(1) if match else None


def _iter_split_items(ds: Any, preferred_splits: Iterable[str]) -> list[Any]:
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        for split in preferred_splits:
            if split in keys:
                return list(ds[split])
        if keys:
            return list(ds[keys[0]])
        return []
    return list(ds)


def _load_mbpp(path: Path, download: bool, config_name: str) -> Any:
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
            print(f"[warn] Could not save_to_disk ({e}); continuing in-memory")
        return ds

    raise FileNotFoundError(
        f"MBPP data not found at {path}. Pass --download or place save_to_disk data there."
    )


def _normalize_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "keys"):
        return {k: item[k] for k in item.keys()}
    return dict(item)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_assertions(assertion_block: str) -> list[str]:
    lines = []
    for line in (assertion_block or "").splitlines():
        s = line.strip()
        if s.startswith("assert "):
            lines.append(s)
    return lines


def _plus_inputs_to_asserts(
    entry_point: str,
    canonical_solution: str,
    plus_input: list,
    *,
    max_cases: int = 40,
) -> list[str]:
    """Materialize EvalPlus plus_input into assert statements via canonical solution."""
    if not plus_input:
        return []
    ns: dict[str, Any] = {}
    try:
        exec(canonical_solution, ns, ns)  # noqa: S102 — trusted EvalPlus ground truth
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to exec canonical_solution for {entry_point}: {e}"
        ) from e
    if entry_point not in ns or not callable(ns[entry_point]):
        raise RuntimeError(
            f"canonical_solution did not define callable {entry_point!r}"
        )
    fn = ns[entry_point]
    out: list[str] = []
    for args in plus_input[:max_cases]:
        if not isinstance(args, (list, tuple)):
            args = [args]
        try:
            expected = fn(*args)
        except Exception:
            # Skip inputs that even the oracle rejects (contracts etc.)
            continue
        arg_list = ", ".join(repr(a) for a in args)
        out.append(f"assert {entry_point}({arg_list}) == {repr(expected)}")
    return out


# ---------------------------------------------------------------------------
# 1) MBPP full → training
# ---------------------------------------------------------------------------


def prepare_grpo_tasks(
    raw_mbpp_full: Path,
    output: Path,
    *,
    download: bool = False,
) -> list[dict[str, Any]]:
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
        tasks.append(
            {
                "task_id": f"mbpp_{task_id}",
                "prompt": prompt,
                "test_cases": list(test_cases),
                "code_solution": item.get("code") or item.get("code_solution") or "",
                "source": "mbpp_full",
                "benchmark": "mbpp_train",
                "synthetic": False,
            }
        )

    min_rows = BENCHMARKS["mbpp_train"]["min_rows"]
    if download and len(tasks) < min_rows:
        raise RuntimeError(
            f"MBPP full prepared only {len(tasks)} tasks; expected >={min_rows}. "
            "Download may have failed."
        )

    _write_jsonl(output, tasks)
    print(f"Prepared {len(tasks)} GRPO training tasks (source=mbpp_full) → {output}")
    return tasks


# ---------------------------------------------------------------------------
# 2) MBPP+ via EvalPlus → primary eval
# ---------------------------------------------------------------------------


def prepare_mbpp_plus_evalplus(
    output: Path,
    *,
    max_plus_cases: int = 40,
) -> list[dict[str, Any]]:
    """
    Download/load **real MBPP+** through EvalPlus (not HF sanitized MBPP).

    Requires: ``pip install evalplus`` and network on first call.
    """
    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as e:
        raise ImportError(
            "evalplus is required for real MBPP+. Install: pip install 'evalplus>=0.2.0'"
        ) from e

    print("[info] Loading MBPP+ via evalplus.data.get_mbpp_plus() ...")
    problems = get_mbpp_plus()
    if not problems:
        raise RuntimeError("evalplus.get_mbpp_plus() returned empty")

    tasks: list[dict[str, Any]] = []
    for task_id, problem in problems.items():
        entry = problem.get("entry_point") or ""
        base_asserts = _split_assertions(problem.get("assertion") or "")
        plus_asserts = _plus_inputs_to_asserts(
            entry,
            problem.get("canonical_solution") or "",
            list(problem.get("plus_input") or []),
            max_cases=max_plus_cases,
        )
        test_cases = base_asserts + plus_asserts
        if not test_cases:
            raise RuntimeError(
                f"MBPP+ task {task_id} produced zero assert test_cases; aborting."
            )
        tasks.append(
            {
                "task_id": str(task_id),
                "prompt": problem.get("prompt") or "",
                "entry_point": entry,
                "test_cases": test_cases,
                "n_base_tests": len(base_asserts),
                "n_plus_tests": len(plus_asserts),
                "code_solution": problem.get("canonical_solution") or "",
                "source": "evalplus_mbpp_plus",
                "benchmark": "mbpp_plus",
                "synthetic": False,
            }
        )

    min_rows = BENCHMARKS["mbpp_plus"]["min_rows"]
    if len(tasks) < min_rows:
        raise RuntimeError(
            f"MBPP+ only has {len(tasks)} tasks (< {min_rows}). EvalPlus download incomplete?"
        )

    _write_jsonl(output, tasks)
    print(
        f"Prepared {len(tasks)} MBPP+ eval tasks (source=evalplus_mbpp_plus, "
        f"plus_cases≤{max_plus_cases}/task) → {output}"
    )
    return tasks


def prepare_eval_tasks(
    raw_mbpp_sanitized: Path,
    output: Path,
    *,
    download: bool = False,
) -> list[dict[str, Any]]:
    """
    Deprecated path: HF sanitized is **not** MBPP+.

    Kept only to fail loudly if someone still calls it for mbpp_plus_test.jsonl.
    """
    raise RuntimeError(
        "prepare_eval_tasks() (HF MBPP sanitized) is disabled for "
        f"{output}. MBPP+ must come from EvalPlus via prepare_mbpp_plus_evalplus(). "
        "Run: python src/prepare_data.py --download"
    )


# ---------------------------------------------------------------------------
# 3) LiveCodeBench easy → secondary eval
# ---------------------------------------------------------------------------


def prepare_lcb_easy(output: Path, *, download: bool = False) -> list[dict[str, Any]]:
    """
    Prepare LiveCodeBench-easy. Fails hard if data cannot be loaded when download=True.

    Never writes a silent empty stub that looks like success.
    """
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets package required for LiveCodeBench download") from e

    candidates = [
        ("livecodebench/code_generation_lite", None),
        ("livecodebench/code_generation", None),
    ]

    for name, subset in candidates:
        try:
            print(f"[info] Trying LiveCodeBench dataset {name!r} ...")
            kwargs = {"trust_remote_code": True}
            if subset:
                ds = load_dataset(name, subset, **kwargs)
            else:
                ds = load_dataset(name, **kwargs)
            rows = _iter_split_items(ds, preferred_splits=("test", "train", "validation"))
            for i, item in enumerate(rows):
                item = _normalize_item(item)
                difficulty = str(item.get("difficulty", item.get("level", ""))).lower()
                if difficulty and difficulty not in ("easy", "e"):
                    continue
                prompt = (
                    item.get("question_content")
                    or item.get("prompt")
                    or item.get("problem")
                    or item.get("question")
                    or ""
                )
                raw_tests = (
                    item.get("public_test_cases")
                    or item.get("test_cases")
                    or item.get("tests")
                    or []
                )
                if isinstance(raw_tests, str):
                    try:
                        raw_tests = json.loads(raw_tests)
                    except json.JSONDecodeError:
                        raw_tests = [raw_tests]
                # Prefer assert-style strings; JSON dumps are NOT executable by
                # code_executor (evaluate refuses LCB if coverage is too low).
                test_cases: list[str] = []
                for tc in raw_tests:
                    if isinstance(tc, str):
                        s = tc.strip()
                        if s.startswith("assert") or s.startswith("assert("):
                            test_cases.append(s)
                        else:
                            # Keep raw string only if it looks like an assert later;
                            # skip opaque payloads that would fake a prepared dataset.
                            try:
                                parsed = json.loads(s)
                            except json.JSONDecodeError:
                                test_cases.append(s)
                                continue
                            if isinstance(parsed, dict) and (
                                "assert" in parsed or "code" in parsed
                            ):
                                test_cases.append(s)
                            # else: drop stdin JSON blobs (not runnable here)
                    elif isinstance(tc, dict):
                        # Functional: {"input": ..., "output": ...} needs a harness —
                        # do not store json.dumps as a fake "test".
                        continue
                    else:
                        continue
                if not prompt or not test_cases:
                    continue
                tasks.append(
                    {
                        "task_id": f"lcb_{item.get('question_id', item.get('task_id', i))}",
                        "prompt": prompt,
                        "test_cases": test_cases,
                        "difficulty": "easy",
                        "source": "livecodebench_easy",
                        "benchmark": "lcb_easy",
                        "dataset": name,
                        "synthetic": False,
                    }
                )
            if tasks:
                break
            errors.append(f"{name}: loaded but 0 easy rows")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            continue

    min_rows = BENCHMARKS["lcb_easy"]["min_rows"]
    if len(tasks) < min_rows:
        msg = (
            "Failed to prepare real LiveCodeBench-easy data.\n"
            f"  got {len(tasks)} tasks (min_rows={min_rows})\n"
            f"  tried: {errors}\n"
            "Manual options:\n"
            "  1) Clone https://github.com/LiveCodeBench/LiveCodeBench and convert easy split\n"
            "  2) Write data/lcb_easy.jsonl with fields: "
            "task_id, prompt, test_cases, source='livecodebench_easy', synthetic=false\n"
            "Refusing to write an empty stub that would fake a successful prepare."
        )
        if download:
            raise RuntimeError(msg)
        print("[error]", msg)
        raise RuntimeError(msg)

    _write_jsonl(output, tasks)
    print(
        f"Prepared {len(tasks)} LiveCodeBench-easy tasks "
        f"(source=livecodebench_easy) → {output}"
    )
    return tasks


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare REAL MBPP-full / MBPP+ / LCB-easy datasets",
        epilog=list_benchmarks(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--download", action="store_true", help="Fetch datasets from the network")
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print required benchmarks and exit",
    )
    parser.add_argument("--raw-mbpp-full", type=Path, default=ROOT / "data" / "raw" / "mbpp_full")
    parser.add_argument(
        "--train-output", type=Path, default=ROOT / "data" / "mbpp_train.jsonl"
    )
    parser.add_argument(
        "--eval-output", type=Path, default=ROOT / "data" / "mbpp_plus_test.jsonl"
    )
    parser.add_argument("--lcb-output", type=Path, default=ROOT / "data" / "lcb_easy.jsonl")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-mbpp-plus", action="store_true")
    parser.add_argument("--skip-lcb", action="store_true")
    parser.add_argument(
        "--require-lcb",
        action="store_true",
        default=True,
        help="Fail if LCB-easy cannot be prepared (default: true)",
    )
    parser.add_argument(
        "--allow-missing-lcb",
        action="store_true",
        help="Do not fail the whole script if LCB prepare fails (not recommended)",
    )
    parser.add_argument("--max-plus-cases", type=int, default=40)
    args = parser.parse_args(argv)

    if args.list_benchmarks:
        print(list_benchmarks())
        return

    if not args.download and not args.skip_mbpp_plus:
        # Still allow offline re-prep from cached evalplus files once downloaded.
        print(
            "[info] Tip: first-time setup needs --download "
            "(EvalPlus MBPP+ release + HF MBPP full + LCB)."
        )

    if not args.skip_train:
        if not args.download and not args.raw_mbpp_full.exists():
            raise SystemExit(
                "[error] MBPP full not on disk. Re-run with --download.\n"
                + list_benchmarks()
            )
        prepare_grpo_tasks(args.raw_mbpp_full, args.train_output, download=args.download)

    if not args.skip_mbpp_plus:
        prepare_mbpp_plus_evalplus(args.eval_output, max_plus_cases=args.max_plus_cases)

    if not args.skip_lcb:
        try:
            prepare_lcb_easy(args.lcb_output, download=args.download or True)
        except Exception as e:  # noqa: BLE001
            if args.allow_missing_lcb:
                print(f"[warn] LCB prepare failed (--allow-missing-lcb): {e}")
            else:
                raise SystemExit(f"[error] {e}\n\n{list_benchmarks()}") from e

    print("✓ Real benchmark/training data preparation finished")
    print(list_benchmarks())


if __name__ == "__main__":
    main()
