"""Canonical benchmark registry + fail-fast gates for real data.

Benchmarks used by this project (GOAL):

1. **MBPP full** (`mbpp_train`) — GRPO *training prompts* only (not the success-metric eval set).
2. **MBPP+ / EvalPlus** (`mbpp_plus`) — primary evaluation (augmented tests). Required for Phase 1/4.
3. **LiveCodeBench-easy** (`lcb_easy`) — secondary contamination-free evaluation. Required for full eval.

Synthetic/example fixtures are for `--dry_run` / smoke tests only and must carry
``"synthetic": true`` (or live under ``*.example.jsonl``). Real runs refuse them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.failfast import ConfigError

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "mbpp_train": {
        "role": "train",
        "name": "MBPP full (Google)",
        "default_path": "data/mbpp_train.jsonl",
        "source_required": "mbpp_full",
        "how": "python src/prepare_data.py --download  # datasets: google-research-datasets/mbpp full",
        "min_rows": 100,
        "description": "Training prompts + base asserts for GRPO (not MBPP+).",
    },
    "mbpp_plus": {
        "role": "eval_primary",
        "name": "MBPP+ (EvalPlus)",
        "default_path": "data/mbpp_plus_test.jsonl",
        "source_required": "evalplus_mbpp_plus",
        "how": "python src/prepare_data.py --download  # uses evalplus.data.get_mbpp_plus()",
        "min_rows": 300,
        "description": "Primary success metric (~378 tasks, base+plus tests).",
    },
    "lcb_easy": {
        "role": "eval_secondary",
        "name": "LiveCodeBench-easy",
        "default_path": "data/lcb_easy.jsonl",
        "source_required": "livecodebench_easy",
        "how": (
            "python src/prepare_data.py --download  "
            "# stores harness=lcb + lcb_tests (stdin/call); eval uses utils/lcb_executor.py"
        ),
        "min_rows": 20,
        "description": (
            "Contamination-free secondary eval (easy split). "
            "Requires structured lcb_tests, not MBPP-style asserts."
        ),
    },
}


def list_benchmarks() -> str:
    lines = ["Benchmarks required by GOAL / this repo:", ""]
    for key, meta in BENCHMARKS.items():
        lines.append(f"- [{meta['role']}] {key}: {meta['name']}")
        lines.append(f"    path: {meta['default_path']}")
        lines.append(f"    {meta['description']}")
        lines.append(f"    prepare: {meta['how']}")
        lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path, *, max_rows: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def inspect_dataset(path: Path, *, sample: int = 5) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    if path.stat().st_size == 0:
        return {"exists": True, "empty": True, "path": str(path), "n": 0}
    rows = _read_jsonl(path)
    sources = {r.get("source") for r in rows if r.get("source")}
    synthetic = sum(1 for r in rows if r.get("synthetic") is True)
    return {
        "exists": True,
        "empty": False,
        "path": str(path),
        "n": len(rows),
        "sources": sorted(s for s in sources if s),
        "synthetic_rows": synthetic,
        "sample_task_ids": [r.get("task_id") for r in rows[:sample]],
    }


def require_real_benchmark(
    key: str,
    path: Path,
    *,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Fail-fast unless ``path`` looks like a real prepared benchmark dataset.

    Real datasets must:
    - exist and be non-empty
    - meet min_rows
    - declare ``source`` matching the registry (on every row, or at least the first)
    - not be marked synthetic (unless allow_synthetic)
    """
    if key not in BENCHMARKS:
        raise ConfigError(f"Unknown benchmark key {key!r}. Known: {sorted(BENCHMARKS)}")

    meta = BENCHMARKS[key]
    info = inspect_dataset(path)
    if not info.get("exists"):
        raise ConfigError(
            f"Missing real benchmark data for {key} ({meta['name']}):\n"
            f"  expected file: {path}\n"
            f"  prepare with: {meta['how']}\n"
            f"Refusing to start with missing data."
        )
    if info.get("empty") or info.get("n", 0) == 0:
        raise ConfigError(
            f"Benchmark file is empty for {key} ({meta['name']}): {path}\n"
            f"This usually means prepare_data wrote a stub. Run:\n  {meta['how']}\n"
            f"Refusing to treat an empty file as {meta['name']}."
        )

    n = int(info["n"])
    rows = _read_jsonl(path, max_rows=min(20, n))
    if any(r.get("synthetic") is True for r in rows) and not allow_synthetic:
        raise ConfigError(
            f"Benchmark {key} contains synthetic=true rows at {path}. "
            f"Smoke fixtures cannot be used for real training/eval. Prepare real data:\n"
            f"  {meta['how']}"
        )

    min_rows = int(meta["min_rows"])
    if n < min_rows and not allow_synthetic:
        raise ConfigError(
            f"Benchmark {key} has only {n} rows (< min_rows={min_rows}) at {path}. "
            f"Likely a smoke/toy fixture. Prepare the real dataset:\n  {meta['how']}"
        )

    required_source = meta["source_required"]
    sources = {r.get("source") for r in rows}
    if None in sources or "" in sources:
        raise ConfigError(
            f"Benchmark {key} rows at {path} are missing a 'source' field. "
            f"Re-run prepare_data so each row records source={required_source!r}. "
            f"Old HF-sanitized files mislabeled as mbpp_plus are no longer accepted."
        )
    if required_source not in sources and not allow_synthetic:
        raise ConfigError(
            f"Benchmark {key} at {path} has sources={sorted(s for s in sources if s)} "
            f"but requires source={required_source!r}. "
            f"You may be using the wrong dataset (e.g. HF sanitized MBPP labeled as MBPP+). "
            f"Prepare with:\n  {meta['how']}"
        )

    # Path name heuristic: refuse *.example.jsonl for real runs
    if path.name.endswith(".example.jsonl") and not allow_synthetic:
        raise ConfigError(
            f"{path} is an example fixture. Copy/prepare real data to {meta['default_path']}."
        )

    # LCB must carry structured harness cases (stdin/call), not assert-only leftovers.
    if key == "lcb_easy" and not allow_synthetic:
        from utils.lcb_executor import task_lcb_cases

        full_rows = _read_jsonl(path)
        ready = sum(1 for r in full_rows if task_lcb_cases(r) or r.get("harness") == "lcb")
        if ready < max(1, int(0.5 * len(full_rows))):
            raise ConfigError(
                f"Benchmark lcb_easy at {path} lacks structured lcb_tests "
                f"({ready}/{len(full_rows)} ready). Re-run prepare_data.py --download "
                "so rows include harness='lcb' and stdin/call cases. "
                "Assert-only MBPP-style fixtures are not valid LiveCodeBench data."
            )

    return info


def require_real_benchmarks(
    root: Path,
    keys: Iterable[str],
    *,
    path_overrides: Optional[Dict[str, Path]] = None,
    allow_synthetic: bool = False,
) -> Dict[str, Dict[str, Any]]:
    path_overrides = path_overrides or {}
    out: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for key in keys:
        meta = BENCHMARKS[key]
        path = path_overrides.get(key) or (root / meta["default_path"])
        try:
            out[key] = require_real_benchmark(
                key, Path(path), allow_synthetic=allow_synthetic
            )
        except ConfigError as e:
            errors.append(str(e))
    if errors:
        raise ConfigError(
            "Real benchmark data required before this command can run.\n\n"
            + "\n\n".join(errors)
            + "\n\n"
            + list_benchmarks()
        )
    return out
