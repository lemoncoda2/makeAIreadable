"""Turn GRPO reward_audit rollouts into collect-style traces.

Used so DPO can reuse training-time 4-sample rollouts instead of a second
live collect pass. Default keep rule: reward > 0 and late training steps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from utils.code_executor import separate_output


AUDIT_GLOB = "reward_rank*.jsonl"


def find_reward_audit_dir(model_path: str | Path) -> Optional[Path]:
    """Locate ``reward_audit`` next to an RL adapter (or its merged sibling)."""
    root = Path(model_path)
    candidates = [
        root / "reward_audit",
        root.parent / "model_rl" / "reward_audit",
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.glob(AUDIT_GLOB)):
            return cand
    return None


def load_audit_rows(audit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob(AUDIT_GLOB)):
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {path}:{line_no}: {exc}"
                    ) from exc
                row["_audit_file"] = path.name
                rows.append(row)
    return rows


def load_task_prompts(tasks_path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with open(tasks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_id = item.get("task_id")
            if task_id is None:
                continue
            prompts[str(task_id)] = item.get("prompt") or item.get("task_prompt") or ""
    return prompts


def _code_key(code: str) -> str:
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()


def harvest_rollouts(
    audit_dir: Path,
    tasks_path: Path,
    *,
    min_reward: float = 0.0,
    later_frac: float = 1.0 / 3.0,
    min_later_traces: int = 32,
    max_traces: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build collect-format traces from reward_audit jsonl.

    Keep rollouts with ``reward > min_reward``. Prefer the last ``later_frac``
    of ``call`` indices; if that set is thinner than ``min_later_traces`` after
    split/dedup, fall back to every positive-reward rollout.
    """
    if not 0.0 < later_frac <= 1.0:
        raise ValueError(f"later_frac must be in (0, 1], got {later_frac}")

    raw = load_audit_rows(audit_dir)
    prompts = load_task_prompts(tasks_path)
    stats: dict[str, Any] = {
        "audit_rows": len(raw),
        "audit_dir": str(audit_dir),
        "min_reward": min_reward,
        "later_frac": later_frac,
        "used_later_window": False,
        "fallback_all_positive": False,
    }
    if not raw:
        raise RuntimeError(f"No reward_audit rows under {audit_dir}")

    positive = []
    for row in raw:
        try:
            reward = float(row.get("reward") or 0.0)
        except (TypeError, ValueError):
            continue
        if reward > min_reward:
            positive.append(row)
    stats["positive_rows"] = len(positive)
    if not positive:
        raise RuntimeError(
            f"No rollouts with reward > {min_reward} in {audit_dir} "
            f"({len(raw)} audit rows)."
        )

    max_call = max(int(r.get("call") or 0) for r in raw)
    later_cut = int(max_call * (1.0 - later_frac))
    stats["max_call"] = max_call
    stats["later_cut"] = later_cut

    def to_traces(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[tuple[str, str], dict[str, Any]] = {}
        skipped_no_prompt = 0
        skipped_split = 0
        for row in rows:
            task_id = row.get("task_id")
            if task_id is None:
                skipped_no_prompt += 1
                continue
            task_id = str(task_id)
            prompt = prompts.get(task_id, "")
            if not prompt:
                skipped_no_prompt += 1
                continue
            completion = row.get("completion") or ""
            separated = separate_output(completion)
            code = (separated.get("code") or "").strip()
            collab = (separated.get("collaboration") or "").strip()
            thinking = (separated.get("thinking") or "").strip()
            if not code or not collab:
                skipped_split += 1
                continue
            key = (task_id, _code_key(code))
            call = int(row.get("call") or 0)
            prev = best.get(key)
            if prev is not None and int(prev.get("audit_call") or 0) > call:
                continue
            best[key] = {
                "task_id": task_id,
                "task_prompt": prompt,
                "full_output": completion,
                "thinking": thinking,
                "code": code,
                "collaboration": collab,
                "reward": float(row.get("reward") or 0.0),
                "work_trace": thinking + "\n[CODE]\n" + code,
                "source": "grpo_reward_audit",
                "audit_call": call,
                "audit_file": row.get("_audit_file"),
            }
        stats["skipped_no_prompt"] = skipped_no_prompt
        stats["skipped_split"] = skipped_split
        traces = sorted(
            best.values(),
            key=lambda t: (-int(t["audit_call"]), t["task_id"]),
        )
        return traces

    later_rows = [
        r for r in positive if int(r.get("call") or 0) >= later_cut
    ]
    later_traces = to_traces(later_rows)
    if len(later_traces) >= min_later_traces:
        traces = later_traces
        stats["used_later_window"] = True
    else:
        traces = to_traces(positive)
        stats["fallback_all_positive"] = True
        stats["later_traces_before_fallback"] = len(later_traces)

    if max_traces is not None:
        traces = traces[: max_traces]
    stats["kept"] = len(traces)
    stats["unique_tasks"] = len({t["task_id"] for t in traces})
    return traces, stats


def write_traces(traces: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for row in traces:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
