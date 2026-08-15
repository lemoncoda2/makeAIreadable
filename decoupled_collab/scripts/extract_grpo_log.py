#!/usr/bin/env python3
"""Extract GRPO trainer metrics + audit summary into durable logs.

Does not touch the training process. Safe to re-run; files are overwritten
(snapshot) or appended (jsonl events with de-dup by step).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/makeAIreadable-20260814/workspace/decoupled_collab")
LOG_DIR = ROOT / "logs" / "cycle_0"
GRPO_LOG = LOG_DIR / "grpo.log"
AUDIT_DIR = ROOT / "checkpoints" / "cycle_0" / "model_rl" / "reward_audit"
STATE = ROOT / "pipeline_state.json"
OUT_JSONL = LOG_DIR / "grpo_metrics.jsonl"
OUT_MD = LOG_DIR / "GRPO_TRAIN_LOG.md"
OUT_SNAP = LOG_DIR / "grpo_snapshot.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_progress(text: str) -> list[dict]:
    #  3%|▎         | 10/300 [1:01:08<29:17:09, 363.55s/it]
    pat = re.compile(
        r"(?P<step>\d+)/300\s+\[(?P<elapsed>[^<]+)<(?P<eta>[^,]+),\s+(?P<sec>[0-9.]+)s/it\]"
    )
    rows = []
    for m in pat.finditer(text):
        rows.append(
            {
                "step": int(m.group("step")),
                "elapsed": m.group("elapsed").strip(),
                "eta": m.group("eta").strip(),
                "sec_per_it": float(m.group("sec")),
            }
        )
    # keep last occurrence per step
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[k] for k in sorted(by_step)]


def parse_metrics(text: str) -> list[dict]:
    # Trainer dicts dumped after logging_steps
    rows = []
    for m in re.finditer(r"\{[^{}]*'loss'[^{}]*\}", text):
        raw = m.group(0)
        try:
            obj = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "epoch" in obj:
            rows.append(obj)
    return rows


def load_existing_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def audit_summary() -> dict:
    rows = []
    if not AUDIT_DIR.is_dir():
        return {"exists": False, "rows": 0}
    for p in sorted(AUDIT_DIR.glob("reward_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {"exists": True, "rows": 0}
    rewards = [float(r.get("reward") or 0.0) for r in rows]
    pos = sum(1 for x in rewards if x > 0)
    hist = Counter(round(x, 2) for x in rewards)
    return {
        "exists": True,
        "rows": len(rows),
        "reward_gt0": pos,
        "reward_gt0_frac": round(pos / len(rows), 4),
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "has_think_end": sum(1 for r in rows if r.get("has_think_end")),
        "has_python_fence": sum(1 for r in rows if r.get("has_python_fence")),
        "task_ids": len({r.get("task_id") for r in rows}),
        "call_min": min(int(r.get("call") or 0) for r in rows),
        "call_max": max(int(r.get("call") or 0) for r in rows),
        "reward_hist": hist.most_common(8),
    }


def adapter_info() -> dict:
    rl = ROOT / "checkpoints" / "cycle_0" / "model_rl"
    adapters = list(rl.glob("adapter_model*"))
    ckpts = sorted(p.name for p in rl.glob("checkpoint-*") if p.is_dir())
    return {
        "adapter_config": (rl / "adapter_config.json").exists(),
        "adapter_weights": [p.name for p in adapters],
        "checkpoints": ckpts,
        "placeholder": (rl / "DRY_RUN_PLACEHOLDER").exists(),
    }


def write_md(snap: dict) -> None:
    prog = snap.get("progress") or []
    last = prog[-1] if prog else {}
    metrics = snap.get("metrics") or []
    last_m = metrics[-1] if metrics else {}
    audit = snap.get("audit") or {}
    ad = snap.get("adapter") or {}
    lines = [
        "# Cycle-0 GRPO training log",
        "",
        f"Updated (UTC): {snap['utc']}",
        "Source: `logs/cycle_0/grpo.log` + `checkpoints/cycle_0/model_rl/reward_audit/`",
        "Training process was not modified. This file is a snapshot of observed results.",
        "",
        "## Status",
        "",
        f"- pipeline: `{snap.get('pipeline')}`",
        f"- last step: **{last.get('step', '?')}/300**",
        f"- sec/it: {last.get('sec_per_it', '?')}",
        f"- elapsed: {last.get('elapsed', '?')}",
        f"- ETA in bar: {last.get('eta', '?')}",
        f"- adapter_config: {ad.get('adapter_config')}",
        f"- adapter weights: {ad.get('adapter_weights')}",
        f"- trainer checkpoints: {ad.get('checkpoints')}",
        f"- DRY_RUN_PLACEHOLDER: {ad.get('placeholder')}",
        "",
        "## Latest Trainer metrics",
        "",
    ]
    if last_m:
        for k in (
            "loss",
            "grad_norm",
            "learning_rate",
            "rewards/code_execution_reward",
            "reward",
            "reward_std",
            "completion_length",
            "kl",
            "epoch",
        ):
            if k in last_m:
                lines.append(f"- `{k}`: {last_m[k]}")
    else:
        lines.append("- (no logged Trainer dict yet; first dump is log_every_n_steps=10)")
    lines += [
        "",
        "## Reward audit (all ranks, cumulative)",
        "",
        f"- rows: {audit.get('rows')}",
        f"- reward>0: {audit.get('reward_gt0')} ({audit.get('reward_gt0_frac')})",
        f"- mean reward: {audit.get('reward_mean')}",
        f"- unique task_ids: {audit.get('task_ids')}",
        f"- audit call range: {audit.get('call_min')} .. {audit.get('call_max')}",
        f"- has_think_end: {audit.get('has_think_end')}",
        f"- has_python_fence: {audit.get('has_python_fence')}",
        f"- reward_hist: {audit.get('reward_hist')}",
        "",
        "## Step timeline (from tqdm)",
        "",
        "| step | elapsed | eta | sec/it |",
        "|-----:|---------|-----|-------:|",
    ]
    for r in prog:
        lines.append(
            f"| {r['step']} | {r['elapsed']} | {r['eta']} | {r['sec_per_it']} |"
        )
    lines += [
        "",
        "## Trainer metric history",
        "",
    ]
    if metrics:
        keys = [
            "epoch",
            "reward",
            "reward_std",
            "completion_length",
            "kl",
            "grad_norm",
            "learning_rate",
            "loss",
        ]
        header = "| " + " | ".join(keys) + " |"
        sep = "|" + "|".join(["---"] * len(keys)) + "|"
        lines += [header, sep]
        for m in metrics:
            lines.append("| " + " | ".join(str(m.get(k, "")) for k in keys) + " |")
    else:
        lines.append("(empty)")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text = GRPO_LOG.read_text(encoding="utf-8", errors="replace") if GRPO_LOG.exists() else ""
    progress = parse_progress(text)
    metrics = parse_metrics(text)
    pipeline = {}
    if STATE.exists():
        pipeline = json.loads(STATE.read_text(encoding="utf-8"))
    snap = {
        "utc": utc_now(),
        "pipeline": pipeline,
        "progress": progress,
        "metrics": metrics,
        "audit": audit_summary(),
        "adapter": adapter_info(),
        "grpo_log_bytes": GRPO_LOG.stat().st_size if GRPO_LOG.exists() else 0,
    }
    OUT_SNAP.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    existing = load_existing_jsonl(OUT_JSONL)
    seen_steps = {e.get("step") for e in existing if "step" in e}
    seen_epochs = {(round(float(e.get("epoch", -1)), 4), e.get("reward")) for e in existing if "reward" in e and "step" not in e}
    new_events = []
    for p in progress:
        if p["step"] not in seen_steps:
            ev = {"kind": "progress", "utc": snap["utc"], **p}
            new_events.append(ev)
            seen_steps.add(p["step"])
    for m in metrics:
        key = (round(float(m.get("epoch", -1)), 4), m.get("reward"))
        if key not in seen_epochs:
            ev = {"kind": "metrics", "utc": snap["utc"], **m}
            new_events.append(ev)
            seen_epochs.add(key)
    if new_events:
        with OUT_JSONL.open("a", encoding="utf-8") as fh:
            for ev in new_events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    write_md(snap)
    last = progress[-1]["step"] if progress else None
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_SNAP}")
    print(f"appended {len(new_events)} events -> {OUT_JSONL}")
    print(f"last_step={last} audit_rows={snap['audit'].get('rows')}")


if __name__ == "__main__":
    main()
