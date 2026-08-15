#!/usr/bin/env python3
"""Write cycle-0 TRAIN_REPORT.md from the real phase4 eval JSON."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/makeAIreadable-20260814/workspace/decoupled_collab")


def _load(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _yn(ok) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "n/a"


def _metric(block: dict | None, *keys):
    if not isinstance(block, dict):
        return None
    for key in keys:
        if key in block and block[key] is not None:
            return block[key]
    return None


def _fmt(val, nd=4):
    if val is None:
        return "n/a"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.{nd}f}"
    return str(val)


def build_report(root: Path) -> str:
    eval_path = root / "results" / "cycle_0_eval.json"
    if not eval_path.exists():
        eval_path = root / "results" / "cycle_0_full_eval.json"
    payload = _load(eval_path) or {}
    state = _load(root / "pipeline_state.json") or {}
    models = payload.get("models") or {}
    hyp = payload.get("hypothesis_results") or {}
    base = models.get("base") or {}
    rl = models.get("rl") or {}
    final = models.get("final") or {}

    h1 = hyp.get("H1_rl_improves_coding") or {}
    h2 = hyp.get("H2_rl_hurts_readability") or {}
    h3 = hyp.get("H3_dpo_recovers_readability") or {}
    h4 = hyp.get("H4_dpo_preserves_coding") or {}

    kept_read = h3.get("verified")
    kept_code = h4.get("verified")
    if kept_read is True and kept_code is True:
        verdict = (
            "DPO kept readability (H3) and coding (H4) relative to the GOAL thresholds."
        )
    elif kept_read is True and kept_code is False:
        verdict = "DPO kept readability (H3) but did not keep coding within H4's 2% gap."
    elif kept_read is False and kept_code is True:
        verdict = "DPO kept coding (H4) but did not recover readability to Base (H3)."
    elif kept_read is False and kept_code is False:
        verdict = "DPO did not keep readability (H3) or coding (H4) on this cycle."
    else:
        verdict = "H3/H4 incomplete — eval JSON missing final metrics."

    harvest = ""
    collect_log = root / "logs" / "cycle_0" / "collect.log"
    if collect_log.exists():
        for line in collect_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[harvest]" in line or "Wrote " in line:
                harvest += line.strip() + "\n"

    filter_tail = ""
    filter_log = root / "logs" / "cycle_0" / "filter.log"
    if filter_log.exists():
        lines = filter_log.read_text(encoding="utf-8", errors="replace").splitlines()
        filter_tail = "\n".join(lines[-12:])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lcb_base = _metric(base, "lcb_easy_pass1", "lcb_pass1")
    lcb_rl = _metric(rl, "lcb_easy_pass1", "lcb_pass1")
    lcb_final = _metric(final, "lcb_easy_pass1", "lcb_pass1")

    return f"""# Cycle-0 GRPO + DPO report

Generated: {now}
Source eval: `{eval_path}`
Pipeline state: cycle={state.get("current_cycle")} phase={state.get("current_phase")} status={state.get("status")}

## Verdict

{verdict}

- H3 readability kept (Final ≥ Base): **{_yn(kept_read)}**
- H4 coding kept (|Final − RL| ≤ 2%): **{_yn(kept_code)}**

## Measured scores

| model | MBPP+ pass@1 | readability overall | LCB-easy pass@1 |
|-------|--------------|---------------------|-----------------|
| Base  | {_fmt(base.get("mbpp_plus_pass1"))} | {_fmt(base.get("readability_overall"), 3)} | {_fmt(lcb_base)} |
| RL (GRPO) | {_fmt(rl.get("mbpp_plus_pass1"))} | {_fmt(rl.get("readability_overall"), 3)} | {_fmt(lcb_rl)} |
| Final (GRPO+DPO) | {_fmt(final.get("mbpp_plus_pass1"))} | {_fmt(final.get("readability_overall"), 3)} | {_fmt(lcb_final)} |

## H1–H4 (GOAL Success Criteria)

| id | claim | verified | delta | notes |
|----|-------|----------|-------|-------|
| H1 | RL MBPP+ ≥ Base + 5% | {_yn(h1.get("verified"))} | {h1.get("delta")} | base={h1.get("base")} rl={h1.get("rl")} |
| H2 | RL readability < Base | {_yn(h2.get("verified"))} | {h2.get("delta")} | base={h2.get("base")} rl={h2.get("rl")} |
| H3 | Final readability ≥ Base | {_yn(h3.get("verified"))} | {h3.get("delta")} | final={h3.get("final")} |
| H4 | Final MBPP+ within 2% of RL | {_yn(h4.get("verified"))} | {h4.get("delta")} | final={h4.get("final")} |

## LCB-easy vs MBPP+

- Base LCB={_fmt(lcb_base)} / MBPP+={_fmt(base.get("mbpp_plus_pass1"))}
- RL   LCB={_fmt(lcb_rl)} / MBPP+={_fmt(rl.get("mbpp_plus_pass1"))}
- Final LCB={_fmt(lcb_final)} / MBPP+={_fmt(final.get("mbpp_plus_pass1"))}

If LCB-easy and MBPP+ move in the same direction, contamination on the train split is less likely to be the whole story. Opposite moves need a closer look at the per-task dumps.

## Pair construction (approved rules, not GOAL 3.2 API gate)

Harvest log excerpts:

```
{harvest.strip() or "(collect.log missing or no [harvest] line)"}
```

Filter tail:

```
{filter_tail.strip() or "(filter.log missing)"}
```

DeepSeek is eval-only. Pairs: reward>0 GRPO rollouts (prefer later third) + Base rewrite as chosen + RL collab as rejected.

## Log pointers

- `logs/pipeline_master.log`
- `logs/cycle_0/grpo.log`
- `logs/cycle_0/phase1_eval.log`
- `logs/cycle_0/collect.log`
- `logs/cycle_0/regen.log`
- `logs/cycle_0/filter.log`
- `logs/cycle_0/dpo.log`
- `logs/cycle_0/eval.log`
- `pipeline_state.json`
- `checkpoints/cycle_0/model_rl/` (GRPO adapter)
- `checkpoints/cycle_0/model_rl_dpo/` (DPO adapter)
- `results/cycle_0_eval.json`
"""


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    eval_path = root / "results" / "cycle_0_eval.json"
    alt = root / "results" / "cycle_0_full_eval.json"
    if not eval_path.exists() and not alt.exists():
        print(f"[error] no cycle_0 eval JSON under {root / 'results'}")
        return 2
    text = build_report(root)
    out_dir = root / "logs" / "cycle_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "TRAIN_REPORT.md"
    out.write_text(text, encoding="utf-8")
    also = root / "results" / "TRAIN_REPORT.md"
    also.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {also}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
