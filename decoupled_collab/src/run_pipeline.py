#!/usr/bin/env python3
"""
Full automation pipeline — supports resume and dry-run.

Usage:
  python src/run_pipeline.py --config configs/pipeline_config.yaml [--resume]
  python src/run_pipeline.py --config configs/pipeline_config.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STATE_FILE_NAME = "pipeline_state.json"

PHASES = [
    "phase1_grpo",
    "phase1_eval",
    "phase2_collect",
    "phase3_regen",
    "phase3_filter",
    "phase3_dpo",
    "phase4_eval",
]


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against project_root (default: ROOT)
    project_root = Path(cfg.get("general", {}).get("project_root", ".")).expanduser()
    if not project_root.is_absolute():
        project_root = (ROOT / project_root).resolve()
    cfg["_project_root"] = str(project_root)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def state_path(project_root: Path, configured: Optional[str] = None) -> Path:
    return resolve_path(project_root, configured or STATE_FILE_NAME)


def load_state(
    project_root: Path,
    *,
    resume: bool,
    cycle_id: Optional[int],
    state_file: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Always attempt to load existing state when --resume is set.
    When not resuming, start fresh (but still overwrite STATE_FILE as we go).

    Fixes GOAL bugs:
    - resume always loaded state even when resume=False (both branches identical)
    - start_phase indexing crashed when current_phase missing from PHASES
    """
    path = state_file or state_path(project_root)
    if resume and path.exists():
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"[state] Resuming from {path}")
    else:
        state = {
            "status": "running",
            "current_cycle": cycle_id if cycle_id is not None else 0,
            "current_phase": PHASES[0],
            "history": [],
        }
        if not resume:
            print("[state] Starting fresh pipeline state")

    if cycle_id is not None:
        state["current_cycle"] = cycle_id
        # When jumping to a specific cycle without resume, reset phase
        if not resume:
            state["current_phase"] = PHASES[0]

    return state


def save_state(
    project_root: Path,
    state: dict[str, Any],
    *,
    state_file: Optional[Path] = None,
) -> None:
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    path = state_file or state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def next_phase_after(phase: str) -> Optional[str]:
    """Return the phase that should run after a successful `phase`, or None if cycle done."""
    try:
        idx = PHASES.index(phase)
    except ValueError:
        return PHASES[0]
    if idx + 1 < len(PHASES):
        return PHASES[idx + 1]
    return None


def advance_state_after_phase(state: dict[str, Any], phase: str, cycle: int) -> None:
    """
    After a phase succeeds, point state at the *next* work unit.

    ``current_phase`` always means "next phase to run" (not the one just finished),
    so ``--resume`` does not re-execute a completed phase.
    """
    nxt = next_phase_after(phase)
    if nxt is not None:
        state["current_phase"] = nxt
        state["current_cycle"] = cycle
    else:
        state["current_phase"] = PHASES[0]
        state["current_cycle"] = cycle + 1
        state.setdefault("history", []).append(
            {
                "cycle": cycle,
                "completed": datetime.now(timezone.utc).isoformat(),
                "results": f"./results/cycle_{cycle}_eval.json",
            }
        )


def phases_from(current_phase: Optional[str]) -> list[str]:
    """Return PHASES slice starting at current_phase; safe if phase missing."""
    if not current_phase:
        return list(PHASES)
    try:
        idx = PHASES.index(current_phase)
    except ValueError:
        print(
            f"[warn] Unknown current_phase={current_phase!r}; "
            f"restarting from {PHASES[0]}"
        )
        idx = 0
    return PHASES[idx:]


def run_cmd(cmd: str, log_file: Path, *, cwd: Path) -> None:
    print(f"[CMD] {cmd}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        f.write(cmd + "\n")
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {cmd}")


def resolve_path(project_root: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_grpo_train_file(config: dict[str, Any], project_root: Path) -> Path:
    """Resolve the GRPO task file, allowing isolated smoke configurations."""
    configured = config.get("grpo", {}).get("train_file", "./data/mbpp_train.jsonl")
    return resolve_path(project_root, configured)


def configured_root(
    config: dict[str, Any], project_root: Path, key: str, default: str
) -> Path:
    return resolve_path(project_root, config.get("general", {}).get(key, default))


def get_model_path(
    cycle: int,
    phase: str,
    config: dict[str, Any],
    *,
    start_model: Optional[str],
) -> str:
    project_root = Path(config["_project_root"])
    checkpoint_root = configured_root(
        config, project_root, "checkpoint_root", "./checkpoints"
    )
    base = config["general"]["base_model"]
    if cycle == 0 and phase == "phase1_grpo":
        return str(resolve_path(project_root, start_model or base))
    if phase == "phase1_grpo":
        if start_model and cycle == config.get("_start_cycle", 0):
            return str(resolve_path(project_root, start_model))
        return str((checkpoint_root / f"cycle_{cycle - 1}" / "model_rl_dpo").resolve())
    if phase == "phase3_dpo":
        return str((checkpoint_root / f"cycle_{cycle}" / "model_rl").resolve())
    return str(resolve_path(project_root, base))


def _py() -> str:
    return shlex.quote(sys.executable)


def _flag(dry_run: bool) -> str:
    return " --dry_run" if dry_run else ""


def run_phase(
    phase: str,
    cycle: int,
    config: dict[str, Any],
    *,
    start_model: Optional[str],
    dry_run: bool,
    resume_phase: bool = False,
) -> None:
    project_root = Path(config["_project_root"])
    checkpoint_root = configured_root(
        config, project_root, "checkpoint_root", "./checkpoints"
    )
    log_root = configured_root(config, project_root, "log_root", "./logs")
    trace_root = configured_root(
        config, project_root, "trace_root", "./data/traces"
    )
    dpo_pairs_root = configured_root(
        config, project_root, "dpo_pairs_root", "./data/dpo_pairs"
    )
    results_root = configured_root(config, project_root, "results_root", "./results")
    cycle_dir = checkpoint_root / f"cycle_{cycle}"
    log_dir = log_root / f"cycle_{cycle}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = int(config["general"].get("num_gpus", 4))
    grpo_cfg = resolve_path(project_root, config["grpo"]["config"])
    dpo_cfg = resolve_path(project_root, config["dpo"]["config"])
    num_collect = int(config["grpo"].get("num_tasks_collect", 2000))
    threshold = float(config["dpo"].get("threshold", 6.0))
    eval_cfg = config.get("eval", {})
    mbpp_plus = resolve_path(
        project_root, eval_cfg.get("mbpp_plus", "./data/mbpp_plus_test.jsonl")
    )
    lcb_easy = resolve_path(project_root, eval_cfg.get("lcb_easy", "./data/lcb_easy.jsonl"))
    n_bench = int(eval_cfg.get("num_tasks_benchmark", 200))
    n_read = int(eval_cfg.get("num_tasks_readability", 50))
    base_model = str(resolve_path(project_root, config["general"]["base_model"]))
    use_vllm = bool(config.get("inference", {}).get("use_vllm", False))
    if dry_run:
        use_vllm = False

    py = _py()
    dry = _flag(dry_run)
    trainer_resume = " --resume_from_checkpoint" if resume_phase else ""

    if phase == "phase1_grpo":
        model = get_model_path(cycle, phase, config, start_model=start_model)
        out = cycle_dir / "model_rl"
        if dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "DRY_RUN_PLACEHOLDER").write_text(
                f"dry_run grpo from {model}\n"
                "THIS IS NOT A TRAINED MODEL. Real runs refuse to load this directory.\n",
                encoding="utf-8",
            )
            print(
                f"[dry_run] Skipped GRPO training; wrote explicit placeholder at {out}. "
                "Do not resume a real run from this directory."
            )
            return
        train_script = project_root / "src" / "train_grpo.py"
        if not train_script.exists():
            raise FileNotFoundError(
                f"Missing {train_script}. Implement GRPO training before running without --dry_run."
            )
        # Export so train_grpo fail-fast divisibility check sees the real world size.
        os.environ["ACCELERATE_NUM_PROCESSES"] = str(num_gpus)
        run_cmd(
            f"accelerate launch --num_processes {num_gpus} {shlex.quote(str(train_script))} "
            f"--config {shlex.quote(str(grpo_cfg))} "
            f"--model {shlex.quote(model)} --output {shlex.quote(str(out))}"
            f"{trainer_resume}",
            log_dir / "grpo.log",
            cwd=project_root,
        )
        if not (out / "adapter_config.json").exists():
            raise RuntimeError(
                f"GRPO finished but {out}/adapter_config.json is missing. "
                "Refusing to mark phase complete."
            )

    elif phase == "phase1_eval":
        rl_path = cycle_dir / "model_rl"
        if not dry_run:
            from utils.failfast import assert_not_dry_run_placeholder

            assert_not_dry_run_placeholder(rl_path, what="phase1_eval rl_model")
            if not (rl_path / "adapter_config.json").exists():
                raise RuntimeError(
                    f"phase1_eval: missing LoRA adapter at {rl_path}/adapter_config.json"
                )
        out = cycle_dir / "phase1_check.json"
        mock = " --mock_judge" if dry_run else ""
        if dry_run and not mock.strip():
            raise RuntimeError("internal: dry_run eval must pass --mock_judge explicitly")
        run_cmd(
            f"{py} src/evaluate.py --mode hypothesis_check "
            f"--base_model {shlex.quote(base_model)} "
            f"--rl_model {shlex.quote(str(rl_path))} "
            f"--eval_data {shlex.quote(str(mbpp_plus))} "
            f"--num_tasks_benchmark {n_bench} --num_tasks_readability {n_read} "
            f"--output {shlex.quote(str(out))}{dry}{mock}",
            log_dir / "phase1_eval.log",
            cwd=project_root,
        )

    elif phase == "phase2_collect":
        rl_path = cycle_dir / "model_rl"
        if not dry_run:
            from utils.failfast import assert_not_dry_run_placeholder

            assert_not_dry_run_placeholder(rl_path, what="phase2_collect model")
            if not (rl_path / "adapter_config.json").exists():
                raise RuntimeError(
                    f"phase2_collect: missing LoRA adapter at {rl_path}/adapter_config.json"
                )
        collect_model = rl_path
        if use_vllm and not dry_run:
            merged = cycle_dir / "model_rl_merged"
            merge_script = project_root / "src" / "merge_lora.py"
            run_cmd(
                f"{py} {shlex.quote(str(merge_script))} "
                f"--base_model {shlex.quote(base_model)} "
                f"--adapter {shlex.quote(str(rl_path))} "
                f"--output {shlex.quote(str(merged))}",
                log_dir / "merge_lora.log",
                cwd=project_root,
            )
            collect_model = merged
        out = trace_root / f"cycle_{cycle}_traces.jsonl"
        # Fresh collect for this cycle — avoid mixing stale task_ids from older models.
        if out.exists() and not dry_run:
            out.unlink()
        train_tasks = resolve_grpo_train_file(config, project_root)
        # Prefer GRPO reward_audit rollouts (collect_traces auto-harvests).
        # --live is only for the old second sampling pass.
        run_cmd(
            f"{py} src/collect_traces.py "
            f"--model {shlex.quote(str(collect_model))} "
            f"--base_model_path {shlex.quote(base_model)} "
            f"--tasks {shlex.quote(str(train_tasks))} "
            f"--output {shlex.quote(str(out))} "
            f"--num_tasks {num_collect} "
            f"--use_vllm {'true' if use_vllm else 'false'}"
            f"{dry}",
            log_dir / "collect.log",
            cwd=project_root,
        )

    elif phase == "phase3_regen":
        traces = trace_root / f"cycle_{cycle}_traces.jsonl"
        out = dpo_pairs_root / f"cycle_{cycle}_raw.jsonl"
        if out.exists() and not dry_run:
            out.unlink()
        run_cmd(
            f"{py} src/regen_collaboration.py "
            f"--base_model {shlex.quote(base_model)} "
            f"--traces {shlex.quote(str(traces))} "
            f"--output {shlex.quote(str(out))} "
            f"--use_vllm {'true' if use_vllm else 'false'}"
            f"{dry}",
            log_dir / "regen.log",
            cwd=project_root,
        )

    elif phase == "phase3_filter":
        raw = dpo_pairs_root / f"cycle_{cycle}_raw.jsonl"
        out = dpo_pairs_root / f"cycle_{cycle}_filtered.jsonl"
        mock = " --mock_judge" if dry_run else ""
        min_pairs = int(config.get("dpo", {}).get("min_pairs", 1))
        if dry_run:
            min_pairs = 1
        # DeepSeek is eval-only. Pair construction is structural.
        run_cmd(
            f"{py} src/filter_pairs.py "
            f"--raw_pairs {shlex.quote(str(raw))} "
            f"--output {shlex.quote(str(out))} "
            f"--judge_api none "
            f"--min_pairs {min_pairs}{mock}",
            log_dir / "filter.log",
            cwd=project_root,
        )

    elif phase == "phase3_dpo":
        model = get_model_path(cycle, phase, config, start_model=start_model)
        dpo_data = dpo_pairs_root / f"cycle_{cycle}_filtered.jsonl"
        out = cycle_dir / "model_rl_dpo"
        if dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "DRY_RUN_PLACEHOLDER").write_text(
                f"dry_run dpo from {model}\n"
                "THIS IS NOT A TRAINED MODEL. Real runs refuse to load this directory.\n",
                encoding="utf-8",
            )
            print(
                f"[dry_run] Skipped DPO training; wrote explicit placeholder at {out}."
            )
            return
        from utils.failfast import assert_not_dry_run_placeholder

        assert_not_dry_run_placeholder(model, what="phase3_dpo --model")
        train_script = project_root / "src" / "train_dpo.py"
        if not train_script.exists():
            raise FileNotFoundError(
                f"Missing {train_script}. Implement DPO training before running without --dry_run."
            )
        os.environ["ACCELERATE_NUM_PROCESSES"] = str(num_gpus)
        run_cmd(
            f"accelerate launch --num_processes {num_gpus} {shlex.quote(str(train_script))} "
            f"--config {shlex.quote(str(dpo_cfg))} "
            f"--model {shlex.quote(model)} "
            f"--dpo_data {shlex.quote(str(dpo_data))} "
            f"--output {shlex.quote(str(out))}{trainer_resume}",
            log_dir / "dpo.log",
            cwd=project_root,
        )
        if not (out / "adapter_config.json").exists():
            raise RuntimeError(
                f"DPO finished but {out}/adapter_config.json is missing. "
                "Refusing to mark phase complete."
            )

    elif phase == "phase4_eval":
        if not dry_run:
            from utils.failfast import assert_not_dry_run_placeholder

            assert_not_dry_run_placeholder(
                cycle_dir / "model_rl", what="phase4_eval rl_model"
            )
            assert_not_dry_run_placeholder(
                cycle_dir / "model_rl_dpo", what="phase4_eval final_model"
            )
            for tag, p in (
                ("rl", cycle_dir / "model_rl"),
                ("final", cycle_dir / "model_rl_dpo"),
            ):
                if not (p / "adapter_config.json").exists():
                    raise RuntimeError(
                        f"phase4_eval: missing adapter_config.json for {tag} at {p}"
                    )
        out = results_root / f"cycle_{cycle}_eval.json"
        mock = " --mock_judge" if dry_run else ""
        run_cmd(
            f"{py} src/evaluate.py --mode full "
            f"--models base,rl,final "
            f"--base_model {shlex.quote(base_model)} "
            f"--rl_model {shlex.quote(str(cycle_dir / 'model_rl'))} "
            f"--final_model {shlex.quote(str(cycle_dir / 'model_rl_dpo'))} "
            f"--eval_data {shlex.quote(str(mbpp_plus))} "
            f"--lcb_data {shlex.quote(str(lcb_easy))} "
            f"--num_tasks_benchmark {n_bench} --num_tasks_readability {n_read} "
            f"--cycle {cycle} "
            f"--output {shlex.quote(str(out))}{dry}{mock}",
            log_dir / "eval.log",
            cwd=project_root,
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Decoupled-collab master pipeline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cycle_id", type=int, default=None)
    parser.add_argument("--start_model", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Pass dry_run/mock_judge to children; skip GPU training phases",
    )
    parser.add_argument(
        "--only_phase",
        default=None,
        help="Run a single phase name (for smoke tests)",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()

    config = load_config(config_path)
    project_root = Path(config["_project_root"])
    os.chdir(project_root)

    if not args.dry_run:
        from utils.benchmarks import require_real_benchmarks
        from utils.failfast import ConfigError

        eval_cfg = config.get("eval", {})
        overrides = {
            "mbpp_train": resolve_grpo_train_file(config, project_root),
            "mbpp_plus": resolve_path(
                project_root, eval_cfg.get("mbpp_plus", "./data/mbpp_plus_test.jsonl")
            ),
            "lcb_easy": resolve_path(
                project_root, eval_cfg.get("lcb_easy", "./data/lcb_easy.jsonl")
            ),
        }
        try:
            require_real_benchmarks(
                project_root,
                ["mbpp_train", "mbpp_plus", "lcb_easy"],
                path_overrides=overrides,
                allow_synthetic=False,
            )
        except ConfigError as e:
            raise SystemExit(
                f"[error] Pipeline refuses to start without REAL benchmark data.\n{e}\n"
                "Run: python src/prepare_data.py --download"
            ) from e

    pipeline_state_file = state_path(
        project_root, config.get("general", {}).get("state_file")
    )
    state = load_state(
        project_root,
        resume=args.resume,
        cycle_id=args.cycle_id,
        state_file=pipeline_state_file,
    )
    if args.resume and state.get("status") == "completed":
        raise SystemExit(
            "[error] pipeline_state.json status=completed. Refusing --resume "
            "(would re-run finished cycles). Delete the state file to start fresh."
        )
    start_cycle = int(state.get("current_cycle", 0))
    config["_start_cycle"] = start_cycle
    num_cycles = int(config["general"].get("num_cycles", 2))

    if start_cycle >= num_cycles:
        raise SystemExit(
            f"[error] current_cycle={start_cycle} >= num_cycles={num_cycles}. "
            "Nothing left to run; delete pipeline_state.json to restart."
        )

    if args.only_phase:
        if args.only_phase not in PHASES:
            raise SystemExit(f"--only_phase must be one of {PHASES}")
        cycle = start_cycle
        state["current_cycle"] = cycle
        state["current_phase"] = args.only_phase
        save_state(project_root, state, state_file=pipeline_state_file)
        print(f"\n{'=' * 60}\n  Cycle {cycle} | Phase: {args.only_phase}\n{'=' * 60}\n")
        run_phase(
            args.only_phase,
            cycle,
            config,
            start_model=args.start_model,
            dry_run=args.dry_run,
            resume_phase=args.resume,
        )
        advance_state_after_phase(state, args.only_phase, cycle)
        save_state(project_root, state, state_file=pipeline_state_file)
        print("✓ Single phase completed")
        return

    for cycle in range(start_cycle, num_cycles):
        state["current_cycle"] = cycle
        # On first cycle of this run, continue from saved *next* phase; later cycles restart
        if cycle == start_cycle:
            phases_to_run = phases_from(state.get("current_phase"))
        else:
            phases_to_run = list(PHASES)

        for phase_index, phase in enumerate(phases_to_run):
            # Persist the phase we are about to run (crash mid-phase → re-run same phase).
            state["current_phase"] = phase
            state["status"] = "running"
            save_state(project_root, state, state_file=pipeline_state_file)

            print(f"\n{'=' * 60}")
            print(f"  Cycle {cycle} | Phase: {phase}")
            print(f"{'=' * 60}\n")

            run_phase(
                phase,
                cycle,
                config,
                start_model=args.start_model,
                dry_run=args.dry_run,
                resume_phase=(
                    args.resume and cycle == start_cycle and phase_index == 0
                ),
            )
            # Advance to next phase *after* success so --resume skips completed work.
            advance_state_after_phase(state, phase, cycle)
            save_state(project_root, state, state_file=pipeline_state_file)

    state["status"] = "completed"
    save_state(project_root, state, state_file=pipeline_state_file)
    print("\n✓ All cycles completed!")


if __name__ == "__main__":
    main()
