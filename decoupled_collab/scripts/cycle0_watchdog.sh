#!/bin/bash
# Server-local watchdog for cycle-0 GRPO → DPO → eval.
# Never deletes reward_audit. Never restarts healthy GRPO from step 0.
# Stops the parent before cycle 1 GRPO. Resumes only if the parent died.
set +e
ROOT=/root/makeAIreadable-20260814/workspace/decoupled_collab
LOGDIR="$ROOT/logs"
STATUS="$LOGDIR/watchdog_status.txt"
HANDOFF="$LOGDIR/watchdog_handoff.json"
PY="$ROOT/.venv-prod/bin/python"
export PATH="$ROOT/.venv-prod/bin:$PATH"
mkdir -p "$LOGDIR"
cd "$ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$STATUS"
}

pipeline_pid() {
  pgrep -f "python src/run_pipeline.py --config configs/pipeline_config.yaml" | head -1
}

grpo_alive() {
  pgrep -f "src/train_grpo.py" >/dev/null 2>&1
}

dpo_alive() {
  pgrep -f "src/train_dpo.py" >/dev/null 2>&1
}

eval_alive() {
  pgrep -f "src/evaluate.py" >/dev/null 2>&1
}

collect_cmdline() {
  ps aux | grep -E '[c]ollect_traces.py' | head -3
}

adapter_ok() {
  [ -f "$ROOT/checkpoints/cycle_0/model_rl/adapter_config.json" ] && \
  ls "$ROOT/checkpoints/cycle_0/model_rl"/adapter_model* >/dev/null 2>&1
}

final_ok() {
  [ -f "$ROOT/checkpoints/cycle_0/model_rl_dpo/adapter_config.json" ] && \
  ls "$ROOT/checkpoints/cycle_0/model_rl_dpo"/adapter_model* >/dev/null 2>&1
}

placeholder() {
  find "$ROOT/checkpoints/cycle_0" -name DRY_RUN_PLACEHOLDER 2>/dev/null
}

eval_json() {
  if [ -f "$ROOT/results/cycle_0_eval.json" ]; then
    echo "$ROOT/results/cycle_0_eval.json"
  elif [ -f "$ROOT/results/cycle_0_full_eval.json" ]; then
    echo "$ROOT/results/cycle_0_full_eval.json"
  else
    echo ""
  fi
}

write_handoff() {
  python3 - <<'PY'
import json, os, time, glob, subprocess
from pathlib import Path
root = Path("/root/makeAIreadable-20260814/workspace/decoupled_collab")
state = {}
sp = root / "pipeline_state.json"
if sp.exists():
    state = json.loads(sp.read_text())
grpo_log = root / "logs" / "cycle_0" / "grpo.log"
step = ""
if grpo_log.exists():
    import re
    ms = re.findall(r"(\d+/300)", grpo_log.read_text(errors="replace"))
    step = ms[-1] if ms else ""
def pgrep(pat):
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    pids = [x for x in r.stdout.split() if x]
    return pids
audit0 = root / "checkpoints/cycle_0/model_rl/reward_audit/reward_rank0.jsonl"
adapter = (root / "checkpoints/cycle_0/model_rl/adapter_config.json").exists() and bool(
    glob.glob(str(root / "checkpoints/cycle_0/model_rl/adapter_model*"))
)
final = (root / "checkpoints/cycle_0/model_rl_dpo/adapter_config.json").exists() and bool(
    glob.glob(str(root / "checkpoints/cycle_0/model_rl_dpo/adapter_model*"))
)
eval_json = ""
for cand in [root / "results/cycle_0_eval.json", root / "results/cycle_0_full_eval.json"]:
    if cand.exists():
        eval_json = str(cand)
        break
ph = [str(p) for p in (root / "checkpoints/cycle_0").rglob("DRY_RUN_PLACEHOLDER")]
ppids = pgrep("python src/run_pipeline.py --config configs/pipeline_config.yaml")
payload = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "cycle": state.get("current_cycle"),
    "phase": state.get("current_phase"),
    "grpo_step": step,
    "grpo_alive": bool(pgrep("src/train_grpo.py")),
    "pipeline_pid": ppids[0] if ppids else "",
    "adapter_ok": bool(adapter),
    "final_ok": bool(final),
    "eval_json": eval_json,
    "placeholder": ph,
    "audit_lines": sum(1 for _ in open(audit0, encoding="utf-8")) if audit0.exists() else 0,
}
out = root / "logs" / "watchdog_handoff.json"
out.write_text(json.dumps(payload, indent=2))
PY
}

stop_cycle1() {
  local pid cycle phase
  pid=$(pipeline_pid)
  cycle=$(python3 -c "import json; print(json.load(open('$ROOT/pipeline_state.json')).get('current_cycle',0))" 2>/dev/null)
  phase=$(python3 -c "import json; print(json.load(open('$ROOT/pipeline_state.json')).get('current_phase',''))" 2>/dev/null)
  if [ -n "$(eval_json)" ] && [ "$cycle" != "0" ]; then
    if [ -n "$pid" ]; then
      log "cycle 0 eval present and cycle=$cycle — stopping parent pid=$pid before cycle 1"
      kill -TERM "$pid" 2>/dev/null
      sleep 5
      kill -KILL "$pid" 2>/dev/null
    fi
    return 0
  fi
  if [ "$cycle" = "1" ] && echo "$phase" | grep -q phase1_grpo; then
    if grpo_alive && [ ! -f "$ROOT/checkpoints/cycle_1/model_rl/adapter_config.json" ]; then
      log "cycle 1 GRPO started — stopping it (goal is cycle 0 only)"
      pkill -TERM -f "src/train_grpo.py" 2>/dev/null
      [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
    fi
  fi
}

resume_if_parent_dead() {
  local pid phase
  pid=$(pipeline_pid)
  if [ -n "$pid" ]; then
    return 0
  fi
  if [ -n "$(eval_json)" ] && final_ok; then
    log "eval already present; not resuming"
    return 0
  fi
  if ! adapter_ok; then
    # GRPO still running or crashed without adapter
    if grpo_alive; then
      return 0
    fi
    # crashed — only resume if a trainer checkpoint exists
    if ls "$ROOT/checkpoints/cycle_0/model_rl"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      log "parent dead, GRPO crashed, valid checkpoint exists — resume"
      cd "$ROOT"
      nohup "$PY" src/run_pipeline.py --config configs/pipeline_config.yaml --resume \
        >> "$LOGDIR/pipeline_master.log" 2>&1 &
      echo $! > "$LOGDIR/pipeline_resume.pid"
      log "resumed pipeline pid=$!"
    else
      log "parent dead, no adapter, no trainer checkpoint — NOT restarting from step 0"
    fi
    return 0
  fi
  # adapter exists, parent dead, cycle 0 not fully done
  log "parent dead after GRPO adapter saved — resuming remaining phases"
  cd "$ROOT"
  nohup "$PY" src/run_pipeline.py --config configs/pipeline_config.yaml --resume \
    >> "$LOGDIR/pipeline_master.log" 2>&1 &
  echo $! > "$LOGDIR/pipeline_resume.pid"
  log "resumed pipeline pid=$!"
}

kill_live_collect() {
  local cmd
  cmd=$(collect_cmdline)
  if echo "$cmd" | grep -q -- '--live'; then
    log "live collect detected — killing it: $cmd"
    pkill -f "src/collect_traces.py" 2>/dev/null
    # harvest explicitly
    mkdir -p "$ROOT/data/traces"
    "$PY" src/collect_traces.py \
      --model "$ROOT/checkpoints/cycle_0/model_rl" \
      --base_model_path "$ROOT/models/Qwen3-4B" \
      --tasks "$ROOT/data/mbpp_train.jsonl" \
      --output "$ROOT/data/traces/cycle_0_traces.jsonl" \
      --num_tasks 2000 \
      --use_vllm false \
      >> "$LOGDIR/cycle_0/collect.log" 2>&1
    log "manual harvest finished rc=$?"
  fi
}

log "watchdog start pid=$$"
while true; do
  {
    echo "==== $(date -u) ===="
    nvidia-smi --query-gpu=index,name,compute_cap,memory.used,utilization.gpu --format=csv
    echo "pipeline_pid=$(pipeline_pid) grpo=$(grpo_alive && echo yes || echo no) dpo=$(dpo_alive && echo yes || echo no) eval=$(eval_alive && echo yes || echo no)"
    echo "--- pipeline_state ---"
    cat "$ROOT/pipeline_state.json" 2>/dev/null
    echo
    echo "--- grpo progress ---"
    grep -oE '[0-9]+/300' "$LOGDIR/cycle_0/grpo.log" 2>/dev/null | tail -3
    echo "audit_lines=$(wc -l "$ROOT/checkpoints/cycle_0/model_rl/reward_audit/"reward_rank*.jsonl 2>/dev/null | tail -1)"
    echo "placeholder=$(placeholder)"
    echo "adapter=$(adapter_ok && echo yes || echo no) final=$(final_ok && echo yes || echo no) eval=$(eval_json)"
    echo "collect: $(collect_cmdline)"
    tail -c 400 "$LOGDIR/cycle_0/grpo.log" 2>/dev/null
    echo
  } >> "$LOGDIR/watchdog_pulse.log"

  write_handoff
  kill_live_collect
  stop_cycle1
  resume_if_parent_dead

  # crash fingerprints (do not kill)
  if grep -E 'CUDA out of memory|Traceback \(most recent call last\)' "$LOGDIR/cycle_0/grpo.log" >/dev/null 2>&1; then
    if ! grpo_alive; then
      log "GRPO log has Traceback/OOM and process is dead"
    fi
  fi

  if [ -n "$(eval_json)" ] && final_ok && adapter_ok; then
    log "cycle 0 eval artifact present — watchdog idle"
    write_handoff
    # keep looping slowly so we still stop cycle 1 if parent races
    sleep 120
    continue
  fi
  sleep 120
done
