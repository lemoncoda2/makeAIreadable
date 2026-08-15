#!/bin/bash
# Server-local handoff: after cycle-0 GRPO, harvest → regen → filter → DPO → full eval.
# Does not delete reward_audit. Does not restart healthy GRPO from step 0.
# Stops the parent before cycle 1. Writes TRAIN_REPORT.md when eval JSON exists.
set +e
ROOT=/root/makeAIreadable-20260814/workspace/decoupled_collab
LOGDIR="$ROOT/logs"
STATUS="$LOGDIR/handoff_status.txt"
PIDFILE="$LOGDIR/handoff.pid"
PY="$ROOT/.venv-prod/bin/python"
export PATH="$ROOT/.venv-prod/bin:$PATH"
mkdir -p "$LOGDIR/cycle_0"
cd "$ROOT"
# Children inherit this. Needed if we --resume after the original parent dies.
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi
echo $$ > "$PIDFILE"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$STATUS"
}

pipeline_pid() {
  pgrep -f "src/run_pipeline.py --config configs/pipeline_config.yaml" 2>/dev/null | head -1
}

grpo_alive() {
  pgrep -f "src/train_grpo.py --config" >/dev/null 2>&1
}

dpo_alive() {
  pgrep -f "src/train_dpo.py --config" >/dev/null 2>&1
}

eval_alive() {
  pgrep -f "src/evaluate.py" >/dev/null 2>&1
}

adapter_ok() {
  [ -f "$ROOT/checkpoints/cycle_0/model_rl/adapter_config.json" ] && \
  ls "$ROOT/checkpoints/cycle_0/model_rl"/adapter_model* >/dev/null 2>&1
}

final_ok() {
  [ -f "$ROOT/checkpoints/cycle_0/model_rl_dpo/adapter_config.json" ] && \
  ls "$ROOT/checkpoints/cycle_0/model_rl_dpo"/adapter_model* >/dev/null 2>&1
}

eval_json() {
  if [ -f "$ROOT/results/cycle_0_eval.json" ]; then
    echo "$ROOT/results/cycle_0_eval.json"
  elif [ -f "$ROOT/results/cycle_0_full_eval.json" ]; then
    echo "$ROOT/results/cycle_0_full_eval.json"
  fi
}

state_cycle() {
  python3 -c "import json; print(json.load(open('$ROOT/pipeline_state.json')).get('current_cycle',''))" 2>/dev/null
}

state_phase() {
  python3 -c "import json; print(json.load(open('$ROOT/pipeline_state.json')).get('current_phase',''))" 2>/dev/null
}

write_report_if_ready() {
  if [ -z "$(eval_json)" ]; then
    return 1
  fi
  if [ -f "$ROOT/logs/cycle_0/TRAIN_REPORT.md" ]; then
    return 0
  fi
  log "eval JSON present — writing TRAIN_REPORT.md"
  "$PY" "$ROOT/scripts/write_cycle0_report.py" "$ROOT" >> "$STATUS" 2>&1
  return 0
}

stop_cycle1() {
  local pid cycle phase
  pid=$(pipeline_pid)
  cycle=$(state_cycle)
  phase=$(state_phase)
  if [ -n "$(eval_json)" ] && [ "$cycle" != "0" ]; then
    if [ -n "$pid" ]; then
      log "cycle 0 eval done and cycle=$cycle — stopping parent pid=$pid before cycle 1"
      kill -TERM "$pid" 2>/dev/null
      sleep 4
      kill -KILL "$pid" 2>/dev/null
    fi
    return 0
  fi
  if [ "$cycle" = "1" ]; then
    if grpo_alive && [ ! -f "$ROOT/checkpoints/cycle_1/model_rl/adapter_config.json" ]; then
      log "cycle 1 GRPO detected — stopping (handoff is cycle 0 only)"
      pkill -TERM -f "src/train_grpo.py --config" 2>/dev/null
      [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
    fi
  fi
}

resume_if_parent_dead() {
  local pid
  pid=$(pipeline_pid)
  if [ -n "$pid" ]; then
    return 0
  fi
  if [ -n "$(eval_json)" ] && final_ok; then
    return 0
  fi
  if ! adapter_ok; then
    if grpo_alive; then
      return 0
    fi
    if ls "$ROOT/checkpoints/cycle_0/model_rl"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      log "parent dead, GRPO crashed, valid checkpoint exists — resume"
      nohup "$PY" src/run_pipeline.py --config configs/pipeline_config.yaml --resume \
        >> "$LOGDIR/pipeline_master.log" 2>&1 &
      echo $! > "$LOGDIR/pipeline_resume.pid"
      log "resumed pipeline pid=$!"
    else
      log "parent dead, no adapter, no trainer checkpoint — NOT restarting from step 0"
    fi
    return 0
  fi
  log "parent dead after GRPO adapter saved — resuming remaining phases"
  nohup "$PY" src/run_pipeline.py --config configs/pipeline_config.yaml --resume \
    >> "$LOGDIR/pipeline_master.log" 2>&1 &
  echo $! > "$LOGDIR/pipeline_resume.pid"
  log "resumed pipeline pid=$!"
}

kill_live_collect() {
  local cmd
  cmd=$(ps aux | grep "[c]ollect_traces.py" | head -3)
  if echo "$cmd" | grep -q -- "--live"; then
    log "live collect detected — killing and harvesting instead"
    pkill -f "src/collect_traces.py" 2>/dev/null
    mkdir -p "$ROOT/data/traces" "$LOGDIR/cycle_0"
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

log "handoff start pid=$$ host=$(hostname)"
if [ -n "$DEEPSEEK_API_KEY" ]; then
  log "dotenv loaded (DEEPSEEK_API_KEY=set)"
else
  log "WARNING dotenv missing DEEPSEEK_API_KEY — eval/resume may fail"
fi
log "will let healthy GRPO finish, then harvest/DPO/eval; stop before cycle 1"

while true; do
  {
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
    nvidia-smi --query-gpu=index,name,compute_cap,memory.used,utilization.gpu --format=csv,noheader
    echo "pipeline_pid=$(pipeline_pid) grpo=$(grpo_alive && echo yes || echo no) dpo=$(dpo_alive && echo yes || echo no) eval=$(eval_alive && echo yes || echo no)"
    echo "adapter=$(adapter_ok && echo yes || echo no) final=$(final_ok && echo yes || echo no) eval_json=$(eval_json)"
    echo "--- pipeline_state ---"
    cat "$ROOT/pipeline_state.json" 2>/dev/null
    echo
    echo "--- grpo steps ---"
    grep -oE "[0-9]+/300" "$LOGDIR/cycle_0/grpo.log" 2>/dev/null | tail -3
    echo "audit_lines=$(wc -l "$ROOT/checkpoints/cycle_0/model_rl/reward_audit/"reward_rank*.jsonl 2>/dev/null | tail -1)"
    echo "placeholder=$(find "$ROOT/checkpoints/cycle_0" -name DRY_RUN_PLACEHOLDER 2>/dev/null)"
  } >> "$LOGDIR/handoff_pulse.log"

  kill_live_collect
  stop_cycle1
  resume_if_parent_dead
  write_report_if_ready

  if [ -n "$(eval_json)" ] && final_ok && adapter_ok; then
    if [ ! -f "$LOGDIR/cycle_0/HANDOFF_DONE" ]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGDIR/cycle_0/HANDOFF_DONE"
      log "cycle 0 complete (GRPO+DPO+eval). report: logs/cycle_0/TRAIN_REPORT.md"
    fi
    stop_cycle1
    sleep 180
    continue
  fi

  if adapter_ok || dpo_alive || eval_alive; then
    sleep 60
  else
    sleep 180
  fi
done
