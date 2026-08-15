#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_NAME="${1:-cycle0_grpo}"
RUN_DIR="logs/$RUN_NAME"
PID_FILE="$RUN_DIR/launcher.pid"
LOG_FILE="$RUN_DIR/train.log"

echo "=== training status: $(date --iso-8601=seconds) ==="
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "launcher: alive pid=$pid"
  else
    echo "launcher: not alive (recorded pid=${pid:-empty})"
  fi
else
  echo "launcher: PID file missing"
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>&1 || true

echo "--- training processes ---"
ps -eo pid,ppid,etimes,%cpu,%mem,stat,args | grep -E '[a]ccelerate launch|[t]rain_grpo.py' || true

echo "--- recent metrics/errors ---"
if [[ -f "$LOG_FILE" ]]; then
  grep -E 'reward|kl|loss|Traceback|CUDA out of memory|NCCL|Error|error' "$LOG_FILE" \
    | tail -n 40 || true
  echo "--- log tail ---"
  tail -n 40 "$LOG_FILE"
else
  echo "log missing: $LOG_FILE"
fi

echo "--- checkpoints ---"
find checkpoints -maxdepth 4 -type f \
  \( -name 'adapter_model.safetensors' -o -name 'trainer_state.json' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20

echo "--- disk ---"
df -h "$ROOT" | tail -n 1
