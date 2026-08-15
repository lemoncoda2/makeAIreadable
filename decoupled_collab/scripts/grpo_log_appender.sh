#!/bin/bash
ROOT=/root/makeAIreadable-20260814/workspace/decoupled_collab
LOG=$ROOT/logs/cycle_0/grpo_log_appender.out
PY=$ROOT/.venv-prod/bin/python
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] appender start pid=$$" >> "$LOG"
while true; do
  "$PY" "$ROOT/scripts/extract_grpo_log.py" >> "$LOG" 2>&1
  # stop quietly once a real adapter exists and train_grpo is gone
  if [ -f "$ROOT/checkpoints/cycle_0/model_rl/adapter_config.json" ] \
     && ! pgrep -f 'src/train_grpo.py' >/dev/null 2>&1; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GRPO finished; final snapshot done" >> "$LOG"
    exit 0
  fi
  sleep 600
done
