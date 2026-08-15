#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/grpo_config.yaml}"
RUN_NAME="${RUN_NAME:-cycle0_grpo}"
RUN_DIR="logs/$RUN_NAME"
PID_FILE="$RUN_DIR/launcher.pid"
LOG_FILE="$RUN_DIR/train.log"
PYTHON="${PYTHON:-$ROOT/.venv-prod/bin/python}"
ACCELERATE="${ACCELERATE:-$ROOT/.venv-prod/bin/accelerate}"
RESUME="${RESUME:-0}"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume_from_checkpoint)
fi

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[error] GRPO launcher is already alive: pid=$old_pid"
    exit 1
  fi
  echo "[warn] Removing stale launcher PID file: $PID_FILE (pid=${old_pid:-empty})"
  rm -f -- "$PID_FILE"
fi

[[ -x "$PYTHON" ]] || { echo "[error] Python not executable: $PYTHON"; exit 1; }
[[ -x "$ACCELERATE" ]] || { echo "[error] accelerate not executable: $ACCELERATE"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "[error] Config missing: $CONFIG"; exit 1; }

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[preflight] CUDA and required imports"
"$PYTHON" -c 'import importlib.util, torch; from trl import GRPOConfig, GRPOTrainer; assert importlib.util.find_spec("bitsandbytes") is None, "bitsandbytes must not be installed; use unquantized FP16"; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; print(torch.__version__, [torch.cuda.get_device_capability(i) for i in range(4)])'

echo "[preflight] Real benchmarks and train/eval isolation"
PYTHONPATH=src "$PYTHON" -c 'from pathlib import Path; from utils.benchmarks import require_real_benchmarks; print({k:v["n"] for k,v in require_real_benchmarks(Path("."), ["mbpp_train", "mbpp_plus", "lcb_easy"]).items()})'

echo "[preflight] Training configuration"
WORLD_SIZE=4 "$PYTHON" src/train_grpo.py --config "$CONFIG" --dry_run \
  "${resume_args[@]}"

echo "[launch] $RUN_NAME -> $LOG_FILE"
nohup "$ACCELERATE" launch --num_processes 4 src/train_grpo.py \
  --config "$CONFIG" "${resume_args[@]}" >"$LOG_FILE" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$PID_FILE"
sleep 2

if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "[error] Launcher exited immediately. Last log lines:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

echo "[ok] launcher_pid=$launcher_pid"
echo "[ok] status: bash scripts/training_status.sh $RUN_NAME"
