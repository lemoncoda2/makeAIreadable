#!/bin/bash
# Server-local: print ONLY DONE / FAILED / CANCELLED.
# DONE = final LoRA adapter exists and train_grpo is gone,
#      or a valid Trainer checkpoint-N exists and train_grpo is gone (resume path).
set +e
ROOT=/root/makeAIreadable-20260814/workspace/decoupled_collab
RL="$ROOT/checkpoints/cycle_0/model_rl"
LOG="$ROOT/logs/cycle_0/grpo.log"

adapter_ok() {
  [ -f "$RL/adapter_config.json" ] || return 1
  ls "$RL"/adapter_model* >/dev/null 2>&1
}

ckpt_ok() {
  ls "$RL"/checkpoint-*/trainer_state.json >/dev/null 2>&1
}

grpo_alive() {
  pgrep -f 'src/train_grpo.py' >/dev/null 2>&1
}

pipeline_alive() {
  pgrep -f 'python src/run_pipeline.py --config configs/pipeline_config.yaml' >/dev/null 2>&1
}

while true; do
  if adapter_ok && ! grpo_alive; then
    echo DONE
    exit 0
  fi
  if ckpt_ok && ! grpo_alive; then
    echo DONE
    exit 0
  fi
  if ! grpo_alive && ! pipeline_alive && ! adapter_ok && ! ckpt_ok; then
    if grep -E 'CUDA out of memory|Traceback \(most recent call last\)' "$LOG" >/dev/null 2>&1; then
      echo FAILED
      exit 1
    fi
    echo FAILED
    exit 1
  fi
  sleep 180
done
