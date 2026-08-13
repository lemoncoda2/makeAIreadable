#!/usr/bin/env bash
# Setup conda / pip environment for decoupled_collab (GOAL Step 0.2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Project root: $ROOT"

if command -v conda >/dev/null 2>&1; then
  echo "==> Creating/updating conda env 'collab' (python=3.11)"
  conda create -n collab python=3.11 -y 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate collab
else
  echo "==> conda not found; using current Python: $(python3 --version 2>/dev/null || python --version)"
fi

echo "==> Installing PyTorch 2.4.0 (cu121). Skip if already installed."
echo "    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 || {
  echo "[warn] torch cu121 install failed or skipped; continuing with requirements.txt"
}

echo "==> pip install -r requirements.txt"
pip install -r requirements.txt

echo "==> Verifying CUDA (optional)"
python3 - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
except Exception as e:
    print("torch check skipped:", e)
PY

echo "✓ setup_env.sh finished"
echo "Remember: export DEEPSEEK_API_KEY=... WANDB_API_KEY=... HF_HOME=..."
