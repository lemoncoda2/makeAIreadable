#!/usr/bin/env bash
# Setup conda/pip env for Qwen3-4B on V100-32G (sm_70). See GOAL Step 0.2.
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

TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_VER="${TORCH_VER:-2.5.1}"

echo "==> Installing PyTorch ${TORCH_VER} from ${TORCH_CUDA_INDEX}"
pip install "torch==${TORCH_VER}" --index-url "${TORCH_CUDA_INDEX}" || {
  echo "[error] torch install failed. Set TORCH_VER/TORCH_CUDA_INDEX or install manually."
  exit 1
}

echo "==> pip install -r requirements.txt"
pip install -r requirements.txt

if [[ "${INSTALL_VLLM:-1}" == "1" ]]; then
  echo "==> Installing optional vllm==0.8.5 (Qwen3 + V100 pin). Set INSTALL_VLLM=0 to skip."
  pip install "vllm==0.8.5" || {
    echo "[warn] vllm==0.8.5 install failed. Continuing with HF-only inference."
    echo "       Keep configs/pipeline_config.yaml inference.use_vllm: false"
  }
fi

echo "==> Verifying CUDA / arch / transformers"
python3 - <<'PY'
import torch, transformers
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA not available after install")
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    print(f"  gpu{i}: {name} sm_{major}{minor}")
    if (major, minor) != (7, 0):
        print(f"  [warn] Expected V100 sm_70; got sm_{major}{minor}. Re-validate vLLM/torch pins.")
ver = transformers.__version__
print(f"transformers={ver}")
parts = [int(x) for x in ver.split(".")[:2]]
if parts < [4, 51]:
    raise SystemExit(f"[error] transformers>={4}.{51} required for Qwen3; got {ver}")
try:
    import vllm
    print(f"vllm={vllm.__version__}")
    print("Remember: export VLLM_USE_V1=0 on V100 before using vLLM")
except ImportError:
    print("vllm not installed (HF-only OK)")
print("✓ setup_env.sh checks passed")
PY

echo "Remember: export DEEPSEEK_API_KEY=... WANDB_API_KEY=... HF_HOME=..."
echo "Optional: export VLLM_USE_V1=0"
echo "✓ setup_env.sh finished"
