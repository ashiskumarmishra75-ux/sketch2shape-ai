#!/usr/bin/env bash
# Sketch2Shape AI - one-shot setup (Linux / macOS / Colab / WSL).
#   bash scripts/setup.sh              # deps + TripoSR + model pre-download
#   bash scripts/setup.sh --no-download
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TRIPOSR_COMMIT="107cefdc244c39106fa830359024f6a2f1c78871"   # verified VAST-AI-Research/TripoSR HEAD
PY="${PYTHON:-python}"

echo "==> [1/5] Python: $($PY --version)"
if ! $PY -c "import torch" 2>/dev/null; then
  echo "!! PyTorch is not installed. Install the CUDA build for your system first, e.g.:"
  echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
  exit 1
fi
$PY -c "import torch; print('    torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "==> [2/5] Installing Python requirements"
$PY -m pip install -q -r requirements.txt

echo "==> [3/5] Cloning TripoSR (MIT) into third_party/TripoSR"
mkdir -p third_party
if [ ! -d third_party/TripoSR/.git ]; then
  git clone -q https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR
fi
git -C third_party/TripoSR fetch -q --depth 1 origin "$TRIPOSR_COMMIT" 2>/dev/null || true
git -C third_party/TripoSR checkout -q "$TRIPOSR_COMMIT" 2>/dev/null || echo "    (using current TripoSR checkout)"

echo "==> [4/5] Building torchmcubes (optional - a PyMCubes/scikit-image fallback is used if this fails)"
if $PY -c "import torchmcubes" 2>/dev/null; then
  echo "    torchmcubes already installed"
else
  $PY -m pip install -q --no-build-isolation "git+https://github.com/tatsy/torchmcubes.git" > third_party/torchmcubes_build.log 2>&1 \
    && echo "    torchmcubes built" \
    || echo "    torchmcubes build failed -> Sketch2Shape will use its built-in marching-cubes fallback (fine)"
fi

echo "==> [5/5] Environment check"
$PY scripts/check_env.py || true

if [[ "${1:-}" != "--no-download" ]]; then
  echo "==> Pre-downloading model weights (~7 GB, cached in ~/.cache/huggingface)"
  $PY scripts/download_models.py
fi

[ -f .env ] || cp .env.example .env
echo
echo "Setup complete. Start the app with:  python app.py   (add --share on Colab)"
