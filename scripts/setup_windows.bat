@echo off
REM Sketch2Shape AI - Windows setup (run from the project folder, inside an activated venv)
REM Requires: Python 3.10-3.12, Git, an NVIDIA GPU driver. Install CUDA PyTorch first:
REM   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -c "import torch" 2>NUL || (echo Install PyTorch first - see comment at top of this file & exit /b 1)
python -m pip install -r requirements.txt || exit /b 1
if not exist third_party mkdir third_party
if not exist third_party\TripoSR git clone https://github.com/VAST-AI-Research/TripoSR.git third_party\TripoSR
pushd third_party\TripoSR
git checkout 107cefdc244c39106fa830359024f6a2f1c78871 2>NUL
popd
REM torchmcubes needs Visual Studio Build Tools + CUDA toolkit; skipped on Windows on purpose.
REM Sketch2Shape automatically uses its PyMCubes / scikit-image marching-cubes fallback instead.
python scripts\check_env.py
python scripts\download_models.py
if not exist .env copy .env.example .env
echo.
echo Setup complete. Start the app with:  python app.py
