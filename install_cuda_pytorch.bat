@echo off
echo ============================================================
echo   PyTorch CPU Installation for OOTDiffusion
echo   Intel Iris Xe Graphics ke liye (CPU-only mode)
echo ============================================================
echo.

set PYTHON=d:\Virtual Try On System\ootd_env\Scripts\python.exe
set PIP=d:\Virtual Try On System\ootd_env\Scripts\pip.exe

echo [1/3] Uninstalling any existing PyTorch...
"%PIP%" uninstall torch torchvision torchaudio -y
echo.

echo [2/3] Installing PyTorch 2.0.1 (CPU version)...
"%PIP%" install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
echo.

echo [3/3] Verifying installation...
"%PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('Device: CPU (Intel Iris Xe - no CUDA needed)')"
echo.

echo ============================================================
echo   Done! App ab CPU pe chalega.
echo   Run karne ke liye:
echo     venv\Scripts\activate
echo     python app.py
echo ============================================================
pause
