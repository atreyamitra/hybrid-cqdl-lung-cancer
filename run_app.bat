@echo off
setlocal
cd /d "%~dp0"
REM Launch the local Hybrid CQDL app (inference + experiment results).
REM Not executed on Windows by the authors; the same commands were executed on Linux.
set PY=python
where py >nul 2>nul && py -3.12 -c "import sys" >nul 2>nul && set PY=py -3.12
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat
python -c "import torch" >nul 2>nul
if errorlevel 1 (
  python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 python -m pip install torch==2.14.0
  if errorlevel 1 goto fail
)
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist demo_data\manifest.csv python -m src.demo_data --out demo_data
if not exist experiments\endsem_synthetic\aggregate.json (
  echo No saved experiment found: running the 3-seed synthetic experiment once (about a minute on a laptop CPU^)...
  python -m src.experiments run --manifest demo_data/manifest.csv --out experiments/endsem_synthetic --data-kind synthetic --configs matched hybrid_L1 hybrid_L3 classical --resume
  if errorlevel 1 goto fail
)
python -m streamlit run app\streamlit_app.py
exit /b 0
:fail
echo Setup failed. Check the error above. Python 3.12 and internet are required for the first installation.
pause
exit /b 1
