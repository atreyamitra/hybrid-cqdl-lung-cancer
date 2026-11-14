@echo off
setlocal
cd /d "%~dp0"
REM Hybrid CQDL demo launcher. Not executed on Windows by the authors; the same
REM commands were executed on Linux (Python 3.12, CPU). See START_HERE.md.
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
python -m unittest discover -s tests -v
if errorlevel 1 goto fail
if not exist demo_data\manifest.csv python -m src.demo_data --out demo_data
if exist results\hybrid\best.pt (
  python -m src.predict --checkpoint results\hybrid\best.pt --image demo_data\fixture_23.npy
  python -m src.diagnose --checkpoint results\hybrid\best.pt --manifest demo_data\manifest.csv
) else (
  if exist runs\launcher_hybrid rmdir /s /q runs\launcher_hybrid
  python -m src.train --manifest demo_data/manifest.csv --out runs/launcher_hybrid --model hybrid --epochs 1 --data-kind synthetic
  if errorlevel 1 goto fail
  python -m src.predict --checkpoint runs\launcher_hybrid\best.pt --image demo_data\fixture_23.npy
)
if errorlevel 1 goto fail
start "" DEMO.html
pause
exit /b 0
:fail
echo Setup or verification failed. Check the error above. Python 3.12 and internet are required for the first installation.
pause
exit /b 1
