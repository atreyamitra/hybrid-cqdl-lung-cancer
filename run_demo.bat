@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
python -m unittest discover -s tests -v
if errorlevel 1 goto fail
if exist demo_data rmdir /s /q demo_data
if exist runs\demo_hybrid rmdir /s /q runs\demo_hybrid
python -m src.demo_data --out demo_data
if errorlevel 1 goto fail
python -m src.train --manifest demo_data/manifest.csv --out runs/demo_hybrid --model hybrid --epochs 1 --data-kind synthetic
if errorlevel 1 goto fail
python -m src.predict --checkpoint runs/demo_hybrid/best.pt --image demo_data/fixture_23.npy
if errorlevel 1 goto fail
start "" DEMO.html
pause
exit /b 0
:fail
echo Setup or verification failed. Check the error above. Python 3.12 and internet are required for initial installation.
pause
exit /b 1
