@echo off
REM SatQuery AI — Windows Lab GPU Environment Setup
echo ==========================================================
echo SatQuery AI — Windows Lab GPU Provisioning
echo ==========================================================

python -m venv venv_satquery
call venv_satquery\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ==========================================================
echo Running Diagnostics...
python training\launcher.py --task eval

echo ==========================================================
echo Environment Ready!
echo Run: python training\launcher.py --task all
echo ==========================================================
pause
