@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\verify_stage.py
) else (
  py -3 scripts\verify_stage.py 2>nul || python scripts\verify_stage.py
)
if errorlevel 1 (
  echo VERIFICATION FAILED
  pause
  exit /b 1
)
echo VERIFICATION PASSED
pause
