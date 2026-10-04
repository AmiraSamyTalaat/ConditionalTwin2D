@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo DEFECTTWIN_2D - STAGE 2B
echo GRAPH_DERIVED_BASELINE_COMPARISON
echo Version 2.1.0
echo ============================================================

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher "py" was not found.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating isolated Python environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :fail
)

echo [2/5] Installing/verifying dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/5] Verifying Stage 2A FIX1 input...
".venv\Scripts\python.exe" scripts\self_test_stage2b.py
if errorlevel 1 goto :fail

echo [4/5] Extracting graph descriptors and running LOHO baselines...
".venv\Scripts\python.exe" scripts\run_stage2b.py --config config\project_config.yaml
if errorlevel 1 goto :fail

echo [5/5] Verifying outputs...
".venv\Scripts\python.exe" scripts\verify_stage2b.py --output stage2b_output
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 2B FINISHED.
echo Open: stage2b_output\ARABIC_STAGE2B_SUMMARY.md
echo Upload: STAGE2B_RESULTS.zip
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [FAILED] Review stage2b_output\RUN_LOG.txt and ERROR_LOG.txt.
pause
exit /b 1
