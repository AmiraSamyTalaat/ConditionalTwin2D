@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set OMP_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set MKL_NUM_THREADS=2
set NUMEXPR_NUM_THREADS=2

set PYEXE=
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if defined PYEXE goto have_python

where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv .venv
) else (
  where python >nul 2>nul
  if errorlevel 1 goto no_python
  python -m venv .venv
)
if errorlevel 1 goto failed
set "PYEXE=.venv\Scripts\python.exe"

:have_python
"%PYEXE%" -c "import numpy,pandas,scipy,sklearn,matplotlib,yaml" >nul 2>nul
if errorlevel 1 (
  "%PYEXE%" -m pip install --upgrade pip
  if errorlevel 1 goto failed
  "%PYEXE%" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)

"%PYEXE%" -u scripts\run_transport_rank.py
if errorlevel 1 goto failed
"%PYEXE%" scripts\verify_stage.py
if errorlevel 1 goto failed

echo.
echo ==================================================
echo SUCCESS. Upload outputs\RESULT_PACKAGE.zip
echo ==================================================
pause
exit /b 0

:no_python
echo Python 3 was not found. Install Python 3.11 or newer, then run this file again.
goto failed

:failed
echo.
echo ==================================================
echo FAILED. Read outputs\logs\ERROR_TRACEBACK.txt and outputs\VERIFICATION.json
echo ==================================================
pause
exit /b 1
