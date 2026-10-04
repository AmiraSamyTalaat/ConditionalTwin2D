@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo DefectTwin-2D final package verification
py -3 scripts\verify_final_package.py
if errorlevel 1 goto :fail
echo.
echo VERIFICATION SUCCESS
pause
exit /b 0
:fail
echo.
echo VERIFICATION FAILED
pause
exit /b 1
