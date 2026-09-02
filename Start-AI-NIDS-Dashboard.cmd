@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-AI-NIDS-Dashboard.ps1"
if errorlevel 1 (
    echo.
    echo The dashboard launcher reported an error.
    pause
)
endlocal
