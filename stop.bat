@echo off
cd /d "%~dp0"

if not exist vox.pid (
    echo Vox is not running.
    pause
    exit /b 1
)

set /p PID=<vox.pid
taskkill /PID %PID% >nul 2>&1

if %errorlevel% equ 0 (
    echo Vox stopped (PID %PID%).
) else (
    echo Vox process (PID %PID%) not found. Cleaning up stale PID file.
)

del vox.pid >nul 2>&1
pause
