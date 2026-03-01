@echo off
cd /d "%~dp0"

if not exist vox.pid (
    echo Vox is not running.
    pause
    exit /b 1
)

REM vox.pid contains one PID per line (launcher + real python process).
REM Kill each to ensure the full process tree is terminated.
set KILLED=0
for /f %%p in (vox.pid) do (
    taskkill /F /PID %%p >nul 2>&1
    if not errorlevel 1 set KILLED=1
)

if %KILLED% equ 1 (
    echo Vox stopped.
) else (
    echo Vox process not found. Cleaning up stale PID file.
)

del vox.pid >nul 2>&1
pause
