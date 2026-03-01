@echo off
cd /d "%~dp0"

set KILLED=0

REM 1. PID ファイルがあれば使う
if exist vox.pid (
    for /f %%p in (vox.pid) do (
        taskkill /F /PID %%p >nul 2>&1
        if not errorlevel 1 set KILLED=1
    )
    del vox.pid >nul 2>&1
)

REM 2. コマンドラインで vox プロセスを検出（フォールバック）
for /f %%p in ('powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or Name='python.exe'\" | Where-Object { $_.CommandLine -match '-m\s+vox' } | Select-Object -ExpandProperty ProcessId" 2^>nul') do (
    taskkill /F /PID %%p >nul 2>&1
    set KILLED=1
)

if %KILLED% equ 1 (
    echo Vox stopped.
) else (
    echo Vox is not running.
)
pause
