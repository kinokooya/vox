@echo off
cd /d "%~dp0"
start "" /B .venv\Scripts\pythonw.exe -m vox %*
