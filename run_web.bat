@echo off
title AIR-PEN Web Studio
echo ============================================================
echo   Starting AIR-PEN Web Studio on localhost:8000...
echo ============================================================
cd /d "%~dp0"
start http://localhost:8000
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe AIR_PEN\backend\server.py
) else (
    python AIR_PEN\backend\server.py
)
pause
