@echo off
title AIR-PEN Studio
echo ============================================================
echo   Launching AIR-PEN Studio...
echo ============================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe AIR_PEN\main.py
) else (
    python AIR_PEN\main.py
)
pause
