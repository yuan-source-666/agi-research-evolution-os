@echo off
rem PRIMORDIA v4 launcher - autonomous evolution demo (double-click me)
chcp 65001 >nul
title PRIMORDIA v4 - Autonomous Evolution Demo
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH. Please install Python 3.8+ first.
  pause
  exit /b 1
)
python -X utf8 run.py --mode demo
echo.
echo [Demo finished. Growth records saved in out\ ]
pause
