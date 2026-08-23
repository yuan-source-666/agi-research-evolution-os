@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo SEED OS v2 启动中...
where py >nul 2>nul && (py -3 run.py --mode interactive & goto end)
where python >nul 2>nul && (python run.py --mode interactive & goto end)
"C:\Users\[USER]\.workbuddy\binaries\python\versions\3.13.12\python.exe" run.py --mode interactive
:end
echo.
pause
