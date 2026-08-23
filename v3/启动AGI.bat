@echo off
chcp 65001 >nul
cd /d %~dp0
echo === v3 基元社会 · 交互模式（「帮助」看协议，「退出」结束） ===
python run.py --mode live
pause
