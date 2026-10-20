@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=src
echo ============================================================
echo  心电分析智能体 - 网页端
echo  启动后浏览器打开 http://localhost:8600
echo  （端口用 8600：8501 落在 Windows 保留端口段 8457-8556 里，
echo    绑定会报 WinError 10013，别改回 8501）
echo  关闭本窗口即停止服务
echo ============================================================
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8600 --server.headless false --browser.gatherUsageStats false
pause
