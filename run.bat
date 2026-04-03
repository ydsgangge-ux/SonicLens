@echo off
chcp 65001 >nul
echo ================================================
echo   librosa 音频分析仪 - 启动服务
echo ================================================
echo.

call venv\Scripts\activate

echo [启动] 正在启动服务器...
echo [提示] 访问地址: http://localhost:8000
echo [提示] 按 Ctrl+C 停止服务
echo.
python app.py
