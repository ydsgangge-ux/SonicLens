@echo off
chcp 65001 >nul
title librosa 音频分析仪
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║      librosa 专业音频分析仪 · 一键启动       ║
echo  ╚══════════════════════════════════════════════╝
echo.

REM ── 1. 检查 Python ──
python --version >nul 2>&1
if errorlevel 1 (
    echo  [X] 未检测到 Python！
    echo.
    echo  请先安装 Python 3.9 或更高版本:
    echo  https://www.python.org/downloads/
    echo  安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER%
echo.

REM ── 2. 创建虚拟环境 ──
if not exist "venv\Scripts\python.exe" (
    echo  [*] 首次运行，正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo  [X] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo  [OK] 虚拟环境已创建
    echo.
)

REM ── 3. 安装依赖 ──
call venv\Scripts\activate.bat

REM 检查是否需要安装（简单判断：看 librosa 是否存在）
python -c "import librosa" >nul 2>&1
if errorlevel 1 (
    echo  [*] 正在安装依赖包（首次约需 1-3 分钟，请耐心等待）...
    pip install -r requirements.txt -q
    if errorlevel 1 (
        echo  [!] 部分包安装失败，尝试升级 pip 后重试...
        pip install --upgrade pip -q
        pip install -r requirements.txt -q
    )
    echo  [OK] 依赖安装完成
    echo.
)

REM ── 4. 检查端口 ──
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo  [!] 端口 8000 已被占用，可能是旧进程
    echo      尝试关闭旧进程...
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
        taskkill /F /PID %%p >nul 2>&1
    )
    timeout /t 1 /nobreak >nul
)

REM ── 5. 启动服务 ──
echo  [*] 正在启动服务...
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  浏览器访问: http://localhost:8000           ║
echo  ║  按 Ctrl+C 停止服务                          ║
echo  ╚══════════════════════════════════════════════╝
echo.

REM 延迟 2 秒后自动打开浏览器
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

python app.py

REM 如果服务异常退出
echo.
echo  服务已停止。按任意键退出...
pause >nul
