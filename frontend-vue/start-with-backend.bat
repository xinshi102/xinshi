@echo off
echo 正在启动气象数据可视化系统 - 完整版...
echo.

REM 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到Node.js，请先安装Node.js
    echo 下载地址：https://nodejs.org/
    pause
    exit /b 1
)

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到Python，请先安装Python
    echo 下载地址：https://python.org/
    pause
    exit /b 1
)

REM 检查是否已安装依赖
if not exist "node_modules" (
    echo 正在安装前端依赖包...
    npm install
    if %errorlevel% neq 0 (
        echo 错误：前端依赖安装失败
        pause
        exit /b 1
    )
)

echo 启动后端服务器...
echo 后端将在 http://localhost:5000 启动
echo.

REM 启动后端服务器（在后台运行）
start "Backend Server" cmd /k "cd ..\backend && python app.py"

REM 等待后端启动
timeout /t 3 /nobreak >nul

echo 启动前端开发服务器...
echo 前端将在 http://localhost:3000 启动
echo 按 Ctrl+C 停止服务器
echo.

npm run dev

pause 