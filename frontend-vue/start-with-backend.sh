#!/bin/bash

echo "正在启动气象数据可视化系统 - 完整版..."
echo

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "错误：未检测到Node.js，请先安装Node.js"
    echo "下载地址：https://nodejs.org/"
    exit 1
fi

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "错误：未检测到Python，请先安装Python"
    echo "下载地址：https://python.org/"
    exit 1
fi

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "正在安装前端依赖包..."
    npm install
    if [ $? -ne 0 ]; then
        echo "错误：前端依赖安装失败"
        exit 1
    fi
fi

echo "启动后端服务器..."
echo "后端将在 http://localhost:5000 启动"
echo

# 启动后端服务器（在后台运行）
cd ../backend && python app.py &
BACKEND_PID=$!

# 等待后端启动
sleep 3

echo "启动前端开发服务器..."
echo "前端将在 http://localhost:3000 启动"
echo "按 Ctrl+C 停止服务器"
echo

# 启动前端服务器
npm run dev

# 清理后台进程
kill $BACKEND_PID 2>/dev/null 