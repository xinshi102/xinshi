#!/bin/bash

echo "正在启动气象数据可视化系统 - Vue版本..."
echo

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "错误：未检测到Node.js，请先安装Node.js"
    echo "下载地址：https://nodejs.org/"
    exit 1
fi

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "正在安装依赖包..."
    npm install
    if [ $? -ne 0 ]; then
        echo "错误：依赖安装失败"
        exit 1
    fi
fi

echo "启动开发服务器..."
echo "应用将在 http://localhost:3000 启动"
echo "按 Ctrl+C 停止服务器"
echo

npm run dev 