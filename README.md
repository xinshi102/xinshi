# 气象数据可视化及分析系统

## 项目概述
基于Vue 3 + Python Flask的气象数据可视化及分析系统，支持天气数据上传、分析、预测和出行建议功能。

## 技术栈
- **前端**: Vue 3 + Vue Router 4 + Vite + ECharts + Bootstrap 5
- **后端**: Python Flask + Pandas + NumPy + PyTorch
- **数据可视化**: ECharts 5.4.3
- **UI框架**: Bootstrap 5.3.0

## 项目结构
```
├── frontend-vue/          # Vue 3前端项目
├── backend/               # Python Flask后端
├── model/                 # 机器学习模型
├── data/                  # 数据文件
├── uploads/               # 上传文件目录
└── requirements.txt       # Python依赖
```

## 快速启动

### 方式一：一键启动（推荐）
```bash
cd frontend-vue
# Windows
start-with-backend.bat

# Linux/Mac
chmod +x start-with-backend.sh
./start-with-backend.sh
```

### 方式二：手动启动
```bash
# 启动后端
cd backend
python app.py

# 启动前端（新终端）
cd frontend-vue
npm install
npm run dev
```

## 访问地址
- 前端: http://localhost:3000
- 后端API: http://localhost:5000

## 功能特性
- 📊 气象数据可视化分析
- 🌤️ 天气预测
- 🚗 出行建议
- 📱 响应式设计
