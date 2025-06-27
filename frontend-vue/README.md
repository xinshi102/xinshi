# 气象数据可视化及分析系统 - Vue版本

这是一个基于Vue 3的现代化前端应用，用于气象数据的可视化分析和天气预测。

## 技术栈

- **Vue 3** - 渐进式JavaScript框架
- **Vue Router 4** - 官方路由管理器
- **Vite** - 下一代前端构建工具
- **ECharts** - 数据可视化图表库
- **Bootstrap 5** - CSS框架
- **Bootstrap Icons** - 图标库
- **Axios** - HTTP客户端

## 功能特性

### 🏠 首页
- 欢迎界面和功能导航
- 响应式设计，支持移动端
- 平滑的动画效果

### 📊 数据分析
- CSV文件上传和解析
- 多种图表展示：
  - 温度和湿度趋势图
  - 降水量分布饼图
  - 天气类型分布图
  - 云量和风速分布图
  - 特征相关性热力图
- 支持小时/天时间单位切换

### 🌤️ 天气预测
- 基于当前天气数据的预测
- 24小时和7天预测图表
- 预测结果摘要展示
- 温度、湿度、降水概率等指标

### 🚗 出行建议
- 根据天气状况提供出行建议
- 涵盖驾车、骑行、步行、户外活动
- 风险评估和注意事项
- 天气状况详细分析

## 项目结构

```
frontend-vue/
├── public/                 # 静态资源
├── src/
│   ├── assets/            # 资源文件
│   │   └── styles/        # 样式文件
│   ├── components/        # 公共组件
│   ├── views/             # 页面组件
│   │   ├── Home.vue       # 首页
│   │   ├── DataAnalysis.vue  # 数据分析
│   │   ├── WeatherPrediction.vue  # 天气预测
│   │   └── TravelAdvice.vue  # 出行建议
│   ├── App.vue            # 根组件
│   └── main.js            # 入口文件
├── index.html             # HTML模板
├── package.json           # 项目配置
├── vite.config.js         # Vite配置
└── README.md              # 项目说明
```

## 安装和运行

### 前置要求
- Node.js 16.0 或更高版本
- npm 或 yarn 包管理器

### 安装依赖
```bash
cd frontend-vue
npm install
```

### 开发模式运行
```bash
npm run dev
```
应用将在 `http://localhost:3000` 启动

### 构建生产版本
```bash
npm run build
```
构建文件将生成在 `dist` 目录

### 预览生产版本
```bash
npm run preview
```

## API接口

项目需要后端提供以下API接口：

### 数据分析
- `POST /api/upload` - 上传CSV文件
- 返回：处理后的天气数据和天气代码

### 天气预测
- `POST /api/predict` - 提交预测参数
- 参数：温度、湿度、气压、风速、云量、降水量
- 返回：预测结果和图表数据

### 出行建议
- `POST /api/travel-advice` - 获取出行建议
- 参数：温度、湿度、风速、降水量、能见度、紫外线指数
- 返回：出行建议和风险评估

## 开发指南

### 添加新组件
1. 在 `src/views/` 目录创建新的Vue组件
2. 在 `src/main.js` 中添加路由配置
3. 在 `src/App.vue` 中添加导航链接

### 样式开发
- 使用Bootstrap 5的CSS类
- 自定义样式写在组件的 `<style scoped>` 中
- 全局样式写在 `src/assets/styles/main.css`

### 图表开发
- 使用ECharts库创建图表
- 在组件挂载后初始化图表实例
- 监听数据变化更新图表

## 部署

### 静态部署
1. 运行 `npm run build`
2. 将 `dist` 目录内容部署到Web服务器

### Docker部署
```dockerfile
FROM nginx:alpine
COPY dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 浏览器支持

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## 贡献

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请提交 Issue 或联系开发团队。 