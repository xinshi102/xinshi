<template>
  <div id="app">
    <!-- 侧边导航栏 -->
    <div class="sidebar">
      <div class="sidebar-header">
        <h3>气象数据可视化及分析系统</h3>
      </div>
      <ul class="sidebar-menu">
        <li>
          <router-link to="/" class="nav-link">
            <i class="bi bi-house"></i>首页
          </router-link>
        </li>
        <li>
          <router-link to="/data-analysis" class="nav-link">
            <i class="bi bi-graph-up"></i>数据分析
          </router-link>
        </li>
        <li>
          <router-link to="/weather-prediction" class="nav-link">
            <i class="bi bi-cloud-sun"></i>天气预测
          </router-link>
        </li>
        <li>
          <router-link to="/travel-advice" class="nav-link">
            <i class="bi bi-umbrella"></i>出行建议
          </router-link>
        </li>
      </ul>
    </div>

    <!-- 主内容区域 -->
    <div class="main-content">
      <router-view />
    </div>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style scoped>
/* 导航栏样式 */
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  width: 250px;
  height: 100vh;
  background-color: #2c3e50;
  padding: 20px 0;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
  z-index: 1000;
}

.sidebar-header {
  text-align: center;
  padding: 20px;
  color: white;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar-menu {
  list-style: none;
  padding: 0;
  margin: 20px 0;
}

.sidebar-menu li {
  padding: 10px 20px;
  transition: all 0.3s;
}

.sidebar-menu li:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

.nav-link {
  color: white;
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 10px;
}

.nav-link i {
  width: 20px;
  text-align: center;
}

.nav-link.router-link-active {
  color: #3498db;
  font-weight: bold;
}

/* 主内容区域样式 */
.main-content {
  margin-left: 250px;
  padding: 20px;
  min-height: 100vh;
  background-color: #f8f9fa;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
  }
  
  .main-content {
    margin-left: 0;
  }
}
</style> 