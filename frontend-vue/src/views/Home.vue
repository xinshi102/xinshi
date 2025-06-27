<template>
  <div class="home">
    <div class="welcome-section">
      <h1>欢迎使用气象数据可视化及分析系统</h1>
      <div class="feature-list">
        <div 
          v-for="(feature, index) in features" 
          :key="feature.id"
          class="feature-card"
          @click="navigateTo(feature.route)"
          :style="{ animationDelay: `${index * 0.1}s` }"
        >
          <div class="feature-icon">
            <i :class="feature.icon"></i>
          </div>
          <h3>{{ feature.title }}</h3>
          <p>{{ feature.description }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

export default {
  name: 'Home',
  setup() {
    const router = useRouter()
    
    const features = ref([
      {
        id: 1,
        title: '天气数据分析',
        description: '查看历史天气数据统计和趋势，深入了解天气变化规律',
        icon: 'bi bi-graph-up',
        route: '/data-analysis'
      },
      {
        id: 2,
        title: '天气预测',
        description: '基于历史数据预测未来天气状况，提前做好规划',
        icon: 'bi bi-cloud-sun',
        route: '/weather-prediction'
      },
      {
        id: 3,
        title: '出行建议',
        description: '根据天气状况提供出行建议，让您的出行更加安全舒适',
        icon: 'bi bi-umbrella',
        route: '/travel-advice'
      }
    ])

    const navigateTo = (route) => {
      router.push(route)
    }

    onMounted(() => {
      // 页面加载动画
      const welcomeSection = document.querySelector('.welcome-section')
      if (welcomeSection) {
        welcomeSection.style.opacity = '0'
        welcomeSection.style.transform = 'translateY(20px)'
        
        setTimeout(() => {
          welcomeSection.style.transition = 'all 0.5s ease'
          welcomeSection.style.opacity = '1'
          welcomeSection.style.transform = 'translateY(0)'
        }, 100)
      }
    })

    return {
      features,
      navigateTo
    }
  }
}
</script>

<style scoped>
.home {
  padding: 20px;
}

.welcome-section {
  text-align: center;
  margin-bottom: 3rem;
}

.welcome-section h1 {
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 2rem;
  font-weight: 600;
}

.feature-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-top: 3rem;
}

.feature-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  animation: fadeInUp 0.6s ease forwards;
  opacity: 0;
  transform: translateY(20px);
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #3498db;
}

.feature-card h3 {
  font-size: 1.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
  font-weight: 600;
}

.feature-card p {
  color: #666;
  line-height: 1.6;
  font-size: 1rem;
}

@keyframes fadeInUp {
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .welcome-section h1 {
    font-size: 2rem;
  }
  
  .feature-list {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .feature-card {
    margin-bottom: 1rem;
  }
}
</style> 