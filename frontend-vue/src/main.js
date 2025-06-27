import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Home from './views/Home.vue'
import DataAnalysis from './views/DataAnalysis.vue'
import WeatherPrediction from './views/WeatherPrediction.vue'
import TravelAdvice from './views/TravelAdvice.vue'
import './assets/styles/main.css'

// 路由配置
const routes = [
  { path: '/', component: Home },
  { path: '/data-analysis', component: DataAnalysis },
  { path: '/weather-prediction', component: WeatherPrediction },
  { path: '/travel-advice', component: TravelAdvice }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(router)
app.mount('#app') 