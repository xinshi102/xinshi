<template>
  <div class="weather-prediction">
    <h1 class="mb-4">天气预测</h1>
    
    <!-- 加载动画和错误提示 -->
    <div v-if="loading" class="loading-spinner text-center">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">预测中...</span>
      </div>
      <p class="mt-2">正在分析天气数据并生成预测...</p>
    </div>
    
    <div v-if="error" class="alert alert-danger">
      <i class="bi bi-exclamation-triangle"></i>
      <span>{{ error }}</span>
    </div>

    <!-- 预测操作区域 -->
    <div class="prediction-input-section">
      <div class="text-center">
        <p class="mb-4">基于已上传的气象数据，生成未来72小时天气预测</p>
        <button @click="makePrediction" class="btn btn-primary btn-lg" :disabled="loading">
          <i class="bi bi-cloud-sun"></i> 开始预测
        </button>
      </div>
    </div>

    <!-- 预测结果区域 -->
    <div v-if="predictionResult" class="prediction-result-section">
      <h2 class="mb-4">预测结果</h2>
      
      <div class="row">
        <div class="col-12">
          <div class="prediction-card">
            <h3>未来72小时天气预测</h3>
            <div class="prediction-chart">
              <div ref="hourlyChart" class="chart-container"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="row mt-4">
        <div class="col-12">
          <div class="prediction-card">
            <h3>3天温度趋势</h3>
            <div class="prediction-chart">
              <div ref="dailyChart" class="chart-container"></div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 简化的预测提示 -->
      <div class="mt-4">
        <div class="alert alert-info">
          <i class="bi bi-info-circle"></i>
          预测完成！现在可以前往 <strong>出行建议</strong> 页面获取基于此预测的智能出行建议。
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'

export default {
  name: 'WeatherPrediction',
  setup() {
    const loading = ref(false)
    const error = ref('')
    const predictionResult = ref(null)
    
    // 预测输入，初始化为空
    const predictionInput = ref({
      temperature: '',
      humidity: '',
      pressure: '',
      windSpeed: '',
      cloudCover: '',
      precipitation: ''
    })
    
    // 图表引用
    const hourlyChart = ref(null)
    const dailyChart = ref(null)
    
    // 图表实例
    let hourlyChartInstance = null
    let dailyChartInstance = null

    // 自动获取CSV后三天数据
    const fetchLast3DaysData = async () => {
      try {
        const response = await fetch('/api/analysis-data')
        if (!response.ok) throw new Error('无法获取分析数据')
        const data = await response.json()
        // 假设数据为按小时排列的数组，取最后72个数据点
        const len = data.temperature_trend.values.length
        const last72 = {
          temperature: data.temperature_trend.values.slice(len - 72),
          humidity: data.humidity_trend.values.slice(len - 72),
          pressure: data.hasOwnProperty('pressure_trend') ? data.pressure_trend.values.slice(len - 72) : Array(72).fill(1000),
          windSpeed: data.wind_speed_distribution.values.slice(-72),
          cloudCover: data.cloud_cover_distribution.values.slice(-72),
          precipitation: data.precipitation_distribution.values.slice(-72)
        }
        // 取后三天的均值作为预测输入
        predictionInput.value.temperature = average(last72.temperature)
        predictionInput.value.humidity = average(last72.humidity)
        predictionInput.value.pressure = average(last72.pressure)
        predictionInput.value.windSpeed = average(last72.windSpeed)
        predictionInput.value.cloudCover = average(last72.cloudCover)
        predictionInput.value.precipitation = average(last72.precipitation)
      } catch (err) {
        error.value = '自动获取数据失败，请先上传并分析数据'
      }
    }

    // 计算均值
    function average(arr) {
      if (!arr || arr.length === 0) return ''
      const sum = arr.reduce((a, b) => Number(a) + Number(b), 0)
      return (sum / arr.length).toFixed(2)
    }

    // 页面加载时自动获取数据
    onMounted(() => {
      fetchLast3DaysData()
    })

    const makePrediction = async () => {
      loading.value = true
      error.value = ''
      try {
        const response = await fetch('/api/predict?duration=72', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        })
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '预测请求失败')
        }
        const result = await response.json()
        
        // 直接存储后端返回的原始数据，供出行建议使用
        localStorage.setItem('weatherPredictionResult', JSON.stringify(result))
        
        // 为前端图表转换数据格式
        predictionResult.value = {
          hourly: {
            hours: result.times || [],
            temperature: result.temperatures || [],
            humidity: result.humidities || [],
            precipitationProbability: result.precipitations || []
          },
          daily: {
            days: result.times ? result.times.filter((_, index) => index % 24 === 0).slice(0, 7) : [],
            maxTemperature: [],
            minTemperature: [],
            avgHumidity: []
          },
          // 添加原始数据引用，方便图表使用
          rawData: result
        }
        
        // 计算每日统计数据
        if (result.temperatures && result.humidities) {
          const temps = result.temperatures
          const humids = result.humidities
          
          for (let day = 0; day < 3; day++) { // 3天数据
            const dayStart = day * 24
            const dayEnd = Math.min((day + 1) * 24, temps.length)
            
            if (dayStart < temps.length) {
              const dayTemps = temps.slice(dayStart, dayEnd)
              const dayHumids = humids.slice(dayStart, dayEnd)
              
              predictionResult.value.daily.maxTemperature.push(Math.max(...dayTemps))
              predictionResult.value.daily.minTemperature.push(Math.min(...dayTemps))
              predictionResult.value.daily.avgHumidity.push(dayHumids.reduce((a, b) => a + b, 0) / dayHumids.length)
            }
          }
        }
        await nextTick()
        initializeCharts()
      } catch (err) {
        error.value = err.message || '预测过程中出错'
      } finally {
        loading.value = false
      }
    }

    const initializeCharts = () => {
      if (!predictionResult.value) return
      hourlyChartInstance = echarts.init(hourlyChart.value)
      dailyChartInstance = echarts.init(dailyChart.value)
      drawHourlyChart()
      drawDailyChart()
    }
    const drawHourlyChart = () => {
      if (!hourlyChartInstance || !predictionResult.value) return
      const data = predictionResult.value.hourly
      const option = {
        title: { text: '72小时天气预测' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['温度', '湿度', '降水量'] },
        xAxis: { 
          type: 'category', 
          data: data.hours,
          axisLabel: {
            rotate: 45,
            interval: 'auto'
          }
        },
        yAxis: [
          { type: 'value', name: '温度 (°C)' },
          { type: 'value', name: '湿度 (%) / 降水量 (mm)', max: 100 }
        ],
        series: [
          { 
            name: '温度', 
            type: 'line', 
            data: data.temperature, 
            smooth: true, 
            yAxisIndex: 0,
            itemStyle: { color: '#ff6b6b' }
          },
          { 
            name: '湿度', 
            type: 'line', 
            data: data.humidity, 
            smooth: true, 
            yAxisIndex: 1,
            itemStyle: { color: '#4ecdc4' }
          },
          { 
            name: '降水量', 
            type: 'bar', 
            data: data.precipitationProbability, 
            yAxisIndex: 1,
            itemStyle: { color: '#45b7d1' }
          }
        ]
      }
      hourlyChartInstance.setOption(option)
    }
    const drawDailyChart = () => {
      if (!dailyChartInstance || !predictionResult.value) return
      const data = predictionResult.value.daily
      const option = {
        title: { text: '3天温度趋势' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['最高温度', '最低温度', '平均湿度'] },
        xAxis: { type: 'category', data: data.days },
        yAxis: [
          { type: 'value', name: '温度 (°C)' },
          { type: 'value', name: '湿度 (%)', max: 100 }
        ],
        series: [
          { 
            name: '最高温度', 
            type: 'line', 
            data: data.maxTemperature, 
            smooth: true,
            itemStyle: { color: '#ff6b6b' }
          },
          { 
            name: '最低温度', 
            type: 'line', 
            data: data.minTemperature, 
            smooth: true,
            itemStyle: { color: '#4ecdc4' }
          },
          { 
            name: '平均湿度', 
            type: 'line', 
            data: data.avgHumidity, 
            smooth: true, 
            yAxisIndex: 1,
            itemStyle: { color: '#95e1d3' }
          }
        ]
      }
      dailyChartInstance.setOption(option)
    }
    onMounted(() => {})
    return {
      loading,
      error,
      predictionInput,
      predictionResult,
      hourlyChart,
      dailyChart,
      makePrediction
    }
  }
}
</script>

<style scoped>
.weather-prediction {
  padding: 20px;
}

.prediction-input-section {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.5rem;
  display: block;
}

.form-control {
  border-radius: 8px;
  border: 2px solid #e9ecef;
  padding: 0.75rem;
  transition: all 0.3s ease;
}

.form-control:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
}

.prediction-result-section {
  margin-top: 2rem;
}

.prediction-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 1rem;
}

.prediction-card h3 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-weight: 600;
}

.prediction-summary {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.prediction-summary h3 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-weight: 600;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.summary-item:hover {
  background: #e9ecef;
  transform: translateY(-2px);
}

.summary-item i {
  font-size: 2rem;
  color: #3498db;
}

.summary-item h4 {
  margin: 0;
  color: #2c3e50;
  font-weight: 600;
}

.summary-item p {
  margin: 0;
  color: #666;
  font-size: 1.1rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .summary-grid {
    grid-template-columns: 1fr;
  }
  
  .prediction-input-section {
    padding: 1rem;
  }
  
  .prediction-card {
    padding: 1rem;
  }
}
</style> 