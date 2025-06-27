<template>
  <div class="travel-advice">
    <h1 class="mb-4">出行建议</h1>
    
    <!-- 加载动画和错误提示 -->
    <div v-if="loading" class="loading-spinner text-center">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">AI建议生成中...</span>
      </div>
      <p class="mt-2">正在分析天气预测数据，生成智能出行建议...</p>
    </div>
    
    <div v-if="error" class="alert alert-danger">
      <i class="bi bi-exclamation-triangle"></i>
      <span>{{ error }}</span>
    </div>

    <!-- 预测数据摘要 -->
    <div v-if="!loading && !error && predictionSummary" class="prediction-summary-section">
      <h3>基于以下天气预测数据：</h3>
      <div class="prediction-summary-card">
        <div class="row">
          <div class="col-md-3">
            <div class="summary-item">
              <i class="bi bi-thermometer-half text-danger"></i>
              <div>
                <strong>温度范围</strong>
                <p>{{ predictionSummary.tempRange }}</p>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="summary-item">
              <i class="bi bi-droplet text-primary"></i>
              <div>
                <strong>降水情况</strong>
                <p>{{ predictionSummary.precipitation }}</p>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="summary-item">
              <i class="bi bi-moisture text-info"></i>
              <div>
                <strong>湿度范围</strong>
                <p>{{ predictionSummary.humidityRange }}</p>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="summary-item">
              <i class="bi bi-cloud text-secondary"></i>
              <div>
                <strong>主要天气</strong>
                <p>{{ predictionSummary.mainWeather }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 出行建议结果 -->
    <div v-if="travelAdvice && !loading && !error" class="advice-result-section">
      <h2 class="mb-4">
        <i class="bi bi-robot"></i> AI智能出行建议
      </h2>
      <div class="ai-advice-content">
        <div v-html="formatAdviceContent(travelAdvice)"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  name: 'TravelAdvice',
  setup() {
    const loading = ref(false)
    const error = ref('')
    const travelAdvice = ref('')
    const weatherInput = ref({})
    const predictionSummary = ref(null)

    // 页面加载时自动读取天气预测结果并请求AI建议
    onMounted(async () => {
      try {
        const predictionStr = localStorage.getItem('weatherPredictionResult')
        if (!predictionStr) {
          error.value = '请先完成天气预测';
          return
        }
        const prediction = JSON.parse(predictionStr)
        
        // 生成预测数据摘要用于显示
        generatePredictionSummary(prediction)
        
        loading.value = true
        // 调用后端AI建议接口
        const response = await fetch('/api/deepseek-advice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(prediction)
        })
        const result = await response.json()
        if (result.advice) {
          travelAdvice.value = result.advice
        } else {
          error.value = result.error || 'AI建议生成失败'
        }
      } catch (err) {
        error.value = 'AI建议生成失败'
        console.error('出行建议生成错误:', err)
      } finally {
        loading.value = false
      }
    })

    // 生成预测数据摘要
    const generatePredictionSummary = (prediction) => {
      try {
        if (!prediction || !prediction.temperatures || !prediction.humidities) {
          return
        }

        const temps = prediction.temperatures
        const humids = prediction.humidities
        const precips = prediction.precipitations || []
        const weatherDescs = prediction.weather_descriptions || []

        const tempMin = Math.min(...temps)
        const tempMax = Math.max(...temps)
        const humidMin = Math.min(...humids)
        const humidMax = Math.max(...humids)
        const totalPrecip = precips.reduce((sum, p) => sum + p, 0)
        const mainWeather = weatherDescs.length > 0 ? 
          weatherDescs.reduce((a, b, i, arr) => 
            arr.filter(v => v === a).length >= arr.filter(v => v === b).length ? a : b
          ) : '未知'

        predictionSummary.value = {
          tempRange: `${tempMin.toFixed(1)}°C ~ ${tempMax.toFixed(1)}°C`,
          humidityRange: `${humidMin.toFixed(0)}% ~ ${humidMax.toFixed(0)}%`,
          precipitation: totalPrecip > 0 ? `总计 ${totalPrecip.toFixed(1)}mm` : '无降水',
          mainWeather: mainWeather
        }
      } catch (err) {
        console.error('生成预测摘要失败:', err)
      }
    }

    const formatAdviceContent = (advice) => {
      // 简单的换行处理，保持原有格式
      return advice.replace(/\n/g, '<br>')
    }

    return {
      loading,
      error,
      weatherInput,
      travelAdvice,
      predictionSummary,
      formatAdviceContent
    }
  }
}
</script>

<style scoped>
.travel-advice {
  padding: 20px;
}

.weather-input-section {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.weather-input-section h3 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-weight: 600;
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

.advice-result-section {
  margin-top: 2rem;
}

.overall-advice {
  margin-bottom: 2rem;
}

.travel-advice {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  text-align: center;
}

.travel-advice .icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.travel-advice .advice-text {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  font-weight: 600;
}

.travel-advice .details {
  color: #666;
  font-size: 1.1rem;
}

.travel-advice.good {
  border-left: 5px solid #4CAF50;
}

.travel-advice.warning {
  border-left: 5px solid #FFC107;
}

.travel-advice.danger {
  border-left: 5px solid #F44336;
}

.detailed-advice {
  margin-bottom: 2rem;
}

.advice-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 1rem;
  height: 100%;
}

.advice-card h3 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.advice-content p {
  margin-bottom: 0.5rem;
  line-height: 1.6;
}

.risk-level {
  margin-top: 1rem;
}

.badge {
  font-size: 0.9rem;
  padding: 0.5rem 1rem;
}

.weather-analysis {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.weather-analysis h3 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-weight: 600;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.analysis-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.analysis-item:hover {
  background: #e9ecef;
  transform: translateY(-2px);
}

.analysis-item i {
  font-size: 2rem;
  color: #3498db;
}

.analysis-item h4 {
  margin: 0;
  color: #2c3e50;
  font-weight: 600;
}

.analysis-item p {
  margin: 0;
  color: #666;
  font-size: 1rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .weather-input-section {
    padding: 1rem;
  }
  
  .advice-card {
    padding: 1rem;
  }
  
  .analysis-grid {
    grid-template-columns: 1fr;
  }
  
  .travel-advice .advice-text {
    font-size: 1.2rem;
  }
}

.loading-spinner {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}

.ai-advice-content {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  font-size: 1.1rem;
  line-height: 1.8;
  color: #333;
}

.prediction-summary-section {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.prediction-summary-section h3 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-weight: 600;
}

.prediction-summary-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.summary-item i {
  font-size: 2rem;
  color: #3498db;
}

.summary-item div {
  flex: 1;
}

.summary-item div strong {
  font-weight: 600;
  color: #2c3e50;
}

.summary-item div p {
  margin: 0;
  color: #666;
  font-size: 1rem;
}
</style> 