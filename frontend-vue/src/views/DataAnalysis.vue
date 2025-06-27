<template>
  <div class="data-analysis">
    <h1 class="mb-4">气象数据分析</h1>
    
    <!-- 加载动画和错误提示 -->
    <div v-if="loading" class="loading-spinner">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">加载中...</span>
      </div>
    </div>
    
    <div v-if="error" class="alert alert-danger">
      <span>{{ error }}</span>
    </div>

    <!-- 上传区域 -->
    <div v-if="!dataLoaded" class="upload-section">
      <h3>请上传数据文件</h3>
      <p class="text-muted">支持CSV格式文件</p>
      <div class="mt-4">
        <button @click="triggerFileUpload" class="btn btn-primary">选择文件</button>
        <input 
          ref="fileInput" 
          type="file" 
          class="file-upload-btn" 
          accept=".csv"
          @change="handleFileUpload"
          style="display: none;"
        >
      </div>
    </div>

    <!-- 图表区域 -->
    <div v-if="dataLoaded" class="analysis-section">
      <!-- 趋势图 -->
      <div class="row">
        <div class="col-md-6">
          <div class="chart-header">
            <h3>温度变化趋势</h3>
            <div class="btn-group" role="group">
              <button 
                v-for="unit in timeUnits" 
                :key="unit"
                type="button" 
                class="btn btn-outline-primary"
                :class="{ active: temperatureTimeUnit === unit }"
                :disabled="temperatureLoading"
                @click="updateTemperatureChart(unit)"
              >
                <span v-if="temperatureLoading && temperatureTimeUnit === unit" class="spinner-border spinner-border-sm me-1" role="status"></span>
                {{ unit === 'hour' ? '小时' : '天' }}
              </button>
            </div>
          </div>
          <div ref="temperatureChart" class="chart-container"></div>
        </div>
        <div class="col-md-6">
          <div class="chart-header">
            <h3>湿度变化趋势</h3>
            <div class="btn-group" role="group">
              <button 
                v-for="unit in timeUnits" 
                :key="unit"
                type="button" 
                class="btn btn-outline-primary"
                :class="{ active: humidityTimeUnit === unit }"
                :disabled="humidityLoading"
                @click="updateHumidityChart(unit)"
              >
                <span v-if="humidityLoading && humidityTimeUnit === unit" class="spinner-border spinner-border-sm me-1" role="status"></span>
                {{ unit === 'hour' ? '小时' : '天' }}
              </button>
            </div>
          </div>
          <div ref="humidityChart" class="chart-container"></div>
        </div>
      </div>

      <!-- 分布图 -->
      <div class="row">
        <div class="col-md-6">
          <div class="chart-header">
            <h3>降水量变化趋势</h3>
            <div class="btn-group" role="group">
              <button 
                v-for="unit in timeUnits" 
                :key="unit"
                type="button" 
                class="btn btn-outline-primary"
                :class="{ active: precipitationTimeUnit === unit }"
                :disabled="precipitationLoading"
                @click="updatePrecipitationChart(unit)"
              >
                <span v-if="precipitationLoading && precipitationTimeUnit === unit" class="spinner-border spinner-border-sm me-1" role="status"></span>
                {{ unit === 'hour' ? '小时' : '天' }}
              </button>
            </div>
          </div>
          <div ref="precipitationChart" class="chart-container"></div>
        </div>
        <div class="col-md-6">
          <h3>天气类型分布</h3>
          <div ref="weatherChart" class="chart-container"></div>
          <div v-if="weatherCodes && Object.keys(weatherCodes).length > 0" class="weather-codes-list">
            <div class="weather-codes-header" @click="toggleWeatherCodes">
              <h5>
                <i class="bi" :class="showAllWeatherCodes ? 'bi-chevron-up' : 'bi-chevron-down'"></i>
                天气代码说明 ({{ Object.keys(weatherCodes).length }}项)
              </h5>
            </div>
                         <div v-show="showAllWeatherCodes" class="weather-codes-content">
               <div class="row">
                 <div v-for="(desc, code) in limitedWeatherCodes" :key="code" class="col-md-6 mb-1">
                   <small><strong>{{ code }}:</strong> {{ desc }}</small>
                 </div>
               </div>
               <div v-if="weatherCodes && Object.keys(weatherCodes).length > 6" class="mt-2">
                 <button @click="showMoreCodes = !showMoreCodes" class="btn btn-outline-secondary btn-sm">
                   {{ showMoreCodes ? '收起' : `显示更多 (${Object.keys(weatherCodes).length - 6}项)` }}
                 </button>
               </div>
             </div>
          </div>
        </div>
      </div>

      <div class="row">
        <div class="col-md-6">
          <h3>云量分布</h3>
          <div ref="cloudChart" class="chart-container"></div>
        </div>
        <div class="col-md-6">
          <h3>风速分布</h3>
          <div ref="windChart" class="chart-container"></div>
        </div>
      </div>

      <!-- 相关性热力图 -->
      <div class="row">
        <div class="col-md-12">
          <h3>特征相关性热力图</h3>
          <div ref="correlationChart" class="chart-container"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick, computed } from 'vue'
import * as echarts from 'echarts'

export default {
  name: 'DataAnalysis',
  setup() {
    const fileInput = ref(null)
    const loading = ref(false)
    const error = ref('')
    const dataLoaded = ref(false)
    const weatherData = ref(null)
    const weatherCodes = ref(null)
    
    // 添加独立的时间单位切换加载状态
    const temperatureLoading = ref(false)
    const humidityLoading = ref(false)
    const precipitationLoading = ref(false)
    
    const timeUnits = ['hour', 'day']
    const temperatureTimeUnit = ref('hour')
    const humidityTimeUnit = ref('hour')
    const precipitationTimeUnit = ref('hour')
    
    // 图表引用
    const temperatureChart = ref(null)
    const humidityChart = ref(null)
    const precipitationChart = ref(null)
    const weatherChart = ref(null)
    const cloudChart = ref(null)
    const windChart = ref(null)
    const correlationChart = ref(null)
    
    // 图表实例
    let temperatureChartInstance = null
    let humidityChartInstance = null
    let precipitationChartInstance = null
    let weatherChartInstance = null
    let cloudChartInstance = null
    let windChartInstance = null
    let correlationChartInstance = null

    const triggerFileUpload = () => {
      fileInput.value.click()
    }

    const handleFileUpload = async (event) => {
      const file = event.target.files[0]
      if (!file) return

      loading.value = true
      error.value = ''

      try {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '文件上传失败')
        }

        const result = await response.json()
        console.log('文件上传成功:', result)
        
        // 上传成功后，获取分析数据
        await loadAnalysisData()
        
      } catch (err) {
        error.value = err.message || '处理文件时出错'
      } finally {
        loading.value = false
      }
    }

    const loadAnalysisData = async (timeUnit = 'hour') => {
      try {
        const response = await fetch(`/api/analysis-data?time_unit=${timeUnit}`)
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '获取分析数据失败')
        }

        const analysisData = await response.json()
        console.log('分析数据:', analysisData)
        
        // 转换数据格式为前端期望的格式
        weatherData.value = {
          timestamps: analysisData.temperature_trend.times,
          temperature: analysisData.temperature_trend.values,
          humidity: analysisData.humidity_trend.values,
          precipitation_data: analysisData.precipitation_trend ? analysisData.precipitation_trend.values : analysisData.humidity_trend.values.map(() => 0),
          precipitation: analysisData.precipitation_distribution.values,
          precipitation_categories: analysisData.precipitation_distribution.categories,
          weather_distribution: analysisData.weather_distribution,
          cloud_cover_distribution: analysisData.cloud_cover_distribution,
          wind_speed_distribution: analysisData.wind_speed_distribution,
          correlation_heatmap: analysisData.correlation_heatmap
        }
        
        // 获取天气代码
        try {
          const codesResponse = await fetch('/api/weather-codes')
          if (codesResponse.ok) {
            weatherCodes.value = await codesResponse.json()
          }
        } catch (err) {
          console.warn('获取天气代码失败:', err)
          weatherCodes.value = {}
        }
        
        dataLoaded.value = true

        await nextTick()
        initializeCharts()
        
      } catch (err) {
        error.value = err.message || '获取分析数据失败'
        throw err
      }
    }

    const initializeCharts = () => {
      if (!weatherData.value) return

      // 初始化所有图表
      temperatureChartInstance = echarts.init(temperatureChart.value)
      humidityChartInstance = echarts.init(humidityChart.value)
      precipitationChartInstance = echarts.init(precipitationChart.value)
      weatherChartInstance = echarts.init(weatherChart.value)
      cloudChartInstance = echarts.init(cloudChart.value)
      windChartInstance = echarts.init(windChart.value)
      correlationChartInstance = echarts.init(correlationChart.value)

      // 绘制图表
      drawTemperatureChart()
      drawHumidityChart()
      drawPrecipitationChart()
      drawWeatherChart()
      drawCloudChart()
      drawWindChart()
      drawCorrelationChart()
    }

    const drawTemperatureChart = () => {
      if (!temperatureChartInstance || !weatherData.value) return

      const data = weatherData.value
      const option = {
        title: { text: '温度变化趋势' },
        tooltip: { trigger: 'axis' },
        xAxis: { 
          type: 'category', 
          data: data.timestamps 
        },
        yAxis: { type: 'value' },
        series: [{
          data: data.temperature,
          type: 'line',
          smooth: true,
          name: '温度 (°C)'
        }]
      }
      temperatureChartInstance.setOption(option)
    }

    const drawHumidityChart = () => {
      if (!humidityChartInstance || !weatherData.value) return

      const data = weatherData.value
      const option = {
        title: { text: '湿度变化趋势' },
        tooltip: { trigger: 'axis' },
        xAxis: { 
          type: 'category', 
          data: data.timestamps 
        },
        yAxis: { type: 'value' },
        series: [{
          data: data.humidity,
          type: 'line',
          smooth: true,
          name: '湿度 (%)'
        }]
      }
      humidityChartInstance.setOption(option)
    }

    const drawPrecipitationChart = () => {
      if (!precipitationChartInstance || !weatherData.value) return

      const data = weatherData.value
      
      const option = {
        title: { text: '降水量变化趋势' },
        tooltip: { 
          trigger: 'axis',
          formatter: function (params) {
            let result = params[0].name + '<br/>'
            params.forEach(param => {
              result += param.marker + param.seriesName + ': ' + param.value + ' mm<br/>'
            })
            return result
          }
        },
        legend: {
          data: ['降水量']
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: data.timestamps || []
        },
        yAxis: {
          type: 'value',
          name: '降水量 (mm)',
          min: 0
        },
        series: [{
          name: '降水量',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: {
            color: '#1890ff',
            width: 2
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [{
                offset: 0, color: 'rgba(24, 144, 255, 0.3)'
              }, {
                offset: 1, color: 'rgba(24, 144, 255, 0.05)'
              }]
            }
          },
          data: data.precipitation_data || []
        }]
      }
      precipitationChartInstance.setOption(option)
    }

    const drawWeatherChart = () => {
      if (!weatherChartInstance || !weatherData.value) return

      const data = weatherData.value
      let chartData = []
      
      if (Array.isArray(data.weather_distribution)) {
        // 如果是数组格式
        chartData = data.weather_distribution.map(item => ({
          value: item.count,
          name: item.description || weatherCodes.value[item.code] || `代码${item.code}`
        }))
      } else if (typeof data.weather_distribution === 'object') {
        // 如果是对象格式
        chartData = Object.entries(data.weather_distribution).map(([type, count]) => ({
          value: count,
          name: weatherCodes.value[type] || type
        }))
      }

      const option = {
        title: { text: '天气类型分布' },
        tooltip: { trigger: 'item' },
        series: [{
          type: 'pie',
          radius: '50%',
          data: chartData
        }]
      }
      weatherChartInstance.setOption(option)
    }

    const drawCloudChart = () => {
      if (!cloudChartInstance || !weatherData.value) return

      const data = weatherData.value
      let categories = []
      let values = []
      
      if (data.cloud_cover_distribution && typeof data.cloud_cover_distribution === 'object') {
        if (data.cloud_cover_distribution.categories && data.cloud_cover_distribution.values) {
          // 如果有分类和数值
          categories = data.cloud_cover_distribution.categories.map(val => `${val.toFixed(0)}%`)
          values = data.cloud_cover_distribution.values
        } else if (Array.isArray(data.cloud_cover_distribution)) {
          // 如果是数组
          categories = ['0-25%', '26-50%', '51-75%', '76-100%']
          values = data.cloud_cover_distribution
        }
      }
      
      const option = {
        title: { text: '云量分布' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: categories },
        yAxis: { type: 'value' },
        series: [{
          data: values,
          type: 'bar',
          name: '云量'
        }]
      }
      cloudChartInstance.setOption(option)
    }

    const drawWindChart = () => {
      if (!windChartInstance || !weatherData.value) return

      const data = weatherData.value
      let categories = []
      let values = []
      
      if (data.wind_speed_distribution && typeof data.wind_speed_distribution === 'object') {
        if (data.wind_speed_distribution.categories && data.wind_speed_distribution.values) {
          // 如果有分类和数值
          categories = data.wind_speed_distribution.categories.map(val => `${val.toFixed(1)}`)
          values = data.wind_speed_distribution.values
        } else if (Array.isArray(data.wind_speed_distribution)) {
          // 如果是数组
          categories = ['0-5', '6-10', '11-15', '16-20', '>20']
          values = data.wind_speed_distribution
        }
      }

      const option = {
        title: { text: '风速分布' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: categories },
        yAxis: { type: 'value' },
        series: [{
          data: values,
          type: 'bar',
          name: '风速 (m/s)'
        }]
      }
      windChartInstance.setOption(option)
    }

    const drawCorrelationChart = () => {
      if (!correlationChartInstance || !weatherData.value) return

      const data = weatherData.value
      let features = []
      let correlationData = []
      
      if (data.correlation_heatmap) {
        features = data.correlation_heatmap.features || []
        const correlations = data.correlation_heatmap.correlations || []
        
        // 转换相关性矩阵为ECharts热力图数据格式
        correlationData = []
        for (let i = 0; i < features.length; i++) {
          for (let j = 0; j < features.length; j++) {
            if (correlations[i] && correlations[i][j] !== undefined) {
              correlationData.push([j, i, correlations[i][j]])
            }
          }
        }
      }

      const option = {
        title: { text: '特征相关性热力图' },
        tooltip: { 
          position: 'top',
          formatter: function (params) {
            return `${features[params.data[1]]} vs ${features[params.data[0]]}<br/>相关性: ${params.data[2].toFixed(3)}`
          }
        },
        grid: { height: '50%', top: '10%' },
        xAxis: {
          type: 'category',
          data: features,
          splitArea: { show: true }
        },
        yAxis: {
          type: 'category',
          data: features,
          splitArea: { show: true }
        },
        visualMap: {
          min: -1,
          max: 1,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '15%',
          inRange: {
            color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
          }
        },
        series: [{
          name: '相关性',
          type: 'heatmap',
          data: correlationData,
          label: { 
            show: true, 
            formatter: function(params) {
              return params.value[2].toFixed(2)
            }
          },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
        }]
      }
      correlationChartInstance.setOption(option)
    }

    const updateTemperatureChart = async (unit) => {
      if (temperatureTimeUnit.value === unit) return
      
      try {
        temperatureLoading.value = true
        temperatureTimeUnit.value = unit
        
        // 重新获取指定时间单位的数据
        const response = await fetch(`/api/analysis-data?time_unit=${unit}`)
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '获取分析数据失败')
        }

        const analysisData = await response.json()
        
        // 更新温度数据
        weatherData.value.timestamps = analysisData.temperature_trend.times
        weatherData.value.temperature = analysisData.temperature_trend.values
        
        // 重新绘制温度图表
        drawTemperatureChart()
        
      } catch (err) {
        console.error('更新温度图表失败:', err)
        error.value = err.message || '更新温度图表失败'
      } finally {
        temperatureLoading.value = false
      }
    }

    const updateHumidityChart = async (unit) => {
      if (humidityTimeUnit.value === unit) return
      
      try {
        humidityLoading.value = true
        humidityTimeUnit.value = unit
        
        // 重新获取指定时间单位的数据
        const response = await fetch(`/api/analysis-data?time_unit=${unit}`)
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '获取分析数据失败')
        }

        const analysisData = await response.json()
        
        // 更新湿度数据
        weatherData.value.timestamps = analysisData.temperature_trend.times
        weatherData.value.humidity = analysisData.humidity_trend.values
        
        // 重新绘制湿度图表
        drawHumidityChart()
        
      } catch (err) {
        console.error('更新湿度图表失败:', err)
        error.value = err.message || '更新湿度图表失败'
      } finally {
        humidityLoading.value = false
      }
    }

    const updatePrecipitationChart = async (unit) => {
      if (precipitationTimeUnit.value === unit) return
      
      try {
        precipitationLoading.value = true
        precipitationTimeUnit.value = unit
        
        // 重新获取指定时间单位的数据
        const response = await fetch(`/api/analysis-data?time_unit=${unit}`)
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || '获取分析数据失败')
        }

        const analysisData = await response.json()
        
        // 更新降水量数据
        weatherData.value.timestamps = analysisData.precipitation_trend.times
        weatherData.value.precipitation_data = analysisData.precipitation_trend.values
        
        // 重新绘制降水量图表
        drawPrecipitationChart()
        
      } catch (err) {
        console.error('更新降水量图表失败:', err)
        error.value = err.message || '更新降水量图表失败'
      } finally {
        precipitationLoading.value = false
      }
    }

    const showAllWeatherCodes = ref(false)
    const showMoreCodes = ref(false)
    
    const toggleWeatherCodes = () => {
      showAllWeatherCodes.value = !showAllWeatherCodes.value
    }
    
    // 计算显示的天气代码
    const limitedWeatherCodes = computed(() => {
      if (!weatherCodes.value) return {}
      
      const entries = Object.entries(weatherCodes.value)
      if (!showAllWeatherCodes.value) return {}
      
      if (showMoreCodes.value || entries.length <= 6) {
        return weatherCodes.value
      } else {
        const limited = {}
        entries.slice(0, 6).forEach(([code, desc]) => {
          limited[code] = desc
        })
        return limited
      }
    })

    onMounted(async () => {
      // 组件挂载后尝试加载已有数据
      try {
        await loadAnalysisData()
      } catch (err) {
        // 如果没有数据，显示上传界面
        console.log('没有现有数据，请上传文件')
      }
    })

    return {
      fileInput,
      loading,
      error,
      dataLoaded,
      weatherCodes,
      timeUnits,
      temperatureTimeUnit,
      humidityTimeUnit,
      precipitationTimeUnit,
      temperatureLoading,
      humidityLoading,
      precipitationLoading,
      temperatureChart,
      humidityChart,
      precipitationChart,
      weatherChart,
      cloudChart,
      windChart,
      correlationChart,
      triggerFileUpload,
      handleFileUpload,
      loadAnalysisData,
      updateTemperatureChart,
      updateHumidityChart,
      updatePrecipitationChart,
      showAllWeatherCodes,
      showMoreCodes,
      limitedWeatherCodes,
      toggleWeatherCodes
    }
  }
}
</script>

<style scoped>
.data-analysis {
  padding: 20px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.upload-section {
  text-align: center;
  padding: 3rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.weather-codes-list {
  margin-top: 1rem;
  padding: 0.8rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.weather-codes-header {
  cursor: pointer;
  user-select: none;
  margin-bottom: 0.5rem;
}

.weather-codes-header:hover {
  background: #e9ecef;
  border-radius: 4px;
  padding: 0.2rem 0.5rem;
}

.weather-codes-header h5 {
  margin: 0;
  color: #495057;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.weather-codes-content {
  animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.analysis-section {
  margin-top: 2rem;
}

.row {
  margin-bottom: 2rem;
}

.chart-container {
  height: 400px;
  width: 100%;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .chart-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }
  
  .btn-group {
    width: 100%;
  }
}
</style> 