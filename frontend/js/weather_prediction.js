// 图表管理类
class ChartManager {
    constructor() {
        this.charts = {};
    }

    // 安全初始化echarts实例
    safeInitEcharts(domId) {
        const dom = document.getElementById(domId);
        if (!dom) return null;
        if (echarts.getInstanceByDom(dom)) {
            echarts.dispose(dom);
        }
        return echarts.init(dom);
    }

    // 渲染所有预测图表
    renderPredictionCharts(data) {
        this.renderWeatherCodeChart(data);
        this.renderTemperatureChart(data);
        this.renderHumidityChart(data);
        this.renderPrecipitationChart(data);
    }

    // 渲染天气代码图表
    renderWeatherCodeChart(data) {
        const { times, weather_codes, weather_descriptions } = this.prepareChartData(data);
        
        // 处理无效天气代码，确保全部为数字
        const numericCodes = weather_codes.map(x => {
            const num = Number(x);
            return isNaN(num) ? 0 : num; // 转换为数字，无效值变为0
        });
        
        // 如果没有数据，清空图表并返回
        if (!times.length || !weather_codes.length) {
            if (this.charts.weatherCode) {
                this.charts.weatherCode.clear();
            }
            return;
        }

        // 获取天气代码的唯一值列表
        const uniqueCodes = Array.from(new Set(numericCodes.filter(x => x !== null && !isNaN(x)))).sort((a, b) => a - b);
        
        // 构建天气代码与描述的映射
        const codeDescMap = {};
        (data.unique_weather_codes || []).forEach(item => {
            codeDescMap[Number(item.code)] = item.desc;
        });

        this.charts.weatherCode = this.safeInitEcharts('weatherCodeChart');
        this.charts.weatherCode.setOption({
            title: { text: '天气类型预测' },
            tooltip: {
                trigger: 'axis',
                formatter: (params) => {
                    const idx = params[0].dataIndex;
                    return `${params[0].name}<br/>天气代码: ${weather_codes[idx]}<br/>天气描述: ${weather_descriptions[idx]}`;
                }
            },
            xAxis: { type: 'category', data: times },
            yAxis: { 
                name: '天气代码', 
                type: 'value',
                min: Math.max(0, Math.min(...uniqueCodes) - 1),
                max: Math.max(...uniqueCodes) + 1,//设置y轴的最小值和最大值
                interval: 1,
                axisLabel: {
                    formatter: function (value) {
                        return uniqueCodes.includes(value) ? `${value}: ${codeDescMap[value] || ''}` : '';//如果value在里面，就显示天气代码和买哦书，没有就空
                    }
                }
            },
            series: [{
                name: '天气类型',
                type: 'line',
                data: numericCodes,
                symbol: 'circle',//数据点形状
                connectNulls: true//连接空值
            }]
        });
    }

    // 渲染温度图表
    renderTemperatureChart(data) {
        const { times, temperatures } = this.prepareChartData(data);
        this.charts.temperature = this.safeInitEcharts('temperatureChart');
        this.charts.temperature.setOption({
            title: { text: '温度预测' },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: times },
            yAxis: { name: '温度 (°C)' },
            series: [{ name: '温度', type: 'line', data: temperatures }]
        });
    }

    // 渲染湿度图表
    renderHumidityChart(data) {
        const { times, humidities } = this.prepareChartData(data);
        this.charts.humidity = this.safeInitEcharts('humidityChart');
        this.charts.humidity.setOption({
            title: { text: '湿度预测' },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: times },
            yAxis: { name: '湿度 (%)' },
            series: [{ name: '湿度', type: 'line', data: humidities }]
        });
    }

    // 渲染降水量图表
    renderPrecipitationChart(data) {
        const { times, precipitations } = this.prepareChartData(data);
        this.charts.precipitation = this.safeInitEcharts('precipitationChart');
        this.charts.precipitation.setOption({
            title: { text: '降水量预测' },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: times },
            yAxis: { name: '降水量 (mm)' },
            series: [{ name: '降水量', type: 'line', data: precipitations }]
        });
    }

    // 准备图表数据
    prepareChartData(data) {
        // 取所有数组的最小长度
        const arrs = [
            Array.isArray(data.times) ? data.times : [],
            Array.isArray(data.weather_codes) ? data.weather_codes : [],
            Array.isArray(data.weather_descriptions) ? data.weather_descriptions : [],
            Array.isArray(data.temperatures) ? data.temperatures : [],
            Array.isArray(data.humidities) ? data.humidities : [],
            Array.isArray(data.precipitations) ? data.precipitations : []
        ];
        const minLen = Math.min(...arrs.map(arr => arr.length));

        // 截断并处理空值，全部用null
        function safe(arr) {
            return arr.slice(0, minLen).map(x => (x === null || x === undefined || x === '' ? null : x));
        }

        return {
            times: safe(arrs[0]),
            weather_codes: safe(arrs[1]),
            weather_descriptions: safe(arrs[2]),
            temperatures: safe(arrs[3]),
            humidities: safe(arrs[4]),
            precipitations: safe(arrs[5])
        };
    }

    // 调整所有图表大小
    resizeAllCharts() {
        Object.values(this.charts).forEach(chart => chart && chart.resize());
    }
}

// UI管理类
class UIManager {
    constructor() {
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        this.predictionSection = document.getElementById('predictionSection');
        this.weatherCodeList = document.getElementById('weatherCodeList');
    }

    // 显示加载动画
    showLoading() {
        this.loadingSpinner.style.display = 'block';
    }

    // 隐藏加载动画
    hideLoading() {
        this.loadingSpinner.style.display = 'none';
    }

    // 显示错误信息
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.style.display = 'block';
        setTimeout(() => {
            this.errorAlert.style.display = 'none';
        }, 5000);
    }

    // 更新天气代码图例
    updateWeatherCodeLegend(uniqueWeatherCodes) {
        this.weatherCodeList.innerHTML = '';
        (uniqueWeatherCodes || []).forEach(item => {
            const div = document.createElement('div');
            div.className = 'weather-code-item';
            div.textContent = `${item.code}: ${item.desc}`;
            this.weatherCodeList.appendChild(div);
        });
    }

    // 显示预测结果区域
    showPredictionSection() {
        this.predictionSection.style.display = 'block';
    }
}

// 预测管理类
class PredictionManager {
    constructor() {
        this.chartManager = new ChartManager();
        this.uiManager = new UIManager();
        this.initializeEventListeners();
    }

    // 初始化事件监听器
    initializeEventListeners() {
        document.getElementById('startPredictionBtn').addEventListener('click', () => this.startPrediction());
        window.addEventListener('resize', () => this.chartManager.resizeAllCharts());
    }

    // 启动预测
    async startPrediction() {
        this.uiManager.showLoading();
        try {
            const duration = document.getElementById('predictionDuration').value;
            const response = await fetch(`http://localhost:5000/api/predict?duration=${duration}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.uiManager.showPredictionSection();
            this.uiManager.updateWeatherCodeLegend(data.unique_weather_codes);
            this.chartManager.renderPredictionCharts(data);
        } catch (error) {
            this.uiManager.showError('预测服务启动失败: ' + error.message);
        } finally {
            this.uiManager.hideLoading();
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new PredictionManager();
}); 