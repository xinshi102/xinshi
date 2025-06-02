// 图表管理类
class ChartManager {
    constructor() {
        this.charts = {};
        this.dataCache = {
            temperature: { hour: null, day: null },
            humidity: { hour: null, day: null }
        };
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

    // 渲染温度趋势图
    renderTemperatureTrendChart(data, timeUnit = 'hour') {
        this.charts.temperatureTrend = this.safeInitEcharts('temperatureTrendChart');
        const option = {
            title: { text: '温度变化趋势' },
            tooltip: {//鼠标悬停
                trigger: 'axis',
                formatter: (params) => {
                    const time = params[0].axisValue;
                    const value = params[0].value.toFixed(2);
                    return `${time}<br/>温度: ${value}°C`;
                }
            },
            xAxis: {
                type: 'category',
                data: data.times,
                axisLabel: { rotate: 45 }
            },
            yAxis: {
                type: 'value',
                name: '温度 (°C)'
            },
            series: [{
                name: '温度',
                type: 'line',
                data: data.values,
                smooth: true
            }]
        };
        this.charts.temperatureTrend.setOption(option);
    }

    // 渲染湿度趋势图
    renderHumidityTrendChart(data, timeUnit = 'hour') {
        this.charts.humidityTrend = this.safeInitEcharts('humidityTrendChart');
        const option = {
            title: { text: '湿度变化趋势' },
            tooltip: {
                trigger: 'axis',
                formatter: (params) => {
                    const time = params[0].axisValue;
                    const value = params[0].value.toFixed(2);
                    return `${time}<br/>湿度: ${value}%`;
                }
            },
            xAxis: {
                type: 'category',
                data: data.times,
                axisLabel: { rotate: 45 }
            },
            yAxis: {
                type: 'value',
                name: '湿度 (%)'
            },
            series: [{
                name: '湿度',
                type: 'line',
                data: data.values,
                smooth: true
            }]
        };
        this.charts.humidityTrend.setOption(option);
    }

    // 渲染降水量分布图
    renderPrecipitationDistributionChart(data) {
        this.charts.precipitationDistribution = this.safeInitEcharts('precipitationDistributionChart');
        const option = {
            title: { text: '降水量分布' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: data.categories,
                name: '降水量范围 (mm)'
            },
            yAxis: {
                type: 'value',
                name: '频次'
            },
            series: [{
                name: '降水量',
                type: 'bar',
                data: data.values
            }]
        };
        this.charts.precipitationDistribution.setOption(option);
    }

    // 渲染天气类型分布图
    renderWeatherDistributionChart(data) {
        this.charts.weatherDistribution = this.safeInitEcharts('weatherDistributionChart');
        const option = {
            title: { text: '天气类型分布' },
            tooltip: { trigger: 'item' },
            series: [{
                name: '天气类型',
                type: 'pie',
                radius: '50%',
                data: data.map(item => ({
                    name: item.description,
                    value: item.count
                }))
            }]
        };
        this.charts.weatherDistribution.setOption(option);
    }

    // 渲染云量分布图
    renderCloudCoverDistributionChart(data) {
        this.charts.cloudCoverDistribution = this.safeInitEcharts('cloudCoverDistributionChart');
        // 横坐标每10%一组
        const bins = [];
        for (let i = 0; i < 100; i += 10) {
            bins.push(`${i}~${i+10}`);
        }
        // 统计每组的数量
        const values = new Array(bins.length).fill(0);
        if (data.categories && data.values) {
            data.categories.forEach((val, idx) => {
                const percent = Number(val);
                const groupIdx = Math.min(Math.floor(percent / 10), bins.length - 1);
                values[groupIdx] += data.values[idx];
            });
        }
        const option = {
            title: { text: '云量分布' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: bins,
                name: '云量范围 (%)'
            },
            yAxis: {
                type: 'value',
                name: '频次'
            },
            series: [{
                name: '云量',
                type: 'bar',
                data: values
            }]
        };
        this.charts.cloudCoverDistribution.setOption(option);
    }

    // 渲染风速分布图
    renderWindSpeedDistributionChart(data) {
        this.charts.windSpeedDistribution = this.safeInitEcharts('windSpeedDistributionChart');
        // 横坐标美观且仅保留两位小数
        let categories = [];
        if (data.categories) {
            for (let i = 0; i < data.categories.length; i++) {
                const start = Number(data.categories[i]).toFixed(2);
                let end = '';
                if (i + 1 < data.categories.length) {
                    end = Number(data.categories[i + 1]).toFixed(2);
                } else {
                    end = '';
                }
                if (end) {
                    categories.push(`${start}~${end}`);
                } else {
                    categories.push(`${start}+`);
                }
            }
        }
        const option = {
            title: { text: '风速分布' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: categories,
                name: '风速范围 (m/s)',
                axisLabel: { rotate: 30 }
            },
            yAxis: {
                type: 'value',
                name: '频次'
            },
            series: [{
                name: '风速',
                type: 'bar',
                data: data.values
            }]
        };
        this.charts.windSpeedDistribution.setOption(option);
    }

    // 渲染相关性热力图
    renderCorrelationHeatmapChart(data) {
        this.charts.correlationHeatmap = this.safeInitEcharts('correlationHeatmapChart');
        const option = {
            title: { text: '特征相关性热力图' },
            tooltip: { position: 'top' },
            grid: {
                height: '50%',
                top: '10%'
            },
            xAxis: {
                type: 'category',
                data: data.features,
                splitArea: { show: true }
            },
            yAxis: {
                type: 'category',
                data: data.features,
                splitArea: { show: true }
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '15%'
            },
            series: [{
                name: '相关性',
                type: 'heatmap',
                data: data.correlations,
                label: { show: true },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        };
        this.charts.correlationHeatmap.setOption(option);
    }

    // 调整所有图表大小
    resizeAllCharts() {
        Object.values(this.charts).forEach(chart => chart && chart.resize());//对存在的图表使用resize方法
    }
}

// UI管理类
class UIManager {
    constructor() {
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorAlert = document.getElementById('errorAlert');//错误提示框
        this.errorMessage = document.getElementById('errorMessage');
        this.uploadSection = document.querySelector('.upload-section');//文件上传区域
        this.analysisSection = document.querySelector('.analysis-section');
        this.weatherCodesList = document.getElementById('weatherCodesList');//天气代码及描述的列表
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

    // 显示分析界面
    showAnalysisSection() {
        this.uploadSection.style.display = 'none';
        this.analysisSection.style.display = 'block';
    }

    // 更新天气代码列表
    updateWeatherCodesList(weatherCodes) {
        this.weatherCodesList.innerHTML = weatherCodes.map(code => 
            `<div>${code.code}: ${code.description}</div>`
        ).join('');
    }
}

// 数据分析管理类
class DataAnalysisManager {
    constructor() {
        this.chartManager = new ChartManager();//图表管理
        this.uiManager = new UIManager();//ui管理
        this.initializeEventListeners();//初始化事件监听
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 文件上传相关
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (event) => {
            this.handleFileUpload(event);
        });

        // 温度趋势图时间单位切换按钮
        document.querySelectorAll('.temperature-time-btn').forEach(button => {
            button.addEventListener('click', async () => {
                button.parentElement.querySelectorAll('.temperature-time-btn').forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                const timeUnit = button.dataset.timeUnit;
                this.uiManager.showLoading();
                try {
                    let url = `http://localhost:5000/api/analysis-data`;
                    if (timeUnit) url += `?time_unit=${timeUnit}`;
                    const response = await fetch(url);
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.error || '数据加载失败');
                    this.chartManager.renderTemperatureTrendChart(data.temperature_trend, timeUnit);
                } catch (error) {
                    this.uiManager.showError(error.message || '数据加载失败，请重试');
                } finally {
                    this.uiManager.hideLoading();
                }
            });
        });

        // 湿度趋势图时间单位切换按钮
        document.querySelectorAll('.humidity-time-btn').forEach(button => {
            button.addEventListener('click', async () => {
                button.parentElement.querySelectorAll('.humidity-time-btn').forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                const timeUnit = button.dataset.timeUnit;
                this.uiManager.showLoading();
                try {
                    let url = `http://localhost:5000/api/analysis-data`;
                    if (timeUnit) url += `?time_unit=${timeUnit}`;
                    const response = await fetch(url);
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.error || '数据加载失败');
                    this.chartManager.renderHumidityTrendChart(data.humidity_trend, timeUnit);
                } catch (error) {
                    this.uiManager.showError(error.message || '数据加载失败，请重试');
                } finally {
                    this.uiManager.hideLoading();
                }
            });
        });

        // 其他图表的时间单位切换（如有，可仿照上面写法单独实现）

        // 窗口大小改变时调整图表
        window.addEventListener('resize', () => {
            this.chartManager.resizeAllCharts();
        });
    }

    // 处理文件上传
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.csv')) {
            this.uiManager.showError('请上传CSV格式的文件');
            return;
        }

        this.uiManager.showLoading();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/api/upload', {//上传文件
                method: 'POST',
                body: formData//表单数据
            });
            const data = await response.json();//获取响应数据
            
            if (!response.ok) {
                throw new Error(data.error || '文件上传失败');
            }

            // 清空文件输入框，允许重复上传相同文件
            event.target.value = '';
            
            // 显示分析界面并加载数据
            this.uiManager.showAnalysisSection();
            await this.loadAnalysisData();//加载分析数据
        } catch (error) {
            this.uiManager.showError(error.message || '文件上传失败，请重试');
            event.target.value = '';
        } finally {
            this.uiManager.hideLoading();
        }
    }

    // 加载分析数据
    async loadAnalysisData(timeUnit = 'hour') {
        this.uiManager.showLoading();
        try {
            // 修正API路径
            let url = `http://localhost:5000/api/analysis-data`;
            if (timeUnit) {
                url += `?time_unit=${timeUnit}`;//获取相应数据
            }
            const response = await fetch(url);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '数据加载失败');
            }

            // 缓存数据，便于切换时直接用
            this.lastData = data;

            // 渲染所有图表
            this.chartManager.renderTemperatureTrendChart(data.temperature_trend, timeUnit);
            this.chartManager.renderHumidityTrendChart(data.humidity_trend, timeUnit);
            this.chartManager.renderPrecipitationDistributionChart(data.precipitation_distribution);
            this.chartManager.renderWeatherDistributionChart(data.weather_distribution);
            this.chartManager.renderCloudCoverDistributionChart(data.cloud_cover_distribution);
            this.chartManager.renderWindSpeedDistributionChart(data.wind_speed_distribution);
            // 修正相关性热力图渲染逻辑
            if (data.correlation_heatmap && data.correlation_heatmap.features && data.correlation_heatmap.correlations) {
                // 需要将相关性数据转换为echarts热力图需要的格式，并保留两位小数
                const features = data.correlation_heatmap.features;
                const matrix = [];
                for (let i = 0; i < features.length; i++) {
                    for (let j = 0; j < features.length; j++) {
                        matrix.push([i, j, Number(data.correlation_heatmap.correlations[i][j]).toFixed(2)]);
                    }
                }
                this.chartManager.renderCorrelationHeatmapChart({
                    features: features,
                    correlations: matrix
                });
            }

            // 更新天气代码列表
            this.uiManager.updateWeatherCodesList(data.weather_codes);
        } catch (error) {
            this.uiManager.showError(error.message || '数据加载失败，请重试');
        } finally {
            this.uiManager.hideLoading();
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new DataAnalysisManager();
}); 