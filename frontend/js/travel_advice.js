// 出行建议管理类
class TravelAdviceManager {
    constructor() {
        this.uiManager = new UIManager();
        this.weatherManager = new WeatherManager();
        this.initializeEventListeners();
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 页面加载时获取天气数据
        document.addEventListener('DOMContentLoaded', () => {
            this.loadWeatherData();
        });
    }

    // 加载天气数据
    async loadWeatherData() {
        this.uiManager.showLoading();
        try {
            const weatherData = await this.weatherManager.getCurrentWeather();
            this.uiManager.showAdviceSection();
            this.uiManager.updateAdvice(weatherData.weather_code);
        } catch (error) {
            this.uiManager.showError('获取天气数据失败: ' + error.message);
        } finally {
            this.uiManager.hideLoading();
        }
    }
}

// UI管理类
class UIManager {
    constructor() {
        this.loadingSpinner = document.getElementById('loadingSpinner');//加载动画
        this.errorAlert = document.getElementById('errorAlert');//错误提示框
        this.errorMessage = document.getElementById('errorMessage');
        this.adviceSection = document.getElementById('adviceSection');//建议区域
        this.adviceCard = document.getElementById('adviceCard');
        this.weatherIcon = document.getElementById('weatherIcon');//天气图标
        this.adviceText = document.getElementById('adviceText');
        this.adviceDetails = document.getElementById('adviceDetails');//建议详情
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

    // 显示建议区域
    showAdviceSection() {
        this.adviceSection.style.display = 'block';
    }

    // 更新出行建议
    updateAdvice(weatherCode) {
        const advice = this.getAdviceByWeatherCode(weatherCode);
        this.adviceCard.className = `travel-advice ${advice.type}`;
        this.weatherIcon.innerHTML = advice.icon;
        this.adviceText.textContent = advice.text;
        this.adviceDetails.textContent = advice.details;
    }

    // 根据天气代码获取建议
    getAdviceByWeatherCode(weatherCode) {
        const adviceMap = {
            // 晴天或部分晴天
            '0,1': {
                type: 'warning',
                icon: '<i class="bi bi-sun"></i>',
                text: '阳光明媚，但紫外线有点调皮哦~',
                details: '记得涂防晒霜，戴上可爱的帽子和太阳镜，保护好自己才能美美哒~'
            },
            // 多云
            '2,3': {
                type: 'good',
                icon: '<i class="bi bi-cloud-sun"></i>',
                text: '今天是个适合约会的好天气呢~',
                details: '微风不燥，阳光正好，约上小伙伴一起去感受春天的气息吧！'
            },
            // 毛毛雨
            '51,52,53': {
                type: 'warning',
                icon: '<i class="bi bi-cloud-drizzle"></i>',
                text: '小雨淅淅沥沥，记得带上你的小伞伞~',
                details: '虽然雨不大，但也要注意保暖哦，感冒了可不好受呢~'
            },
            // 小雨
            '61': {
                type: 'warning',
                icon: '<i class="bi bi-cloud-rain"></i>',
                text: '小雨滴答滴答，出门要记得带伞哦~',
                details: '穿上防水鞋，带上雨伞，让雨滴成为你出行的伴奏曲~'
            },
            // 大雨
            '62,63': {
                type: 'danger',
                icon: '<i class="bi bi-cloud-rain-heavy"></i>',
                text: '大雨哗啦啦，不如在家喝杯热茶~',
                details: '如果一定要出门，记得穿好雨衣，带上雨伞，路上要小心哦~'
            }
        };

        // 查找匹配的天气代码
        for (const [codes, advice] of Object.entries(adviceMap)) {
            if (codes.split(',').includes(weatherCode.toString())) {
                return advice;
            }
        }

        // 默认建议
        return {
            type: 'warning',
            icon: '<i class="bi bi-question-circle"></i>',
            text: '天气状况未知，建议谨慎出行~',
            details: '请关注天气预报，做好防护准备~'
        };
    }
}

// 天气数据管理类
class WeatherManager {
    // 获取当前天气数据
    async getCurrentWeather() {
        const response = await fetch('http://localhost:5000/api/current-weather');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        return data;
    }
}

// 初始化应用
new TravelAdviceManager(); 