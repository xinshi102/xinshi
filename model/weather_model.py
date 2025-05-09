import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, weather_categories):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.weather_categories = weather_categories
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # 添加dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 共享特征提取层
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2)
        )
        
        # 天气代码分类器 - 使用注意力机制
        self.weather_attention = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.weather_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, weather_categories)
        )
        
        # 温度回归器 - 添加Sigmoid激活函数
        self.temp_regressor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 将输出限制在0-1之间
        )
        
        # 湿度回归器 - 添加Sigmoid激活函数
        self.humidity_regressor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 将输出限制在0-1之间
        )
        
        # 改进的降水预测系统
        self.precip_attention = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.precip_base = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.ReLU()  # 确保降水量非负
        )
        
        # 降水量到天气代码的转换层
        self.precip_to_weather = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, weather_categories)
        )
        
        # 天气代码融合层
        self.weather_fusion = nn.Sequential(
            nn.Linear(weather_categories * 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, weather_categories)
        )
        
        # 温度范围参数
        self.temp_min = -20.0
        self.temp_max = 40.0
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # 确保序列长度为72
        batch_size, seq_len, hidden = lstm_out.shape
        if seq_len > 72:
            lstm_out = lstm_out[:, :72, :]
        elif seq_len < 72:
            last_step = lstm_out[:, -1:, :]
            padding = last_step.repeat(1, 72 - seq_len, 1)
            lstm_out = torch.cat([lstm_out, padding], dim=1)
        
        # 提取共享特征
        shared_features = self.shared_features(lstm_out)
        
        # 计算天气注意力权重
        attention_weights = self.weather_attention(shared_features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 应用注意力机制
        attended_features = shared_features * attention_weights
        
        # 基础天气代码预测
        base_weather_pred = self.weather_classifier(attended_features)
        
        # 基础预测
        temp_pred = self.temp_regressor(shared_features)
        # 将温度映射到实际范围
        temp_pred = temp_pred * (self.temp_max - self.temp_min) + self.temp_min
        
        humidity_pred = self.humidity_regressor(shared_features)
        # 将湿度映射到0-100%
        humidity_pred = humidity_pred * 100.0
        
        # 计算降水量注意力权重
        precip_attention = self.precip_attention(shared_features)
        precip_attention = F.softmax(precip_attention, dim=1)
        
        # 应用降水量注意力机制
        precip_features = shared_features * precip_attention
        
        # 预测降水量
        precip_pred = self.precip_base(precip_features)
        
        # 从降水量预测天气代码
        precip_weather_pred = self.precip_to_weather(precip_pred)
        
        # 融合天气代码预测
        weather_inputs = torch.cat([base_weather_pred, precip_weather_pred], dim=-1)
        weather_pred = self.weather_fusion(weather_inputs)
        
        return weather_pred, temp_pred, humidity_pred, precip_pred 