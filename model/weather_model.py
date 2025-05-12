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
        self.weather_categories = weather_categories  # 10种天气类型
        
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
        
        # 改进的降水量预测网络
        self.precip_base = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 8),
            nn.Linear(hidden_size // 8, 1),
            nn.ReLU()  # 确保降水量非负
        )
        
        # 添加降水量分类器（10种天气类型）
        self.precip_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, 10)  # 10种天气类型
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
        
        # 定义每个天气类型对应的典型降水量范围
        self.precip_adjustments = torch.tensor([
            0.0,    # 0: 晴天
            0.0,    # 1: 较为晴朗
            0.0,    # 2: 局部多云
            0.0,    # 3: 阴天
            0.05,   # 51: 轻度毛毛雨 (0.01-0.1mm)
            0.15,   # 53: 中度毛毛雨 (0.1-0.2mm)
            0.35,   # 55: 重度毛毛雨 (0.2-0.5mm)
            1.25,   # 61: 小雨 (0.5-2mm)
            3.5,    # 63: 中雨 (2-5mm)
            7.5     # 65: 大雨 (>5mm)
        ])
    
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
        
        # 预测降水量类别
        precip_class = self.precip_classifier(precip_features)
        
        # 根据类别调整降水量预测
        precip_class_probs = F.softmax(precip_class, dim=-1)
        
        # 将precip_adjustments移动到正确的设备上
        precip_adjustments = self.precip_adjustments.to(precip_pred.device)
        
        # 使用类别概率加权计算降水量调整值
        precip_adjustment = torch.sum(precip_class_probs * precip_adjustments.view(1, 1, -1), dim=-1, keepdim=True)
        
        # 最终降水量预测
        precip_pred = precip_pred + precip_adjustment
        
        # 从降水量预测天气代码
        precip_weather_pred = self.precip_to_weather(precip_pred)
        
        # 融合天气代码预测
        weather_inputs = torch.cat([base_weather_pred, precip_weather_pred], dim=-1)
        weather_pred = self.weather_fusion(weather_inputs)
        
        return weather_pred, temp_pred, humidity_pred, precip_pred 

def calculate_loss(outputs, targets):
    # 定义天气类型的层次关系矩阵（10x10）
    weather_hierarchy = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 0: 晴天
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],  # 1: 较为晴朗
        [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],  # 2: 局部多云
        [3, 2, 1, 0, 1, 2, 3, 4, 5, 6],  # 3: 阴天
        [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],  # 51: 毛毛雨(轻)
        [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],  # 53: 毛毛雨(中)
        [6, 5, 4, 3, 2, 1, 0, 1, 2, 3],  # 55: 毛毛雨(重)
        [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],  # 61: 雨(轻)
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 1],  # 63: 雨(中)
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]   # 65: 雨(重)
    ], device=outputs[0].device)
    
    # 基础交叉熵损失
    weather_loss = F.cross_entropy(outputs[0], targets['weather_type'])
    
    # 计算预测和目标的天气类型
    pred_weather = torch.argmax(outputs[0], dim=-1)
    target_weather = targets['weather_type']
    
    # 获取层次距离
    batch_size = pred_weather.size(0)
    hierarchy_distances = weather_hierarchy[pred_weather, target_weather]
    
    # 计算层次惩罚损失
    hierarchy_loss = torch.mean(hierarchy_distances.float())
    
    # 总天气损失 = 基础损失 + 层次惩罚
    total_weather_loss = weather_loss + 0.1 * hierarchy_loss
    
    # 其他任务的损失
    temp_loss = F.mse_loss(outputs[1], targets['temperature'])
    humidity_loss = F.mse_loss(outputs[2], targets['humidity'])
    
    # 改进的降水损失计算
    precip_pred = outputs[3]
    precip_target = targets['precipitation']
    weather_target = targets['weather_type']
    
    # 计算降水量分类损失（使用天气类型作为目标）
    precip_class = outputs[0]  # 使用天气预测结果作为降水量分类
    precip_class_loss = F.cross_entropy(precip_class, weather_target.view(-1))
    
    # 计算降水量回归损失（使用Huber损失）
    precip_regression_loss = F.smooth_l1_loss(precip_pred, precip_target)
    
    # 计算降水量权重
    precip_weights = torch.ones_like(precip_target)
    # 根据天气类型设置权重
    precip_weights[weather_target == 51] = 1.5  # 毛毛雨(轻)
    precip_weights[weather_target == 53] = 1.8  # 毛毛雨(中)
    precip_weights[weather_target == 55] = 2.0  # 毛毛雨(重)
    precip_weights[weather_target == 61] = 2.5  # 雨(轻)
    precip_weights[weather_target == 63] = 3.0  # 雨(中)
    precip_weights[weather_target == 65] = 3.5  # 雨(重)
    
    # 加权降水量损失
    precip_loss = (0.3 * precip_class_loss + 0.7 * precip_regression_loss) * precip_weights.mean()
    
    # 总损失
    total_loss = total_weather_loss + temp_loss + humidity_loss + precip_loss
    
    return total_loss 