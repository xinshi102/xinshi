import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

<<<<<<< HEAD
class WeatherLSTM(nn.Module):#继承module类，自定义lstm模型
    def __init__(self, input_size, hidden_size, num_layers, output_size, weather_categories):
        super(WeatherLSTM, self).__init__()#初始化
=======
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, weather_categories):
        super(WeatherLSTM, self).__init__()
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        self.hidden_size = hidden_size
        self.weather_categories = weather_categories  # 10种天气类型
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
<<<<<<< HEAD
            dropout=0.3 if num_layers > 1 else 0#防止两层lstm层过拟合
        )
        
        # 添加dropout层
        self.dropout = nn.Dropout(0.3)#防止全连接层过拟合
        
        # 共享特征提取层
        self.shared_features = nn.Sequential(#提供通用的天气特征，并且被多个任务共享使用//容器类，包含多层
            nn.Linear(hidden_size, hidden_size // 2),#降低一半特征维度
=======
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # 添加dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 共享特征提取层
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2)
        )
        
        # 云量特征提取层
        self.cloud_features = nn.Sequential(
<<<<<<< HEAD
            nn.Linear(hidden_size // 2, hidden_size // 4),#相对简单的云量特征，降低成四分之一
            nn.ReLU(),#截断负值
            nn.LayerNorm(hidden_size // 4),#层归一化，加速训练
=======
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            nn.Dropout(0.2)
        )
        
        # 天气代码分类器 - 使用注意力机制
        self.weather_attention = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
<<<<<<< HEAD
            nn.Tanh(),#归一化，映射到【-1，1】，增加非线性表达能力，跟后续的softmax配合得到最终的注意力分数
            nn.Linear(hidden_size // 4, 1)#注意力机制只需要一个权重值，所以降维1
=======
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        )
        
        # 云量注意力机制
        self.cloud_attention = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
<<<<<<< HEAD
            nn.Tanh(),#双曲正切
            nn.Linear(hidden_size // 8, 1)#同理
=======
            nn.Tanh(),
            nn.Linear(hidden_size // 8, 1)
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        )
        
        # 天气分类器 - 结合云量特征
        self.weather_classifier = nn.Sequential(
<<<<<<< HEAD
            nn.Linear(hidden_size // 2 + hidden_size // 4 + hidden_size // 2 + hidden_size // 4, hidden_size // 2),#共享特征+云量特征+天气注意力特征+降水特征
=======
            nn.Linear(hidden_size // 2 + hidden_size // 4 + hidden_size // 2 + hidden_size // 4, hidden_size // 2),
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, weather_categories)
        )
        
        # 温度回归器
        self.temp_regressor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 湿度回归器
        self.humidity_regressor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, 1),
            nn.ReLU()
        )
        
        # 改进的降水预测系统
        self.precip_attention = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 改进的降水量预测网络 - 结合云量特征
        self.precip_base = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 8),
<<<<<<< HEAD
            nn.Linear(hidden_size // 8, 1)
=======
            nn.Linear(hidden_size // 8, 1),
            nn.ReLU()
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
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
            0.01,   # 51: 轻度毛毛雨 (0.01-0.1mm)
            0.1,    # 53: 中度毛毛雨 (0.1-0.2mm)
            0.2,    # 55: 重度毛毛雨 (0.2-0.5mm)
            0.5,    # 61: 小雨 (0.5-2mm)
            2.0,    # 63: 中雨 (2-5mm)
            5.0     # 65: 大雨 (>5mm)
        ])
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
<<<<<<< HEAD
        lstm_out = self.dropout(lstm_out)#这里又dropout了一次，防止过拟合
        
        # 确保序列长度为72
        batch_size, seq_len, hidden = lstm_out.shape#获取输出形状
        if seq_len > 72:
            lstm_out = lstm_out[:, :72, :]
        elif seq_len < 72:
            last_step = lstm_out[:, -1:, :]#如果序列长度不够，用最后一个时间步来提取
            padding = last_step.repeat(1, 72 - seq_len, 1)
            lstm_out = torch.cat([lstm_out, padding], dim=1)#拼接
=======
        lstm_out = self.dropout(lstm_out)
        
        # 确保序列长度为72
        batch_size, seq_len, hidden = lstm_out.shape
        if seq_len > 72:
            lstm_out = lstm_out[:, :72, :]
        elif seq_len < 72:
            last_step = lstm_out[:, -1:, :]
            padding = last_step.repeat(1, 72 - seq_len, 1)
            lstm_out = torch.cat([lstm_out, padding], dim=1)
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        
        # 提取共享特征
        shared_features = self.shared_features(lstm_out)
        
        # 提取云量特征
<<<<<<< HEAD
        cloud_features = self.cloud_features(shared_features)#通过在共享特征里面提取云量特征，系统能更好的学习云量与其他天气特征的关系
        
        # 计算云量注意力权重
        cloud_attention = self.cloud_attention(cloud_features)
        cloud_attention = F.softmax(cloud_attention, dim=1)#归一化
        cloud_features = cloud_features * cloud_attention#加权，突出特征
        
        # 计算天气注意力权重
        weather_attention = self.weather_attention(shared_features)
        weather_attention = F.softmax(weather_attention, dim=1)#指定在序列长度维度上计算注意力权重
        attended_features = shared_features * weather_attention#共享特征主要为天气代码服务
        
        # 合并特征用于降水预测
        precip_input = torch.cat([shared_features, cloud_features], dim=-1)#特征加强过的云量加上共享为降水服务
=======
        cloud_features = self.cloud_features(shared_features)
        
        # 计算云量注意力权重
        cloud_attention = self.cloud_attention(cloud_features)
        cloud_attention = F.softmax(cloud_attention, dim=1)
        cloud_features = cloud_features * cloud_attention
        
        # 计算天气注意力权重
        weather_attention = self.weather_attention(shared_features)
        weather_attention = F.softmax(weather_attention, dim=1)
        attended_features = shared_features * weather_attention
        
        # 合并特征用于降水预测
        precip_input = torch.cat([shared_features, cloud_features], dim=-1)
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        # 计算降水注意力权重
        precip_attention = self.precip_attention(precip_input)
        precip_attention = F.softmax(precip_attention, dim=1)
        precip_features = precip_input * precip_attention
        # 预测降水量
        precip_pred = self.precip_base(precip_features)
        
        # 合并特征用于天气分类（加入降水特征，实现特征交互）
        weather_input = torch.cat([attended_features, cloud_features, precip_features], dim=-1)
        weather_pred = self.weather_classifier(weather_input)
        
        # 温度预测
        temp_pred = self.temp_regressor(shared_features)
        temp_pred = torch.clamp(temp_pred, min=-20.0, max=40.0)  # 直接限制温度范围
        
        # 湿度预测
        humidity_pred = self.humidity_regressor(shared_features)
        humidity_pred = torch.clamp(humidity_pred, min=0.0, max=100.0)  # 限制湿度范围
        
        return weather_pred, temp_pred, humidity_pred, precip_pred

def calculate_loss(outputs, targets):
    # 定义天气类型的层次关系矩阵（10x10）
    weather_hierarchy = torch.tensor([
        [0,   0.1, 0.1, 0.1, 1,   1.1, 1.2, 2,   2.1, 2.2],   # 晴天
        [0.1, 0,   0.1, 0.1, 1,   1.1, 1.2, 2,   2.1, 2.2],   # 较为晴朗
        [0.1, 0.1, 0,   0.1, 1,   1.1, 1.2, 2,   2.1, 2.2],   # 局部多云
        [0.1, 0.1, 0.1, 0,   1,   1.1, 1.2, 2,   2.1, 2.2],   # 阴天
        [1,   1,   1,   1,   0,   0.1, 0.2, 1,   1.1, 1.2],   # 毛毛雨(轻)
        [1.1, 1.1, 1.1, 1.1, 0.1, 0,   0.1, 1,   1.1, 1.2],   # 毛毛雨(中)
        [1.2, 1.2, 1.2, 1.2, 0.2, 0.1, 0,   1,   1.1, 1.2],   # 毛毛雨(重)
        [2,   2,   2,   2,   1,   1,   1,   0,   0.1, 0.2],   # 小雨
        [2.1, 2.1, 2.1, 2.1, 1.1, 1.1, 1.1, 0.1, 0,   0.1],   # 中雨
        [2.3, 2.2, 2.1, 2.0, 1.2, 1.1, 1.0, 0.2, 0.1, 0]      # 大雨
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
<<<<<<< HEAD
    # 根据天气类型设置权重（全部提升10倍）
    precip_weights[weather_target == 51] = 15.0  # 毛毛雨(轻)
    precip_weights[weather_target == 53] = 18.0  # 毛毛雨(中)
    precip_weights[weather_target == 55] = 20.0  # 毛毛雨(重)
    precip_weights[weather_target == 61] = 25.0  # 雨(轻)
    precip_weights[weather_target == 63] = 30.0  # 雨(中)
    precip_weights[weather_target == 65] = 35.0  # 雨(重)
=======
    # 根据天气类型设置权重
    precip_weights[weather_target == 51] = 1.5  # 毛毛雨(轻)
    precip_weights[weather_target == 53] = 1.8  # 毛毛雨(中)
    precip_weights[weather_target == 55] = 2.0  # 毛毛雨(重)
    precip_weights[weather_target == 61] = 2.5  # 雨(轻)
    precip_weights[weather_target == 63] = 3.0  # 雨(中)
    precip_weights[weather_target == 65] = 3.5  # 雨(重)
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    
    # 加权降水量损失
    precip_loss = (0.3 * precip_class_loss + 0.7 * precip_regression_loss) * precip_weights.mean()
    
    # 总损失
    total_loss = total_weather_loss + temp_loss + humidity_loss + precip_loss
    
    return total_loss 