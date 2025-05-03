import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeatherForecastModel(nn.Module):
    """
    天气预测模型
    
    该模型使用特征交互层、双向LSTM和单向LSTM的组合来预测天气数据。
    模型结构：
    1. 特征交互层：学习特征之间的关系
    2. 双向LSTM层：捕获时间序列的双向依赖关系
    3. 单向LSTM层：进一步处理特征
    4. 特征融合层：融合所有时间步的信息
    5. 输出层：预测温度、降水量和风速
    
    参数：
        input_size (int): 输入特征维度，默认为8（时间、天气代码、温度、降水量、风速等）
        hidden_size (int): LSTM隐藏层大小，默认为128
        num_layers (int): LSTM层数，默认为2
        dropout (float): Dropout比率，默认为0.3
    """
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super(WeatherForecastModel, self).__init__()
        
        # 特征交互层：学习特征之间的关系
        self.feature_interaction = nn.Sequential(
            nn.Linear(input_size, hidden_size),# 加权组合将输入特征进行组合成更高层的输入特征
            nn.ReLU(),#避免模型仅学习线性关系
            nn.Dropout(dropout),#防止过拟合，随机屏蔽30%的神经元
            nn.Linear(hidden_size, hidden_size)#进一步混合特征，学习高阶交互
        )
        
        # 双向LSTM层：捕获时间序列的双向依赖关系
        self.bi_lstm = nn.LSTM(
            input_size=hidden_size, #保持维度一致 为128 linear以及已经将原始数据升维
            hidden_size=hidden_size, #避免维度爆炸
            num_layers=num_layers,#堆叠lstm层数，学习更复杂的时间依赖
            batch_first=True,
            bidirectional=True,#启用双向lstm结构
            dropout=dropout#防止过拟合，随机丢弃
        )
        
        # 单向LSTM层：进一步处理特征
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,#输入维度因为之前的双向堆叠所以翻倍
            hidden_size=hidden_size,# 保持维度一致
            num_layers=1,
            batch_first=True
        )
        
        # 特征融合层：融合所有时间步的信息
        self.feature_fusion = nn.Sequential(#整合时序特征的正反向关系
            nn.Linear(hidden_size * 2, hidden_size),#降维度，减少冗余
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 温度输出层（预测最高、最低、平均温度）
        self.temp_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 9)  # 3天 * 3个温度值
        )
        
        # 降水量输出层（预测总降水量和降水时长）
        self.precip_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, 6),  # 3天 * 2个降水量相关值
            nn.Softplus()
        )
        
        # 风速输出层
        self.wind_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, 3),  # 3天的最大风速
            nn.Softplus()#softplus比relu更加平滑
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 特征交互：学习特征之间的关系
        interacted_features = self.feature_interaction(x)
        
        # 双向LSTM处理
        bi_out, _ = self.bi_lstm(interacted_features)
        
        # 单向LSTM处理
        lstm_out, _ = self.lstm(bi_out)
        
        # 特征融合：考虑所有时间步的信息
        last_hidden = lstm_out[:, -1, :]
        avg_hidden = lstm_out.mean(dim=1)
        fused_features = self.feature_fusion(torch.cat([last_hidden, avg_hidden], dim=1))
        
        # 预测温度值（最高、最低、平均）
        temp_values = self.temp_output(fused_features)
        temp_values = temp_values.view(-1, 3, 3)  # [batch_size, 3, 3]
        
        # 预测降水量（总降水量和降水时长）
        precip_values = self.precip_output(fused_features)
        precip_values = precip_values.view(-1, 3, 2)  # [batch_size, 3, 2]
        
        # 预测风速
        wind_values = self.wind_output(fused_features)
        wind_values = wind_values.view(-1, 3, 1)  # [batch_size, 3, 1]
        
        return temp_values, precip_values, wind_values

class WeightedMSELoss(nn.Module):
    """
    基于不确定性的多任务学习损失函数
    
    该损失函数使用可学习的任务权重（不确定性）来自动平衡不同任务的损失。
    权重通过softmax函数确保和为1，并通过温度参数控制权重的分布。
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # 初始化任务权重参数（logits）
        self.task_weights = nn.Parameter(torch.zeros(3))  # 3个任务：温度、降水量、风速
        
    def forward(self, pred_temp, pred_precip, pred_wind, true_temp, true_precip, true_wind):
        """
        计算加权MSE损失
        
        参数:
            pred_temp: 预测的温度值 [batch_size, 3, 3] (3天，每天3个温度值)
            pred_precip: 预测的降水量 [batch_size, 3, 2] (3天，每天2个降水量相关值)
            pred_wind: 预测的风速 [batch_size, 3, 1] (3天)
            true_temp: 真实的温度值 [batch_size, 3, 3]
            true_precip: 真实的降水量 [batch_size, 3, 2]
            true_wind: 真实的风速 [batch_size, 3, 1]
        """
        # 计算各个任务的MSE损失
        temp_loss = F.mse_loss(pred_temp, true_temp)
        precip_loss = F.mse_loss(pred_precip, true_precip)
        wind_loss = F.mse_loss(pred_wind, true_wind)
        
        # 使用softmax计算任务权重
        weights = F.softmax(self.task_weights / self.temperature, dim=0)
        
        # 计算加权总损失
        total_loss = (weights[0] * temp_loss + 
                     weights[1] * precip_loss + 
                     weights[2] * wind_loss)
        
        # 添加权重正则化项，防止某个任务的权重过大
        weight_entropy = -(weights * torch.log(weights + 1e-6)).sum()
        total_loss = total_loss + 0.01 * weight_entropy
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'temp_loss': temp_loss.item(),
            'precip_loss': precip_loss.item(),
            'wind_loss': wind_loss.item(),
            'weights': weights.detach().cpu().numpy()
        }

def prepare_data(data, seq_length=7):
    """
    准备时间序列数据
    
    将原始数据转换为序列数据，用于模型训练。
    每个序列包含seq_length天的数据，用于预测后续3天的天气。
    
    参数：
        data: 原始数据，形状为 (n_samples, n_features)
        seq_length: 序列长度，默认为7天
        
    返回：
        sequences: 输入序列数据
        targets: 目标序列数据
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - 3):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 3]
        sequences.append(seq)
        targets.append(target)
    
    if len(sequences) > 0:
        sequences = np.array(sequences)
        targets = np.array(targets)
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    else:
        return torch.FloatTensor([]), torch.FloatTensor([])