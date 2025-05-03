# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modle import WeatherForecastModel, WeightedMSELoss, prepare_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib
import io
import torch.nn.functional as F
from datetime import datetime

# 设置控制台输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_and_preprocess_data(data_path):
    """
    加载和预处理数据
    
    该函数负责：
    1. 加载CSV格式的天气数据
    2. 提取所需的特征
    3. 对数据进行标准化处理
    4. 保存标准化器供预测时使用
    
    参数：
        data_path: 数据文件路径
        
    返回：
        X_scaled: 标准化后的特征数据
        y: 目标数据
    """
    print(f"正在加载数据文件: {data_path}")
    # 加载数据
    data = pd.read_csv(data_path, encoding='utf-8')
    print(f"数据加载完成，形状: {data.shape}")
    
    # 将时间列转换为数值特征
    data['time'] = pd.to_datetime(data['time'])
    data['day_of_year'] = data['time'].dt.dayofyear
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    
    # 提取特征
    features = [
        'day_of_year', 'month', 'day',  # 时间特征
        'weathercode (wmo code)',  # 天气代码
        'temperature_2m_min (°C)', 'temperature_2m_mean (°C)', 'temperature_2m_max (°C)',  # 温度
        'precipitation_sum (mm)', 'precipitation_hours (h)',  # 降水量
        'windspeed_10m_max (m/s)'  # 风速
    ]
    
    # 检查所有特征是否存在
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"警告：以下特征不存在：{missing_features}")
        # 使用可用的特征
        features = [f for f in features if f in data.columns]
        print(f"将使用以下特征：{features}")
    
    X = data[features].values
    print(f"特征提取完成，形状: {X.shape}")
    
    # 创建标准化器字典
    scalers = {}
    
    # 对不同类型的特征使用不同的标准化方法
    # 时间特征使用MinMaxScaler
    time_scaler = MinMaxScaler()
    time_data = X[:, :3]  # 前3列是时间特征
    time_scaled = time_scaler.fit_transform(time_data)
    scalers['time'] = time_scaler
    
    # 天气代码使用MinMaxScaler
    weather_scaler = MinMaxScaler()
    weather_data = X[:, 3:4]  # 天气代码
    weather_scaled = weather_scaler.fit_transform(weather_data)
    scalers['weather'] = weather_scaler
    
    # 温度使用StandardScaler
    temp_scaler = StandardScaler()
    temp_data = X[:, 4:7]  # 温度特征
    temp_scaled = temp_scaler.fit_transform(temp_data)
    scalers['temperature'] = temp_scaler
    
    # 降水量使用MinMaxScaler
    precip_scaler = MinMaxScaler()
    precip_data = X[:, 7:9]  # 降水量特征
    precip_scaled = precip_scaler.fit_transform(precip_data)
    scalers['precipitation'] = precip_scaler
    
    # 风速使用StandardScaler
    wind_scaler = StandardScaler()
    wind_data = X[:, 9:10]  # 风速特征
    wind_scaled = wind_scaler.fit_transform(wind_data)
    scalers['wind'] = wind_scaler
    
    # 合并标准化后的数据
    X_scaled = np.column_stack((
        time_scaled,
        weather_scaled,
        temp_scaled,
        precip_scaled,
        wind_scaled
    ))
    
    # 保存标准化器
    joblib.dump(scalers, 'scalers.pkl')
    print("标准化器已保存")
    
    return X_scaled, X_scaled

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型
    
    该函数负责：
    1. 执行模型训练循环
    2. 在每个epoch后进行验证
    3. 监控各个任务的损失和权重
    4. 保存最佳模型
    
    参数：
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 计算设备（CPU/GPU）
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_temp_loss = 0
        train_precip_loss = 0
        train_wind_loss = 0
        train_weights = np.zeros(3)  # 记录任务权重
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred_temp, pred_precip, pred_wind = model(batch_X)
            
            # 计算损失
            loss, loss_info = criterion(
                pred_temp, pred_precip, pred_wind,
                batch_y[:, :, 4:7],  # 温度值
                batch_y[:, :, 7:9],  # 降水量
                batch_y[:, :, 9:10]  # 风速
            )
            
            loss.backward()
            optimizer.step()
            
            # 累计损失和权重
            train_loss += loss_info['total_loss']
            train_temp_loss += loss_info['temp_loss']
            train_precip_loss += loss_info['precip_loss']
            train_wind_loss += loss_info['wind_loss']
            train_weights += loss_info['weights']
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_temp_loss = 0
        val_precip_loss = 0
        val_wind_loss = 0
        val_weights = np.zeros(3)  # 记录任务权重
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                pred_temp, pred_precip, pred_wind = model(batch_X)
                loss, loss_info = criterion(
                    pred_temp, pred_precip, pred_wind,
                    batch_y[:, :, 4:7],  # 温度值
                    batch_y[:, :, 7:9],  # 降水量
                    batch_y[:, :, 9:10]  # 风速
                )
                
                # 累计损失和权重
                val_loss += loss_info['total_loss']
                val_temp_loss += loss_info['temp_loss']
                val_precip_loss += loss_info['precip_loss']
                val_wind_loss += loss_info['wind_loss']
                val_weights += loss_info['weights']
        
        # 计算平均损失和权重
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        train_loss /= n_train
        train_temp_loss /= n_train
        train_precip_loss /= n_train
        train_wind_loss /= n_train
        train_weights /= n_train
        
        val_loss /= n_val
        val_temp_loss /= n_val
        val_precip_loss /= n_val
        val_wind_loss /= n_val
        val_weights /= n_val
        
        # 打印训练进度
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print('训练阶段:')
        print(f'总损失: {train_loss:.4f}')
        print(f'温度损失: {train_temp_loss:.4f}, 权重: {train_weights[0]:.3f}')
        print(f'降水量损失: {train_precip_loss:.4f}, 权重: {train_weights[1]:.3f}')
        print(f'风速损失: {train_wind_loss:.4f}, 权重: {train_weights[2]:.3f}')
        
        print('\n验证阶段:')
        print(f'总损失: {val_loss:.4f}')
        print(f'温度损失: {val_temp_loss:.4f}, 权重: {val_weights[0]:.3f}')
        print(f'降水量损失: {val_precip_loss:.4f}, 权重: {val_weights[1]:.3f}')
        print(f'风速损失: {val_wind_loss:.4f}, 权重: {val_weights[2]:.3f}')
        
        # 保存最佳模型（基于总验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("\n保存最佳模型")

def prepare_data(X, seq_length=7):
    """
    准备序列数据
    
    Args:
        X: 输入数据，形状为 [n_samples, n_features]
        seq_length: 输入序列长度，默认为7天
        
    Returns:
        X_seq: 序列输入数据，形状为 [n_samples - seq_length - 2, seq_length, n_features]
        y_seq: 序列目标数据，形状为 [n_samples - seq_length - 2, 3, n_features]
    """
    n_samples = len(X)
    n_features = X.shape[1]
    
    # 创建序列数据
    X_seq = []
    y_seq = []
    
    # 每个序列使用前7天预测后3天
    for i in range(n_samples - seq_length - 2):
        X_seq.append(X[i:i+seq_length])  # 前7天作为输入
        y_seq.append(X[i+seq_length:i+seq_length+3])  # 后3天作为目标
    
    # 转换为numpy数组
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 转换为tensor
    X_seq = torch.FloatTensor(X_seq)
    y_seq = torch.FloatTensor(y_seq)
    
    return X_seq, y_seq

def analyze_feature_importance(model, val_loader, device):
    """
    分析模型学习到的特征重要性
    
    通过计算每个特征的梯度大小来评估其重要性。
    梯度越大，说明该特征对模型预测的影响越大。
    
    参数：
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 计算设备
    """
    model.eval()
    feature_gradients = torch.zeros(5).to(device)  # 5个特征：最低温度、最高温度、平均温度、降水量、风速
    n_samples = 0
    
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # 计算每个特征的梯度
        batch_X.requires_grad_(True)
        pred_temp, pred_precip, pred_wind = model(batch_X)
        
        # 计算总损失
        loss = F.mse_loss(pred_temp, batch_y[:, :, 0:3]) + \
               F.mse_loss(pred_precip, batch_y[:, :, 3:4]) + \
               F.mse_loss(pred_wind, batch_y[:, :, 4:5])
        
        # 计算梯度
        loss.backward()
        feature_gradients += torch.abs(batch_X.grad).mean(dim=(0, 1))
        n_samples += batch_X.size(0)
    
    # 计算平均梯度
    feature_gradients /= n_samples
    
    # 打印特征重要性
    feature_names = ['最低温度', '最高温度', '平均温度', '降水量', '风速']
    print("\n特征重要性分析:")
    for name, importance in zip(feature_names, feature_gradients.cpu().numpy()):
        print(f"{name}: {importance:.4f}")
    
    return feature_gradients

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    data_path = 'weather_data/Fouryear.csv'
    X_scaled, y_scaled = load_and_preprocess_data(data_path)
    
    # 准备序列数据
    X_seq, y_seq = prepare_data(X_scaled)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(X_seq))
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:]
    y_val = y_seq[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = X_scaled.shape[1]  # 使用实际的特征数量
    model = WeatherForecastModel(input_size=input_size).to(device)
    
    # 初始化损失函数和优化器
    criterion = WeightedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main() 