# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modle import WeatherForecastModel
import io

# 设置控制台输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_scalers():
    """加载保存的标准化器"""
    try:
        scalers = joblib.load('scalers.pkl')
        return scalers
    except FileNotFoundError:
        print("错误：找不到标准化器文件 'scalers.pkl'")
        sys.exit(1)

def preprocess_input_data(data, scalers):
    """
    预处理输入数据
    
    参数：
        data: 输入数据，DataFrame格式
        scalers: 标准化器字典
        
    返回：
        processed_data: 处理后的数据，tensor格式
    """
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
        print(f"错误：以下特征不存在：{missing_features}")
        print("请检查数据文件中的列名是否正确")
        sys.exit(1)
    
    X = data[features].values
    
    # 使用标准化器处理数据
    time_scaled = scalers['time'].transform(X[:, :3])
    weather_scaled = scalers['weather'].transform(X[:, 3:4])
    temp_scaled = scalers['temperature'].transform(X[:, 4:7])
    precip_scaled = scalers['precipitation'].transform(X[:, 7:9])
    wind_scaled = scalers['wind'].transform(X[:, 9:10])
    
    # 合并标准化后的数据
    X_scaled = np.column_stack((
        time_scaled,
        weather_scaled,
        temp_scaled,
        precip_scaled,
        wind_scaled
    ))
    
    # 转换为tensor并添加批次维度
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)  # [1, seq_len, n_features]
    
    return X_tensor

def inverse_transform_predictions(pred_temp, pred_precip, pred_wind, scalers):
    """
    将预测结果转换回原始尺度
    
    参数：
        pred_temp: 预测的温度值
        pred_precip: 预测的降水量
        pred_wind: 预测的风速
        scalers: 标准化器字典
        
    返回：
        results: 转换后的预测结果
    """
    # 转换温度
    temp_scaler = scalers['temperature']
    temp_inv = temp_scaler.inverse_transform(pred_temp.reshape(-1, 3))
    
    # 转换降水量
    precip_scaler = scalers['precipitation']
    precip_inv = precip_scaler.inverse_transform(pred_precip.reshape(-1, 2))
    
    # 转换风速
    wind_scaler = scalers['wind']
    wind_inv = wind_scaler.inverse_transform(pred_wind.reshape(-1, 1))
    
    # 组织结果
    results = []
    for i in range(3):  # 3天的预测
        day_result = {
            'temperature_2m_min (°C)': temp_inv[i, 0],
            'temperature_2m_mean (°C)': temp_inv[i, 1],
            'temperature_2m_max (°C)': temp_inv[i, 2],
            'precipitation_sum (mm)': precip_inv[i, 0],
            'precipitation_hours (h)': precip_inv[i, 1],
            'windspeed_10m_max (m/s)': wind_inv[i, 0]
        }
        results.append(day_result)
    
    return results

def predict_next_three_days(input_data_path, model_path='best_model.pth', output_path='predictions.csv'):
    """
    预测未来三天的天气
    
    参数：
        input_data_path: 输入数据文件路径（包含最近7天的数据）
        model_path: 模型文件路径
        output_path: 预测结果保存路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载标准化器
    print("加载标准化器...")
    try:
        scalers = joblib.load('scalers.pkl')
    except FileNotFoundError:
        print("错误：找不到标准化器文件 'scalers.pkl'")
        print("请确保已经运行过训练程序并生成了标准化器文件")
        sys.exit(1)
    
    # 加载模型
    print("加载模型...")
    try:
        model = WeatherForecastModel(input_size=10)  # 10个特征
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{model_path}'")
        print("请确保已经运行过训练程序并生成了模型文件")
        sys.exit(1)
    
    # 加载并预处理输入数据
    print("加载输入数据...")
    try:
        input_data = pd.read_csv(input_data_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误：找不到输入数据文件 '{input_data_path}'")
        print("请确保文件路径正确，例如：'weather_data/Fouryear.csv'")
        sys.exit(1)
        
    if len(input_data) != 7:
        print("错误：输入数据必须包含7天的数据")
        sys.exit(1)
    
    # 预处理数据
    print("预处理数据...")
    input_tensor = preprocess_input_data(input_data, scalers)
    input_tensor = input_tensor.to(device)
    
    # 进行预测
    print("进行预测...")
    with torch.no_grad():
        pred_temp, pred_precip, pred_wind = model(input_tensor)
    
    # 转换预测结果
    print("转换预测结果...")
    results = inverse_transform_predictions(
        pred_temp.cpu().numpy(),
        pred_precip.cpu().numpy(),
        pred_wind.cpu().numpy(),
        scalers
    )
    
    # 打印预测结果
    print("\n未来三天的天气预测：")
    for i, day_result in enumerate(results, 1):
        print(f"\n第{i}天预测：")
        print(f"最低温度: {day_result['temperature_2m_min (°C)']:.1f}°C")
        print(f"平均温度: {day_result['temperature_2m_mean (°C)']:.1f}°C")
        print(f"最高温度: {day_result['temperature_2m_max (°C)']:.1f}°C")
        print(f"总降水量: {day_result['precipitation_sum (mm)']:.1f}mm")
        print(f"降水时长: {day_result['precipitation_hours (h)']:.1f}小时")
        print(f"最大风速: {day_result['windspeed_10m_max (m/s)']:.1f}m/s")
    
    # 保存预测结果到CSV文件
    print(f"\n保存预测结果到文件: {output_path}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print("预测结果已保存")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使用方法: python predict.py <input_data_path>")
        print("例如: python predict.py weather_data/Fouryear.csv")
        sys.exit(1)
    
    input_data_path = sys.argv[1]
    predict_next_three_days(input_data_path) 