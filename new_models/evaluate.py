import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, TensorDataset

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def evaluate_model(model_path, test_data_path, processor_path):
    """评估模型性能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载处理器
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # 加载模型
    weather_categories = 10
    model = WeatherLSTM(
        input_size=36,
        hidden_size=96,
        num_layers=1,
        output_size=72,
        weather_categories=weather_categories
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载测试数据
    test_data = pd.read_csv(test_data_path, parse_dates=['time'], index_col='time')
    
    # 预处理数据
    processed_data = processor.preprocess_data(test_data)
    
    # 创建序列
    sequences, targets = processor.create_sequences(processed_data, input_days=7, output_days=3)
    
    # 创建数据加载器
    test_dataset = TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(targets))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化评估指标
    weather_correct = 0
    weather_total = 0
    temp_errors = []
    humidity_errors = []
    precip_errors = []
    
    # 用于存储预测结果和实际值
    all_weather_preds = []
    all_weather_targets = []
    all_temp_preds = []
    all_temp_targets = []
    all_humidity_preds = []
    all_humidity_targets = []
    all_precip_preds = []
    all_precip_targets = []
    
    print("开始评估...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 获取预测结果
            weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
            
            # 处理天气预测
            weather_pred_classes = torch.argmax(weather_pred, dim=-1)
            weather_targets = targets[:, :, 3].long()
            
            # 累积正确预测数
            weather_correct += (weather_pred_classes == weather_targets).sum().item()
            weather_total += weather_targets.numel()
            
            # 收集预测结果和目标值
            all_weather_preds.extend(weather_pred_classes.cpu().numpy().flatten())
            all_weather_targets.extend(weather_targets.cpu().numpy().flatten())
            all_temp_preds.extend(temp_pred.cpu().numpy().flatten())
            all_temp_targets.extend(targets[:, :, 0].cpu().numpy().flatten())
            all_humidity_preds.extend(humidity_pred.cpu().numpy().flatten())
            all_humidity_targets.extend(targets[:, :, 1].cpu().numpy().flatten())
            all_precip_preds.extend(precip_pred.cpu().numpy().flatten())
            all_precip_targets.extend(targets[:, :, 2].cpu().numpy().flatten())
    
    # 计算评估指标
    weather_accuracy = weather_correct / weather_total
    temp_mse = mean_squared_error(all_temp_targets, all_temp_preds)
    temp_mae = mean_absolute_error(all_temp_targets, all_temp_preds)
    humidity_mse = mean_squared_error(all_humidity_targets, all_humidity_preds)
    humidity_mae = mean_absolute_error(all_humidity_targets, all_humidity_preds)
    precip_mse = mean_squared_error(all_precip_targets, all_precip_preds)
    precip_mae = mean_absolute_error(all_precip_targets, all_precip_preds)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"天气预测准确率: {weather_accuracy:.4f}")
    print(f"温度预测 MSE: {temp_mse:.4f}, MAE: {temp_mae:.4f}")
    print(f"湿度预测 MSE: {humidity_mse:.4f}, MAE: {humidity_mae:.4f}")
    print(f"降水预测 MSE: {precip_mse:.4f}, MAE: {precip_mae:.4f}")
    
    # 绘制评估图表
    save_dir = os.path.join(os.path.dirname(model_path), 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制天气预测混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.hist2d(all_weather_targets, all_weather_preds, bins=(10, 10))
    plt.colorbar()
    plt.xlabel('实际天气代码')
    plt.ylabel('预测天气代码')
    plt.title('天气预测混淆矩阵')
    plt.savefig(os.path.join(save_dir, 'weather_confusion.png'))
    plt.close()
    
    # 绘制温度预测散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_temp_targets, all_temp_preds, alpha=0.5)
    plt.plot([min(all_temp_targets), max(all_temp_targets)], 
             [min(all_temp_targets), max(all_temp_targets)], 'r--')
    plt.xlabel('实际温度')
    plt.ylabel('预测温度')
    plt.title('温度预测散点图')
    plt.savefig(os.path.join(save_dir, 'temperature_scatter.png'))
    plt.close()
    
    # 绘制湿度预测散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_humidity_targets, all_humidity_preds, alpha=0.5)
    plt.plot([min(all_humidity_targets), max(all_humidity_targets)], 
             [min(all_humidity_targets), max(all_humidity_targets)], 'r--')
    plt.xlabel('实际湿度')
    plt.ylabel('预测湿度')
    plt.title('湿度预测散点图')
    plt.savefig(os.path.join(save_dir, 'humidity_scatter.png'))
    plt.close()
    
    # 绘制降水预测散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_precip_targets, all_precip_preds, alpha=0.5)
    plt.plot([min(all_precip_targets), max(all_precip_targets)], 
             [min(all_precip_targets), max(all_precip_targets)], 'r--')
    plt.xlabel('实际降水量')
    plt.ylabel('预测降水量')
    plt.title('降水预测散点图')
    plt.savefig(os.path.join(save_dir, 'precipitation_scatter.png'))
    plt.close()
    
    return {
        'weather_accuracy': weather_accuracy,
        'temp_mse': temp_mse,
        'temp_mae': temp_mae,
        'humidity_mse': humidity_mse,
        'humidity_mae': humidity_mae,
        'precip_mse': precip_mse,
        'precip_mae': precip_mae
    }

def main():
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建文件路径
    model_path = os.path.join(current_dir, '..', 'weather_data', 'weather_lstm_model.pth')
    processor_path = os.path.join(current_dir, '..', 'weather_data', 'weather_processor.pkl')
    test_data_path = os.path.join(current_dir, '..', 'weather_data', 'data_w', '10day.csv')  # 使用10天的数据作为测试集
    
    # 检查文件是否存在
    for path in [model_path, processor_path, test_data_path]:
        if not os.path.exists(path):
            print(f"错误：找不到文件 {path}")
            sys.exit(1)
    
    # 评估模型
    results = evaluate_model(model_path, test_data_path, processor_path)
    
    # 保存评估结果
    results_path = os.path.join(current_dir, '..', 'weather_data', 'evaluation_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"天气预测准确率: {results['weather_accuracy']:.4f}\n")
        f.write(f"温度预测 MSE: {results['temp_mse']:.4f}\n")
        f.write(f"温度预测 MAE: {results['temp_mae']:.4f}\n")
        f.write(f"湿度预测 MSE: {results['humidity_mse']:.4f}\n")
        f.write(f"湿度预测 MAE: {results['humidity_mae']:.4f}\n")
        f.write(f"降水预测 MSE: {results['precip_mse']:.4f}\n")
        f.write(f"降水预测 MAE: {results['precip_mae']:.4f}\n")
    
    print(f"\n评估结果已保存到: {results_path}")

if __name__ == '__main__':
    main() 