import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

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
    print(f"加载测试数据: {len(test_data)} 条记录")
    
    # 确保数据处理器已经拟合
    if not processor.is_fitted:
        print("数据处理器尚未拟合，正在拟合...")
        processor.fit(test_data)
    
    # 预处理数据
    processed_data = processor.preprocess_data(test_data)
    print(f"预处理后的数据形状: {processed_data.shape}")
    
    # 获取特征列（排除目标列）
    feature_columns = [col for col in processed_data.columns if col not in processor.target_columns]
    print(f"输入特征数量: {len(feature_columns)}")
    
    # 创建序列
    input_days = 7
    output_days = 3
    input_hours = input_days * 24
    output_hours = output_days * 24
    
    print(f"可用数据长度: {len(processed_data)} 小时")
    
    if len(processed_data) < input_hours:
        raise ValueError(f"数据长度不足，至少需要{input_hours}小时的数据")
    
    # 准备输入数据
    input_data = processed_data[feature_columns].values[-input_hours:]
    input_sequence = torch.FloatTensor(input_data).unsqueeze(0).to(device)
    
    # 准备目标数据
    target_data = processed_data[processor.target_columns].values[-output_hours:]
    target_sequence = torch.FloatTensor(target_data).unsqueeze(0).to(device)
    
    print(f"输入数据形状: {input_sequence.shape}")
    print(f"目标数据形状: {target_sequence.shape}")
    
    # 初始化评估指标
    weather_correct = 0
    weather_total = 0
    
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
        # 获取预测结果
        weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
        
        # 处理天气预测
        weather_pred_classes = torch.argmax(weather_pred, dim=-1)
        weather_targets = target_sequence[:, :, 3].long()
        
        # 累积正确预测数
        weather_correct += (weather_pred_classes == weather_targets).sum().item()
        weather_total += weather_targets.numel()
        
        # 收集预测结果和目标值
        all_weather_preds.extend(weather_pred_classes.cpu().numpy().flatten())
        all_weather_targets.extend(weather_targets.cpu().numpy().flatten())
        all_temp_preds.extend(temp_pred.cpu().numpy().flatten())
        all_temp_targets.extend(target_sequence[:, :, 0].cpu().numpy().flatten())
        all_humidity_preds.extend(humidity_pred.cpu().numpy().flatten())
        all_humidity_targets.extend(target_sequence[:, :, 1].cpu().numpy().flatten())
        all_precip_preds.extend(precip_pred.cpu().numpy().flatten())
        all_precip_targets.extend(target_sequence[:, :, 2].cpu().numpy().flatten())
    
    print(f"天气预测总数: {weather_total}")
    print(f"天气预测正确数: {weather_correct}")
    
    if weather_total == 0:
        raise ValueError("没有成功处理任何预测数据")
    
    # 计算评估指标
    weather_accuracy = weather_correct / weather_total
    temp_mse = mean_squared_error(all_temp_targets, all_temp_preds)
    temp_mae = mean_absolute_error(all_temp_targets, all_temp_preds)
    temp_rmse = np.sqrt(temp_mse)
    humidity_mse = mean_squared_error(all_humidity_targets, all_humidity_preds)
    humidity_mae = mean_absolute_error(all_humidity_targets, all_humidity_preds)
    humidity_rmse = np.sqrt(humidity_mse)
    precip_mse = mean_squared_error(all_precip_targets, all_precip_preds)
    precip_mae = mean_absolute_error(all_precip_targets, all_precip_preds)
    precip_rmse = np.sqrt(precip_mse)
    
    # 计算天气预测的混淆矩阵
    weather_confusion = confusion_matrix(all_weather_targets, all_weather_preds)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"天气预测准确率: {weather_accuracy:.4f}")
    print(f"温度预测 MSE: {temp_mse:.4f}, MAE: {temp_mae:.4f}, RMSE: {temp_rmse:.4f}")
    print(f"湿度预测 MSE: {humidity_mse:.4f}, MAE: {humidity_mae:.4f}, RMSE: {humidity_rmse:.4f}")
    print(f"降水预测 MSE: {precip_mse:.4f}, MAE: {precip_mae:.4f}, RMSE: {precip_rmse:.4f}")
    
    return {
        'weather_accuracy': weather_accuracy,
        'temp_mse': temp_mse,
        'temp_mae': temp_mae,
        'temp_rmse': temp_rmse,
        'humidity_mse': humidity_mse,
        'humidity_mae': humidity_mae,
        'humidity_rmse': humidity_rmse,
        'precip_mse': precip_mse,
        'precip_mae': precip_mae,
        'precip_rmse': precip_rmse,
        'weather_confusion': weather_confusion
    }

def main():
    try:
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        
        # 构建文件路径
        model_path = os.path.join(current_dir, '..', 'data', 'weather_lstm_model.pth')
        processor_path = os.path.join(current_dir, '..', 'data', 'weather_processor.pkl')
        test_data_path = os.path.join(current_dir, '..', 'data', 'data_w', '3m.csv')
        
        print(f"模型路径: {model_path}")
        print(f"处理器路径: {processor_path}")
        print(f"测试数据路径: {test_data_path}")
        
        # 检查文件是否存在
        for path in [model_path, processor_path, test_data_path]:
            if not os.path.exists(path):
                print(f"错误：找不到文件 {path}")
                sys.exit(1)
            else:
                print(f"文件存在: {path}")
        
        # 评估模型
        print("\n开始评估模型...")
        results = evaluate_model(model_path, test_data_path, processor_path)
        
        # 保存评估结果
        results_path = os.path.join(current_dir, '..', 'data', 'evaluation_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"天气预测准确率: {results['weather_accuracy']:.4f}\n\n")
            f.write("温度预测指标:\n")
            f.write(f"  MSE: {results['temp_mse']:.4f}\n")
            f.write(f"  MAE: {results['temp_mae']:.4f}\n")
            f.write(f"  RMSE: {results['temp_rmse']:.4f}\n\n")
            f.write("湿度预测指标:\n")
            f.write(f"  MSE: {results['humidity_mse']:.4f}\n")
            f.write(f"  MAE: {results['humidity_mae']:.4f}\n")
            f.write(f"  RMSE: {results['humidity_rmse']:.4f}\n\n")
            f.write("降水预测指标:\n")
            f.write(f"  MSE: {results['precip_mse']:.4f}\n")
            f.write(f"  MAE: {results['precip_mae']:.4f}\n")
            f.write(f"  RMSE: {results['precip_rmse']:.4f}\n")
        
        print(f"\n评估结果已保存到: {results_path}")
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 