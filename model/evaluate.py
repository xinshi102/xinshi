import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
import pickle

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

def evaluate_model(model_path, test_data_path, processor_path):
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
        num_layers=2,
        output_size=72,
        weather_categories=weather_categories
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载测试数据
    test_data = pd.read_csv(test_data_path, parse_dates=['time'], index_col='time')
    print(f"加载测试数据: {len(test_data)} 条记录")

    # 预处理数据
    if not processor.is_fitted:
        processor.fit(test_data)
    processed_data = processor.preprocess_data(test_data)

    feature_columns = [col for col in processed_data.columns if col not in processor.target_columns]
    input_days = 7
    output_days = 3
    input_hours = input_days * 24
    output_hours = output_days * 24

    # 只取一批数据，保证不会越界
    if len(processed_data) < input_hours + output_hours:
        raise ValueError("数据长度不足")
    start_idx = 0
    end_idx = start_idx + input_hours
    input_data = processed_data[feature_columns].values[start_idx:end_idx]
    input_sequence = torch.FloatTensor(input_data).unsqueeze(0).to(device)

    target_start = end_idx
    target_end = target_start + output_hours
    target_data = processed_data[processor.target_columns].values[target_start:target_end]
    target_sequence = torch.FloatTensor(target_data).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
        weather_pred_classes = torch.argmax(weather_pred, dim=-1).cpu().numpy()[0]  # shape: (output_hours,)
        temp_pred = temp_pred.cpu().numpy()[0]
        humidity_pred = humidity_pred.cpu().numpy()[0]
        precip_pred = precip_pred.cpu().numpy()[0]

    # 反标准化
    pred_targets = np.concatenate([temp_pred, humidity_pred, precip_pred], axis=-1)
    pred_targets = processor.inverse_transform_targets(pred_targets)
    # 目标
    target_np = target_sequence.cpu().numpy()[0]  # shape: (output_hours, 4)

    # 调整天气代码
    adjusted_weather_preds = []
    for i in range(len(weather_pred_classes)):
        code = weather_pred_classes[i]
        rain = pred_targets[i, 2]
        cloud = input_sequence[0, i, 4].item()
        adjusted_code = processor.adjust_predicted_weather(code, rain, cloud)
        adjusted_weather_preds.append(adjusted_code)
    adjusted_weather_preds = np.array(adjusted_weather_preds)
    target_weather = target_np[:, 3].astype(int)

    # 评估
    weather_acc = np.mean(adjusted_weather_preds == target_weather)
    temp_rmse = np.sqrt(mean_squared_error(target_np[:, 0], pred_targets[:, 0]))
    temp_mae = mean_absolute_error(target_np[:, 0], pred_targets[:, 0])
    humidity_rmse = np.sqrt(mean_squared_error(target_np[:, 1], pred_targets[:, 1]))
    humidity_mae = mean_absolute_error(target_np[:, 1], pred_targets[:, 1])
    precip_rmse = np.sqrt(mean_squared_error(target_np[:, 2], pred_targets[:, 2]))
    precip_mae = mean_absolute_error(target_np[:, 2], pred_targets[:, 2])
    cm = confusion_matrix(target_weather, adjusted_weather_preds)

    print("\n评估结果:")
    print(f"天气预测准确率: {weather_acc:.4f}")
    print(f"温度预测 RMSE: {temp_rmse:.2f}°C, MAE: {temp_mae:.2f}°C")
    print(f"湿度预测 RMSE: {humidity_rmse:.2f}%, MAE: {humidity_mae:.2f}%")
    print(f"降水量预测 RMSE: {precip_rmse:.2f}mm, MAE: {precip_mae:.2f}mm")
    print(f"混淆矩阵:\n{cm}")

    return {
        'weather_accuracy': weather_acc,
        'temp_rmse': temp_rmse,
        'temp_mae': temp_mae,
        'humidity_rmse': humidity_rmse,
        'humidity_mae': humidity_mae,
        'precip_rmse': precip_rmse,
        'precip_mae': precip_mae,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # 路径可根据实际情况调整
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'data', 'weather_lstm_model.pth')
    processor_path = os.path.join(current_dir, '..', 'data', 'weather_processor.pkl')
    test_data_path = os.path.join(current_dir, '..', 'data', 'data_w', '6m.csv')
    evaluate_model(model_path, test_data_path, processor_path) 