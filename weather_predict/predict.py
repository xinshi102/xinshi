import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import pandas as pd
import numpy as np
from new_models.weather_model import WeatherLSTM
from new_models.weather_processor import WeatherDataProcessor
import matplotlib.pyplot as plt
import pickle
import importlib
import importlib.util

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_model_and_processor(model_path, processor_path, input_size, hidden_size, num_layers, output_size):
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行训练脚本 train.py 来训练模型")
        sys.exit(1)
        
    if not os.path.exists(processor_path):
        print(f"错误：找不到处理器文件 {processor_path}")
        print("请先运行训练脚本 train.py 来训练模型")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载处理器
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # 直接使用10种天气类别
    weather_categories = 10
    
    # 创建模型实例
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size, weather_categories).to(device)
    
    # 加载模型状态
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, processor, device

def make_predictions(duration=72, data_file=None):  # 添加data_file参数
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 检查模型和处理器文件
    model_path = os.path.join(root_dir, 'weather_data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'weather_data', 'weather_processor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(processor_path):
        print(f"错误：找不到模型文件 {model_path} 或处理器文件 {processor_path}")
        return None
    
    # 加载模型
    try:
        model = WeatherLSTM(
            input_size=36,  # 特征数量
            hidden_size=96,  # LSTM隐藏层大小
            num_layers=1,  # LSTM层数
            output_size=72,  # 输出序列长度
            weather_categories=10  # 使用固定的10种天气类别
        )
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("成功加载模型")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None
    
    # 加载处理器
    try:
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
        print("成功加载处理器")
    except Exception as e:
        print(f"加载处理器时出错: {str(e)}")
        return None
    
    # 加载数据
    print("正在加载数据...")
    try:
        df = pd.read_csv(data_file)
        print(f"数据加载成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None
    
    # 设置时间索引
    df['time'] = pd.to_datetime(df['time'])
    df.index = df['time']
    df = df.drop('time', axis=1)
    print(f"数据列名: {df.columns.tolist()}")
    
    # 预处理数据
    print("开始数据预处理...")
    try:
        processed_data = processor.preprocess_data(df)
        print(f"预处理后的数据形状: {processed_data.shape}")
    except Exception as e:
        print(f"数据预处理时出错: {str(e)}")
        return None
    
    # 获取特征列（排除目标列）
    feature_columns = [col for col in processed_data.columns if col not in processor.target_columns]
    print(f"特征列数量: {len(feature_columns)}")
    
    # 检查数据长度
    if len(processed_data) < 168:  # 需要至少168小时的数据
        print("警告：数据长度不足168小时，无法进行预测")
        return None
    
    # 准备输入序列
    print("准备输入序列...")
    input_sequence = processed_data[feature_columns].values[-168:]  # 使用最后168小时的数据
    print(f"输入序列形状: {input_sequence.shape}")
    
    # 确保输入维度正确
    expected_features = 36
    if input_sequence.shape[1] != expected_features:
        print(f"错误：特征数量不匹配，期望 {expected_features}，实际 {input_sequence.shape[1]}")
        return None
    
    # 转换为 PyTorch 张量并移动到设备
    input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
    print(f"输入张量形状: {input_sequence.shape}")
    
    # 进行预测
    print("开始预测...")
    try:
        with torch.no_grad():
            weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        return None
    
    # 获取天气代码预测结果
    weather_codes = torch.argmax(weather_pred, dim=-1).cpu().numpy()[:, :duration]
    
    try:
        # 将预测结果转换回原始尺度
        weather_codes = processor.inverse_transform_weather(weather_codes.flatten())
        
        # 反标准化温度和湿度，并添加范围限制
        temp_mean = df['temperature_2m (°C)'].mean()
        temp_std = df['temperature_2m (°C)'].std()
        humidity_mean = df['relativehumidity_2m (%)'].mean()
        humidity_std = df['relativehumidity_2m (%)'].std()
        precip_mean = df['rain (mm)'].mean()
        precip_std = df['rain (mm)'].std()
        
        # 反标准化并限制范围
        temperatures = temp_pred.cpu().numpy().flatten()[:duration] * temp_std + temp_mean
        
        # 限制湿度在0-100之间
        humidities = humidity_pred.cpu().numpy().flatten()[:duration] * humidity_std + humidity_mean
        humidities = np.clip(humidities, 0, 100)
        
        # 限制降水量为非负数
        precipitations = precip_pred.cpu().numpy().flatten()[:duration] * precip_std + precip_mean
        precipitations = np.maximum(precipitations, 0)
        
        # 生成预测时间序列
        last_time = df.index[-1]
        times = [(last_time + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                for i in range(duration)]
        
        return times, weather_codes, temperatures, humidities, precipitations
    except Exception as e:
        print(f"处理预测结果时出错: {str(e)}")
        return None

def main():
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 设置默认数据文件路径
    data_file = os.path.join(root_dir, 'uploads', '10day.csv')
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        print("请先上传数据文件到 uploads 目录")
        return
    
    # 模型参数
    input_size = 36  # 特征维度
    hidden_size = 96  # 修改为与训练时相同的隐藏层大小
    num_layers = 1  # 修改为与训练时相同的层数
    output_size = 72  # 3天 * 24小时
    
    # 检查模型和处理器文件
    model_path = os.path.join(root_dir, 'weather_data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'weather_data', 'weather_processor.pkl')
    
    # 加载模型和处理器
    model, processor, device = load_model_and_processor(
        model_path, processor_path, input_size, hidden_size, num_layers, output_size
    )
    
    # 进行预测
    predictions = make_predictions(data_file=data_file)
    
    if predictions is not None:
        times, weather_codes, temperatures, humidities, precipitations = predictions
        print("\n预测结果：")
        print(f"时间序列：{times}")
        print(f"天气代码：{weather_codes}")
        print(f"温度：{temperatures}")
        print(f"湿度：{humidities}")
        print(f"降水量：{precipitations}")
        print("预测完成！")
        
        # 创建保存图表的目录
        save_dir = os.path.join(root_dir, 'weather_data', 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制总体预测结果
        plt.figure(figsize=(15, 12))
        
        # 天气代码预测
        plt.subplot(4, 1, 1)
        plt.plot(weather_codes, label='预测天气代码')
        plt.title('未来3天天气代码预测')
        plt.xlabel('时间')
        plt.ylabel('天气代码')
        plt.grid(True)
        plt.legend()
        
        # 温度预测
        plt.subplot(4, 1, 2)
        plt.plot(temperatures, label='预测温度')
        plt.title('未来3天温度预测')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.grid(True)
        plt.legend()
        
        # 湿度预测
        plt.subplot(4, 1, 3)
        plt.plot(humidities, label='预测湿度')
        plt.title('未来3天湿度预测')
        plt.xlabel('时间')
        plt.ylabel('湿度 (%)')
        plt.grid(True)
        plt.legend()
        
        # 降水预测
        plt.subplot(4, 1, 4)
        plt.plot(precipitations, label='预测降水量')
        plt.title('未来3天降水量预测')
        plt.xlabel('时间')
        plt.ylabel('降水量 (mm)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_predictions.png'))
        plt.close()
        
        # 为每一天创建单独的预测图表
        for day in range(3):
            plt.figure(figsize=(15, 12))
            
            # 获取当天的数据
            day_weather = weather_codes[day*24:(day+1)*24]
            day_temp = temperatures[day*24:(day+1)*24]
            day_humidity = humidities[day*24:(day+1)*24]
            day_precip = precipitations[day*24:(day+1)*24]
            
            # 创建时间轴
            hours = range(24)
            
            # 天气代码预测
            plt.subplot(4, 1, 1)
            plt.plot(hours, day_weather, 'k-', label='天气代码')
            plt.title(f'第{day+1}天天气代码预测')
            plt.xlabel('时间 (小时)')
            plt.ylabel('天气代码')
            plt.xticks(range(0, 24, 2))
            plt.grid(True)
            plt.legend()
            
            # 温度预测
            plt.subplot(4, 1, 2)
            plt.plot(hours, day_temp, 'r-', label='温度')
            plt.title(f'第{day+1}天温度预测')
            plt.xlabel('时间 (小时)')
            plt.ylabel('温度 (°C)')
            plt.xticks(range(0, 24, 2))
            plt.grid(True)
            plt.legend()
            
            # 湿度预测
            plt.subplot(4, 1, 3)
            plt.plot(hours, day_humidity, 'b-', label='湿度')
            plt.title(f'第{day+1}天湿度预测')
            plt.xlabel('时间 (小时)')
            plt.ylabel('湿度 (%)')
            plt.xticks(range(0, 24, 2))
            plt.grid(True)
            plt.legend()
            
            # 降水预测
            plt.subplot(4, 1, 4)
            plt.plot(hours, day_precip, 'g-', label='降水量')
            plt.title(f'第{day+1}天降水量预测')
            plt.xlabel('时间 (小时)')
            plt.ylabel('降水量 (mm)')
            plt.xticks(range(0, 24, 2))
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'day_{day+1}_predictions.png'))
            plt.close()
        
        print(f"\n预测图表已保存到: {save_dir}")
    else:
        print("预测失败！")

if __name__ == '__main__':
    print("开始执行预测程序...")
    main() 