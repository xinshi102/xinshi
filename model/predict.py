import os
import sys
import torch
import pandas as pd
import numpy as np
from model.weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
import matplotlib.pyplot as plt
import pickle

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_processor(model_path, processor_path, input_size=36, hidden_size=96, num_layers=1, output_size=72):
    """加载模型和处理器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载处理器
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
        
        # 创建并加载模型
        model = WeatherLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            weather_categories=10
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, processor, device
    except Exception as e:
        print(f"加载模型或处理器时出错: {str(e)}")
        return None, None, None

def make_predictions(duration=72, data_file=None):
    """进行天气预测"""
    # 获取文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 设置模型和处理器路径
    model_path = os.path.join(root_dir, 'data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'data', 'weather_processor.pkl')
    
    # 加载模型和处理器
    model, processor, device = load_model_and_processor(model_path, processor_path)
    if model is None:
        return None
    
    try:
        # 加载和预处理数据
        df = pd.read_csv(data_file)
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']
        df = df.drop('time', axis=1)
        
        # 预处理数据
        processed_data = processor.preprocess_data(df)
        feature_columns = [col for col in processed_data.columns if col not in processor.target_columns]
        
        # 准备输入序列
        if len(processed_data) < 168:
            print("警告：数据长度不足168小时")
            return None
            
        input_sequence = processed_data[feature_columns].values[-168:]
        input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
        
        # 处理预测结果
        weather_codes = torch.argmax(weather_pred, dim=-1).cpu().numpy()[:, :duration]
        weather_codes = processor.inverse_transform_weather(weather_codes.flatten())
        
        # 合并预测结果用于反标准化
        pred_targets = np.column_stack((
            temp_pred.cpu().numpy().flatten()[:duration],
            humidity_pred.cpu().numpy().flatten()[:duration],
            precip_pred.cpu().numpy().flatten()[:duration]
        ))
        
        # 使用处理器的反标准化方法
        pred_targets = processor.inverse_transform_targets(pred_targets)
        temperatures = pred_targets[:, 0]
        humidities = pred_targets[:, 1]
        precipitations = pred_targets[:, 2]
        
        # 生成时间序列
        last_time = df.index[-1]
        times = [(last_time + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                for i in range(duration)]
        
        return times, weather_codes, temperatures, humidities, precipitations
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None

def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'data_w', '1m.csv')
    
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return
        
    predictions = make_predictions(data_file=data_path)
    
    if predictions is not None:
        times, weather_codes, temperatures, humidities, precipitations = predictions
        print("\n预测结果：")
        print(f"天气代码：{weather_codes}")
        print(f"温度：{temperatures}")
        print(f"湿度：{humidities}")
        print(f"降水量：{precipitations}")
        
        # 保存预测图表
        save_dir = os.path.join(current_dir, '..', 'data', 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        
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
        
        print(f"\n预测图表已保存到: {save_dir}")
    else:
        print("预测失败！")

if __name__ == '__main__':
    main() 