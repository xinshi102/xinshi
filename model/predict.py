import os
import sys
import torch
import pandas as pd
import numpy as np
from model.weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
<<<<<<< HEAD
import pickle
import traceback

sys.stdout.reconfigure(encoding='utf-8')
# 保证当前目录在Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_processor(model_path, processor_path, input_size=36, hidden_size=128, num_layers=2, output_size=72):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # 加载数据处理器
        with open(processor_path, 'rb') as f:#rb二进制读模式
            processor = pickle.load(f)
        # 创建并加载模型参数
        model = WeatherLSTM(#动态构建lstm实例
=======
import matplotlib.pyplot as plt
import pickle

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_processor(model_path, processor_path, input_size=36, hidden_size=128, num_layers=2, output_size=72):
    """加载模型和处理器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载处理器
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
        
        # 创建并加载模型
        model = WeatherLSTM(
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            weather_categories=10
        ).to(device)
<<<<<<< HEAD
        model.load_state_dict(torch.load(model_path, map_location=device))#加载模型参数，确保模型参数运行在正确的设备上
        model.eval()#关闭随机性，关闭了dropout和batchnorm
=======
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        return model, processor, device
    except Exception as e:
        print(f"加载模型或处理器时出错: {str(e)}")
        return None, None, None

<<<<<<< HEAD

def make_predictions(duration=72, data_file=None):#默认72小时预测
    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
    root_dir = os.path.dirname(current_dir)
    # 设置模型和处理器路径
    model_path = os.path.join(root_dir, 'data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'data', 'weather_processor.pkl')
=======
def make_predictions(duration=72, data_file=None):
    """进行天气预测"""
    # 获取文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 设置模型和处理器路径
    model_path = os.path.join(root_dir, 'data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'data', 'weather_processor.pkl')
    
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    # 加载模型和处理器
    model, processor, device = load_model_and_processor(model_path, processor_path)
    if model is None:
        return None
<<<<<<< HEAD
    try:
        # 加载并预处理数据
        df = pd.read_csv(data_file)
        df['time'] = pd.to_datetime(df['time'])#时间列处理
        df.index = df['time']#设置为索引
        df = df.drop('time', axis=1)#移除原始时间列
        # 预处理
        processed_data = processor.preprocess_data(df)#创建时间特征，滚动特征等
        feature_columns = [col for col in processed_data.columns if col not in processor.target_columns]#代码一致性
        # 检查数据长度
        if len(processed_data) < 168:
            print("警告：数据长度不足168小时")
            return None
        # 构造输入序列
        input_sequence = processed_data[feature_columns].values[-168:]#取最后168个小时的数据
        input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)#转换成张量并且移动到cpu上
        # 预测
        with torch.no_grad():
            weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
        
        # 处理预测结果 - 增强健壮性
        # 提取并验证天气索引
        weather_indices = torch.argmax(weather_pred, dim=-1).cpu().numpy()[:, :duration]#最后一个维度取最大值索引也就是概率最大的天气代码
        weather_indices = weather_indices.flatten()#从2维数组转换成1维数组方便处理
        # 安全转换索引到天气代码
        valid_weather_codes = []
        for idx in weather_indices:
            try:
                # 确保索引在有效范围内
                if idx >= 0 and idx in processor.index_to_weather_code:#索引转天气代码
                    valid_weather_codes.append(processor.index_to_weather_code[idx])
                else:
                    # 默认使用晴天代码(0)作为后备
                    valid_weather_codes.append(0)
            except Exception as e:
                print(f"处理天气代码时出错: {str(e)}, 索引: {idx}")
                valid_weather_codes.append(0)  # 使用默认值
        
        weather_codes = np.array(valid_weather_codes)
        
        # 合并预测结果用于反标准化
        pred_targets = np.column_stack((
            temp_pred.cpu().numpy().flatten()[:duration],#都将张量转成numpy数组，并且拉平成1维数组
=======
    
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
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            humidity_pred.cpu().numpy().flatten()[:duration],
            precip_pred.cpu().numpy().flatten()[:duration]
        ))
        
<<<<<<< HEAD
        # 反标准化
=======
        # 使用处理器的反标准化方法
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        pred_targets = processor.inverse_transform_targets(pred_targets)
        temperatures = pred_targets[:, 0]
        humidities = pred_targets[:, 1]
        precipitations = pred_targets[:, 2]
        
<<<<<<< HEAD
        # 生成预测时间序列
        last_time = df.index[-1]
        times = [(last_time + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                for i in range(duration)]

        # 验证预测长度和数量
        print(f"验证预测结果: 时间数量={len(times)}, 天气代码数量={len(weather_codes)}, "
              f"温度数量={len(temperatures)}, 湿度数量={len(humidities)}, 降水量数量={len(precipitations)}")

        # 修复：保证所有输出长度与duration一致，并且不包含无效值
        def safe_pad_truncate(arr, length, default_value=0):
            # 转换为列表
            arr = list(arr)
            
            # 替换无效值
            for i in range(len(arr)):
                if arr[i] is None or (isinstance(arr[i], float) and np.isnan(arr[i])):
                    arr[i] = default_value
            
            # 长度调整
            if len(arr) < length:
                # 如果有效元素，用最后一个有效元素补齐
                if arr:
                    arr += [arr[-1]] * (length - len(arr))
                else:
                    arr = [default_value] * length
            return arr[:length]

        # 使用安全版本确保输出正确
        times = safe_pad_truncate(times, duration, default_value=times[-1] if times else None)
        weather_codes = safe_pad_truncate(weather_codes, duration, default_value=0)
        temperatures = safe_pad_truncate(temperatures, duration, default_value=25.0)  # 使用合理的默认温度
        humidities = safe_pad_truncate(humidities, duration, default_value=50.0)      # 使用合理的默认湿度
        precipitations = safe_pad_truncate(precipitations, duration, default_value=0.0)
        
        print(f"返回预测结果: 时间数量={len(times)}, 天气代码数量={len(weather_codes)}, 其中天气代码={weather_codes[:5]}...")
        
        return times, weather_codes, temperatures, humidities, precipitations
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        print(traceback.format_exc())  # 打印完整的错误堆栈
        return None


def main():
    """
    测试主函数：本地运行预测并打印结果
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'data_w', '10day.csv')
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return
    predictions = make_predictions(data_file=data_path)
=======
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
    
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    if predictions is not None:
        times, weather_codes, temperatures, humidities, precipitations = predictions
        print("\n预测结果：")
        print(f"天气代码：{weather_codes}")
        print(f"温度：{temperatures}")
        print(f"湿度：{humidities}")
        print(f"降水量：{precipitations}")
<<<<<<< HEAD
    else:
        print("预测失败！")


=======
        
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

>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
if __name__ == '__main__':
    main() 