import os
import sys
import torch
import pandas as pd
import numpy as np
from model.weather_model import WeatherLSTM
from weather_processor import WeatherDataProcessor
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
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            weather_categories=10
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))#加载模型参数，确保模型参数运行在正确的设备上
        model.eval()#关闭随机性，关闭了dropout和batchnorm
        return model, processor, device
    except Exception as e:
        print(f"加载模型或处理器时出错: {str(e)}")
        return None, None, None


def make_predictions(duration=72, data_file=None):#默认72小时预测
    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
    root_dir = os.path.dirname(current_dir)
    # 设置模型和处理器路径
    model_path = os.path.join(root_dir, 'data', 'weather_lstm_model.pth')
    processor_path = os.path.join(root_dir, 'data', 'weather_processor.pkl')
    # 加载模型和处理器
    model, processor, device = load_model_and_processor(model_path, processor_path)
    if model is None:
        return None
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
            humidity_pred.cpu().numpy().flatten()[:duration],
            precip_pred.cpu().numpy().flatten()[:duration]
        ))
        
        # 反标准化
        pred_targets = processor.inverse_transform_targets(pred_targets)
        temperatures = pred_targets[:, 0]
        humidities = pred_targets[:, 1]
        precipitations = pred_targets[:, 2]
        
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
    if predictions is not None:
        times, weather_codes, temperatures, humidities, precipitations = predictions
        print("\n预测结果：")
        print(f"天气代码：{weather_codes}")
        print(f"温度：{temperatures}")
        print(f"湿度：{humidities}")
        print(f"降水量：{precipitations}")
    else:
        print("预测失败！")


if __name__ == '__main__':
    main() 