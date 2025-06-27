from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback
import torch
import json
import math
import shutil
from typing import Any, Dict, List, Optional
import requests

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 添加weather_processor模块到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))

from model.predict import make_predictions
from weather_processor import WeatherDataProcessor

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})

# ========== 全局常量与初始化 ==========
UPLOAD_FOLDER = os.path.join(root_dir, 'uploads')
OLD_FILES_FOLDER = os.path.join(root_dir, 'old_files')
for folder in [UPLOAD_FOLDER, OLD_FILES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 天气代码映射
WEATHER_CODES = {
    0: "晴天",
    1: "多云",
    2: "阴天",
    3: "阴天多云",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨"
}

# 初始化天气处理器
try:
    weather_processor = WeatherDataProcessor()
    print("成功初始化天气处理器")
except Exception as e:
    print(f"初始化天气处理器时出错: {str(e)}")
    print(traceback.format_exc())

# ========== 工具函数 ==========
def move_old_files() -> None:
    """将旧文件移动到old_files目录"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.csv'):
                old_path = os.path.join(UPLOAD_FOLDER, filename)
                new_filename = f"{timestamp}_{filename}"
                new_path = os.path.join(OLD_FILES_FOLDER, new_filename)
                shutil.move(old_path, new_path)
                print(f"已移动文件: {filename} -> {new_filename}")
    except Exception as e:
        print(f"移动旧文件时出错: {str(e)}")

def replace_nan(obj: Any) -> Any:
    """递归将NaN替换为None"""
    if isinstance(obj, float) and math.isnan(obj):
        return None#判断对象是否为浮点数，是则检查是不是NAN，如果是NAN则返回none
    elif isinstance(obj, list):
        return [replace_nan(x) for x in obj]
    elif isinstance(obj, dict):#如果对象是字典使用字典推导式递归处理字典的每个值并返回新字典
        return {k: replace_nan(v) for k, v in obj.items()}
    else:
        return obj

def to_python_type(obj: Any) -> Any:
    """递归将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()#使用item方法将Numpy的通用类型转换成python原生类型
    else:
        return obj

# ========== 路由与API实现 ==========

@app.route('/api/upload', methods=['POST'])
def upload_file() -> Any:
    """文件上传接口，校验并保存CSV，返回原始数据预览"""
    try:
        print("接收到文件上传请求")
        if 'file' not in request.files:
            print("错误：请求中没有文件")
            return jsonify({'error': '没有文件上传'}), 400
        file = request.files['file']
        if file.filename == '':#用户有时可能点击上传文件但是没有选择文件，所以这时文件是空的。
            print("错误：没有选择文件")
            return jsonify({'error': '没有选择文件'}), 400
        print(f"上传的文件名: {file.filename}")
        move_old_files()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"文件已保存到: {file_path}")
        print("正在读取CSV文件...")
        df = pd.read_csv(file_path)
        print(f"CSV文件列名: {df.columns.tolist()}")#返回包含所有列名的普通python列表
        required_columns = [
            'time', 'temperature_2m (°C)', 'relativehumidity_2m (%)',
            'rain (mm)', 'surface_pressure (hPa)', 'cloudcover (%)',
            'windspeed_10m (m/s)', 'weathercode (wmo code)'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"CSV文件缺少必需的列: {', '.join(missing_columns)}"
            print(f"错误：{error_msg}")
            os.remove(file_path)
            return jsonify({'error': error_msg}), 400
        response_data = {
            'times': df['time'].tolist(),
            'temperatures': df['temperature_2m (°C)'].tolist(),
            'humidities': df['relativehumidity_2m (%)'].tolist(),
            'precipitations': df['rain (mm)'].tolist()
        }
        print("数据处理成功，准备返回响应")
        return jsonify(response_data)
    except pd.errors.EmptyDataError:
        error_msg = "上传的CSV文件是空的"
        print(f"错误：{error_msg}")
        if os.path.exists(file_path):
            os.remove(file_path)#删除空文件防止占用
        return jsonify({'error': error_msg}), 400
    except pd.errors.ParserError:
        error_msg = "CSV文件格式错误，请检查文件格式"
        print(f"错误：{error_msg}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"处理文件时出错: {str(e)}"
        print(f"错误：{error_msg}")
        print("详细错误信息:")
        print(traceback.format_exc())
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': error_msg}), 500#500表示服务器处理请求遇到无法处理的错误

@app.route('/api/predict', methods=['GET', 'POST'])
def predict() -> Any:
    """
    天气预测接口。
    - GET/POST: duration参数指定预测时长（小时），默认72小时。
    - 返回：预测的时间序列、天气代码、温度、湿度、降水量等。
    """
    try:
        print("\n=== 开始预测过程 ===")
        # 获取预测时长参数
        if request.method == 'POST':
            data = request.get_json()
            duration = data.get('duration', 72) if data else 72
        else:
            duration = int(request.args.get('duration', 72))
        print(f"预测时长: {duration}小时")
        
        # 检查上传目录中是否有文件
        if not os.path.exists(UPLOAD_FOLDER):
            error_msg = "上传目录不存在"
            print(f"错误：{error_msg}")
            return jsonify({'error': error_msg}), 500
            
        # 获取最新上传的文件
        files = os.listdir(UPLOAD_FOLDER)
        if not files:
            error_msg = "没有找到上传的文件，请先上传数据文件"
            print(f"错误：{error_msg}")
            return jsonify({'error': error_msg}), 400
            
        latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in files], key=os.path.getctime)
        print(f"使用最新上传的文件: {latest_file}")
        
        # 检查文件大小
        file_size = os.path.getsize(latest_file)
        print(f"文件大小: {file_size} 字节")
        if file_size == 0:
            error_msg = "上传的文件是空的"
            print(f"错误：{error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # 检查文件内容
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"文件首行: {first_line}")
                if not first_line:
                    error_msg = "文件是空的或格式不正确"
                    print(f"错误：{error_msg}")
                    return jsonify({'error': error_msg}), 400
        except Exception as e:
            error_msg = f"读取文件时出错: {str(e)}"
            print(f"错误：{error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # 检查CUDA是否可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 调用预测函数，传入文件路径
        print("\n=== 调用预测函数 ===")
        predictions = make_predictions(duration=duration, data_file=latest_file)
        
        if predictions is None:
            error_msg = "预测失败，模型返回空结果"
            print(f"错误：{error_msg}")
            return jsonify({'error': error_msg}), 500
            
        times, weather_codes, temperatures, humidities, precipitations = predictions
        print("\n预测结果概览:")
        print(f"时间序列数量: {len(times)}")
        print(f"天气代码数量: {len(weather_codes)}")
        print(f"温度数据数量: {len(temperatures)}")
        print(f"湿度数据数量: {len(humidities)}")
        print(f"降水量数据数量: {len(precipitations)}")
        
        # 获取天气描述
        try:
            weather_descriptions = [weather_processor.get_weather_description(code) for code in weather_codes]
            print(f"生成了 {len(weather_descriptions)} 个天气描述")
        except Exception as e:
            print(f"警告：生成天气描述时出错: {str(e)}")
            weather_descriptions = ["未知天气" for _ in weather_codes]

        # 生成unique_weather_codes列表
        unique_codes = sorted(set(weather_codes))
        unique_weather_codes = [
            {'code': int(code), 'desc': weather_processor.get_weather_description(code)}
            for code in unique_codes
        ]

        # 准备响应数据，确保所有字段为list且不为None
        response_data = {
            'times': list(times) if times is not None else [],
            'weather_codes': list(weather_codes) if weather_codes is not None else [],
            'weather_descriptions': list(weather_descriptions) if weather_descriptions is not None else [],
            'temperatures': list(temperatures) if temperatures is not None else [],
            'humidities': list(humidities) if humidities is not None else [],
            'precipitations': list(precipitations) if precipitations is not None else [],
            'unique_weather_codes': unique_weather_codes
        }
        # 处理NaN值
        response_data = replace_nan(response_data)
        # 递归转换所有numpy类型为Python原生类型
        response_data = to_python_type(response_data)
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        print(f"错误：{error_msg}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/weather-codes', methods=['GET'])
def get_weather_codes() -> Any:
    """返回天气代码与描述的映射表。"""
    return jsonify(WEATHER_CODES)

@app.route('/api/analysis-data', methods=['GET'])
def get_analysis_data() -> Any:
    """
    数据分析接口。
    - 返回：温度、湿度趋势，降水量分布，天气类型分布，云量/风速分布，特征相关性热力图等。
    """
    try:
        # 获取时间单位参数
        time_unit = request.args.get('time_unit', 'hour')
        
        # 获取最新上传的文件
        upload_dir = os.path.join(root_dir, 'uploads')
        if not os.path.exists(upload_dir):
            return jsonify({'error': '请先上传数据文件'}), 400
        files = os.listdir(upload_dir)
        if not files:
            return jsonify({'error': '请先上传数据文件'}), 400
        latest_file = max([os.path.join(upload_dir, f) for f in files], key=os.path.getctime)
        
        # 读取分析数据
        df = pd.read_csv(latest_file, parse_dates=['time'], index_col='time')
        
        # 根据时间单位进行数据聚合
        if time_unit == 'day':
            # 按天聚合数据
            df = df.resample('D').agg({
                'temperature_2m (°C)': 'mean',
                'relativehumidity_2m (%)': 'mean',
                'rain (mm)': 'sum',
                'surface_pressure (hPa)': 'mean',
                'cloudcover (%)': 'mean',
                'windspeed_10m (m/s)': 'mean',
                'weathercode (wmo code)': lambda x: x.mode()[0] if not x.empty else None
            })
        
        # 降水量分布自定义分箱，保证区间严格递增
        rain_bins = [0, 0.1, 1, 5, 10]
        max_rain = float(df['rain (mm)'].max())
        if max_rain + 1 > rain_bins[-1]:
            rain_bins.append(max_rain + 1)
        rain_bins = sorted(set(rain_bins))  # 去重并排序，确保递增
        rain_labels = [f"{rain_bins[i]}~{rain_bins[i+1]}" for i in range(len(rain_bins)-1)]
        rain_cut = pd.cut(df['rain (mm)'], bins=rain_bins, labels=rain_labels, right=False, include_lowest=True)
        rain_counts = rain_cut.value_counts(sort=False).tolist()
        
        # 新增二值化降水量 rain_flag
        df['rain_flag'] = (df['rain (mm)'] > 0).astype(int)
        
        # 生成weather_codes列表
        weather_codes_list = [
            {'code': int(code), 'description': WEATHER_CODES.get(code, f'未知({code})')}
            for code in df['weathercode (wmo code)'].unique()
        ]
        
        # 生成weather_distribution列表
        weather_distribution_list = [
            {
                'code': int(code),
                'description': WEATHER_CODES.get(code, f'未知({code})'),
                'count': int(df['weathercode (wmo code)'].value_counts()[code])
            }
            for code in df['weathercode (wmo code)'].unique()
        ]
        
        # 准备分析数据，字段名全部下划线风格
        analysis_data = {
            'temperature_trend': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['temperature_2m (°C)'].tolist()
            },
            'humidity_trend': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['relativehumidity_2m (%)'].tolist()
            },
            'precipitation_trend': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['rain (mm)'].tolist()
            },
            'precipitation_distribution': {
                'categories': rain_labels,
                'values': rain_counts
            },
            'weather_distribution': weather_distribution_list,
            'cloud_cover_distribution': {
                'categories': np.histogram(df['cloudcover (%)'], bins=50)[1].tolist()[:-1],
                'values': np.histogram(df['cloudcover (%)'], bins=50)[0].tolist()
            },
            'wind_speed_distribution': {
                'categories': np.histogram(df['windspeed_10m (m/s)'], bins=50)[1].tolist()[:-1],
                'values': np.histogram(df['windspeed_10m (m/s)'], bins=50)[0].tolist()
            },
            'correlation_heatmap': {
                'features': ['温度', '湿度', '降水(有无)', '气压', '云量', '风速'],
                'correlations': df[['temperature_2m (°C)', 'relativehumidity_2m (%)', 'rain_flag',
                                    'surface_pressure (hPa)', 'cloudcover (%)', 'windspeed_10m (m/s)']].corr().values.tolist()
            },
            'weather_codes': weather_codes_list
        }
        
        # 处理NaN值
        analysis_data = replace_nan(analysis_data)
        return jsonify(analysis_data)
    except Exception as e:
        error_msg = f"获取分析数据时出错: {str(e)}"
        print(f"错误：{error_msg}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/current-weather', methods=['GET'])
def get_current_weather() -> Any:
    """返回最新一条天气代码。"""
    try:
        upload_dir = os.path.join(root_dir, 'uploads')
        files = os.listdir(upload_dir)
        if not files:
            return jsonify({'error': '请先上传数据文件'}), 400
        latest_file = max([os.path.join(upload_dir, f) for f in files], key=os.path.getctime)#max是返回最大的元素，key是创建时间，也就是返回最新的元素
        df = pd.read_csv(latest_file)
        weather_code = int(df['weathercode (wmo code)'].iloc[-1])
        return jsonify({'weather_code': weather_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

DEEPSEEK_API_KEY = "sk-068f2a733db941c5a4581fb53ac97ece"  # 用户提供的DeepSeek API密钥

def summarize_prediction_for_ai(prediction):
    """
    将详细的预测数据转换为简洁的自然语言摘要，用于AI分析
    """
    try:
        if not prediction or not isinstance(prediction, dict):
            return "无有效预测数据"
        
        times = prediction.get('times', [])
        temperatures = prediction.get('temperatures', [])
        humidities = prediction.get('humidities', [])
        precipitations = prediction.get('precipitations', [])
        weather_descriptions = prediction.get('weather_descriptions', [])
        
        if not all([times, temperatures, humidities, precipitations]):
            return "预测数据不完整"
        
        # 数据长度
        data_length = len(times)
        if data_length == 0:
            return "无预测数据"
        
        # 时间范围
        start_time = times[0] if times else "未知"
        end_time = times[-1] if times else "未知"
        
        # 温度统计
        temp_min = min(temperatures)
        temp_max = max(temperatures)
        temp_avg = sum(temperatures) / len(temperatures)
        
        # 湿度统计
        humidity_min = min(humidities)
        humidity_max = max(humidities)
        humidity_avg = sum(humidities) / len(humidities)
        
        # 降水统计
        total_precipitation = sum(precipitations)
        rain_hours = sum(1 for p in precipitations if p > 0)
        max_hourly_rain = max(precipitations) if precipitations else 0
        
        # 天气类型统计
        weather_types = {}
        for desc in weather_descriptions:
            weather_types[desc] = weather_types.get(desc, 0) + 1
        main_weather = max(weather_types.items(), key=lambda x: x[1])[0] if weather_types else "未知"
        
        # 按时间段分析（将数据分为3段：前24小时、中间24小时、后24小时）
        periods = []
        segment_size = max(1, data_length // 3)
        
        for i, period_name in enumerate(['近期(1-24小时)', '中期(25-48小时)', '远期(49-72小时)']):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, data_length)
            
            if start_idx < data_length:
                period_temps = temperatures[start_idx:end_idx]
                period_precips = precipitations[start_idx:end_idx]
                period_weathers = weather_descriptions[start_idx:end_idx]
                
                if period_temps and period_precips and period_weathers:
                    period_temp_avg = sum(period_temps) / len(period_temps)
                    period_total_rain = sum(period_precips)
                    period_main_weather = max(set(period_weathers), key=period_weathers.count)
                    
                    periods.append({
                        'name': period_name,
                        'temp_avg': round(period_temp_avg, 1),
                        'total_rain': round(period_total_rain, 1),
                        'main_weather': period_main_weather
                    })
        
        # 构造简洁的自然语言摘要
        summary = f"""天气预测摘要 ({start_time} 至 {end_time})：

总体概况：
- 预测时长：{data_length}小时
- 温度范围：{temp_min:.1f}°C 至 {temp_max:.1f}°C，平均 {temp_avg:.1f}°C
- 湿度范围：{humidity_min:.0f}% 至 {humidity_max:.0f}%，平均 {humidity_avg:.0f}%
- 总降水量：{total_precipitation:.1f}mm
- 降水时段：{rain_hours}小时（最大单小时降水 {max_hourly_rain:.1f}mm）
- 主要天气：{main_weather}

分时段预测："""

        for period in periods:
            summary += f"""
- {period['name']}：平均温度 {period['temp_avg']}°C，降水量 {period['total_rain']}mm，主要天气 {period['main_weather']}"""

        # 添加关键提醒
        warnings = []
        if temp_max > 35:
            warnings.append("高温预警：最高温度超过35°C")
        elif temp_max > 30:
            warnings.append("注意防暑：最高温度超过30°C")
        
        if temp_min < 0:
            warnings.append("低温预警：最低温度低于0°C")
        elif temp_min < 5:
            warnings.append("注意保暖：最低温度低于5°C")
        
        if max_hourly_rain > 10:
            warnings.append("强降水预警：单小时降水量超过10mm")
        elif total_precipitation > 20:
            warnings.append("持续降水：总降水量超过20mm")
        elif total_precipitation > 5:
            warnings.append("有降水：请携带雨具")
        
        if warnings:
            summary += f"\n\n重要提醒：\n" + "\n".join(f"- {w}" for w in warnings)
        
        return summary
        
    except Exception as e:
        return f"数据摘要生成失败：{str(e)}"

@app.route('/api/deepseek-advice', methods=['POST'])
def deepseek_advice():
    """
    调用DeepSeek大模型，根据天气预测结果生成出行建议
    """
    try:
        prediction = request.json
        print("原始prediction数据长度：", len(str(prediction)))
        
        # 将详细预测数据转换为简洁摘要
        prediction_summary = summarize_prediction_for_ai(prediction)
        print("摘要后数据长度：", len(prediction_summary))
        print("预测摘要前200字符：", prediction_summary[:200])
        
        # 进一步简化prompt，减少token数量
        prompt = f"""作为气象出行顾问，基于以下预测生成简洁出行建议：

{prediction_summary}

请分别给出：驾车、骑行、步行、户外活动的建议，每项50字内。"""
        
        print(f"最终prompt长度：{len(prompt)}")
        
        # 调用DeepSeek API，使用更短的超时时间和重试机制
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"尝试第 {attempt + 1} 次调用DeepSeek...")
                response = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "你是气象出行顾问，请简洁回答。"},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "max_tokens": 500  # 限制响应长度
                    },
                    timeout=15  # 缩短超时时间到15秒
                )
                
                if response.status_code == 200:
                    result = response.json()
                    advice = result['choices'][0]['message']['content']
                    print("DeepSeek调用成功！")
                    return jsonify({"advice": advice})
                else:
                    print(f"DeepSeek返回错误状态码：{response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"第 {attempt + 1} 次调用超时")
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回备用建议
                    fallback_advice = f"""基于预测数据的出行建议：

**驾车出行**：注意路况变化，雨天减速慢行，保持车距。

**骑行出行**：检查天气状况，降水时建议室内运动或改期。

**步行出行**：携带适当衣物，注意防滑防雨。

**户外活动**：关注温度变化，做好防护措施。

注：本建议基于天气预测数据生成，请结合实际情况判断。"""
                    
                    print("使用备用建议")
                    return jsonify({"advice": fallback_advice})
            except Exception as e:
                print(f"第 {attempt + 1} 次调用出错：{str(e)}")
                if attempt == max_retries - 1:
                    raise e
        
    except Exception as e:
        print("调用DeepSeek失败:", str(e))
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"调用DeepSeek失败: {str(e)}"}), 500

if __name__ == '__main__':
    print("启动Flask服务器...")
    print("项目根目录:", root_dir)
    print("上传目录:", os.path.abspath(UPLOAD_FOLDER))
    print("旧文件目录:", os.path.abspath(OLD_FILES_FOLDER))
    print("Python路径:", sys.path)
    app.run(debug=True, port=5000) 