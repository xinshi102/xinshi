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

# 确保上传目录和旧文件目录存在
UPLOAD_FOLDER = os.path.join(root_dir, 'uploads')
OLD_FILES_FOLDER = os.path.join(root_dir, 'old_files')
for folder in [UPLOAD_FOLDER, OLD_FILES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 初始化天气处理器以获取天气代码描述
try:
    weather_processor = WeatherDataProcessor()
    print("成功初始化天气处理器")
except Exception as e:
    print(f"初始化天气处理器时出错: {str(e)}")
    print(traceback.format_exc())

# 定义天气代码映射
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

def move_old_files():
    """将旧文件移动到old_files目录"""
    try:
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 移动uploads目录中的所有文件到old_files目录
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.csv'):
                old_path = os.path.join(UPLOAD_FOLDER, filename)
                new_filename = f"{timestamp}_{filename}"
                new_path = os.path.join(OLD_FILES_FOLDER, new_filename)
                shutil.move(old_path, new_path)
                print(f"已移动文件: {filename} -> {new_filename}")
    except Exception as e:
        print(f"移动旧文件时出错: {str(e)}")

def replace_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [replace_nan(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    else:
        return obj

def to_python_type(obj):
    """递归将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        print("接收到文件上传请求")
        
        if 'file' not in request.files:
            print("错误：请求中没有文件")
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("错误：没有选择文件")
            return jsonify({'error': '没有选择文件'}), 400

        print(f"上传的文件名: {file.filename}")

        # 移动旧文件
        move_old_files()

        # 保存新文件
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"文件已保存到: {file_path}")

        # 读取数据
        print("正在读取CSV文件...")
        df = pd.read_csv(file_path)
        print(f"CSV文件列名: {df.columns.tolist()}")
        
        # 检查必需的列是否存在
        required_columns = [
            'time', 
            'temperature_2m (°C)', 
            'relativehumidity_2m (%)',
            'rain (mm)',
            'surface_pressure (hPa)',
            'cloudcover (%)',
            'windspeed_10m (m/s)',
            'weathercode (wmo code)'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"CSV文件缺少必需的列: {', '.join(missing_columns)}"
            print(f"错误：{error_msg}")
            # 删除无效文件
            os.remove(file_path)
            return jsonify({'error': error_msg}), 400

        # 返回原始数据用于显示
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
            os.remove(file_path)
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
        return jsonify({'error': error_msg}), 500

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
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
        
        # 准备响应数据，确保所有字段为list且不为None
        response_data = {
            'times': list(times) if times is not None else [],
            'weather_codes': list(weather_codes) if weather_codes is not None else [],
            'weather_descriptions': list(weather_descriptions) if weather_descriptions is not None else [],
            'temperatures': list(temperatures) if temperatures is not None else [],
            'humidities': list(humidities) if humidities is not None else [],
            'precipitations': list(precipitations) if precipitations is not None else []
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
def get_weather_codes():
    return jsonify(WEATHER_CODES)

@app.route('/api/analysis-data', methods=['GET'])
def get_analysis_data():
    try:
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
        
        # 准备分析数据
        analysis_data = {
            'temperatureTrend': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['temperature_2m (°C)'].tolist()
            },
            'humidityTrend': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['relativehumidity_2m (%)'].tolist()
            },
            'precipitationDistribution': {
                'bins': np.histogram(df['rain (mm)'], bins=50)[1].tolist()[:-1],
                'counts': np.histogram(df['rain (mm)'], bins=50)[0].tolist()
            },
            'weatherDistribution': {
                'categories': [WEATHER_CODES.get(code, f'未知({code})') for code in df['weathercode (wmo code)'].unique()],
                'values': df['weathercode (wmo code)'].value_counts().tolist(),
                'codes': df['weathercode (wmo code)'].unique().tolist()
            },
            'cloudCoverDistribution': {
                'bins': np.histogram(df['cloudcover (%)'], bins=50)[1].tolist()[:-1],
                'counts': np.histogram(df['cloudcover (%)'], bins=50)[0].tolist()
            },
            # 新增风速分布
            'windSpeedDistribution': {
                'bins': np.histogram(df['windspeed_10m (m/s)'], bins=50)[1].tolist()[:-1],
                'counts': np.histogram(df['windspeed_10m (m/s)'], bins=50)[0].tolist()
            },
            # 新增相关性矩阵
            'correlationMatrix': {
                'categories': ['温度', '湿度', '降水量', '气压', '云量', '风速'],
                'values': df[['temperature_2m (°C)', 'relativehumidity_2m (%)', 'rain (mm)', 
                              'surface_pressure (hPa)', 'cloudcover (%)', 'windspeed_10m (m/s)']].corr().values.tolist()
            },
            # 新增湿度热力图数据
            'humidityHeatmap': {
                'times': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df['relativehumidity_2m (%)'].tolist()
            }
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
def get_current_weather():
    try:
        upload_dir = os.path.join(root_dir, 'uploads')
        files = os.listdir(upload_dir)
        if not files:
            return jsonify({'error': '请先上传数据文件'}), 400
        latest_file = max([os.path.join(upload_dir, f) for f in files], key=os.path.getctime)
        df = pd.read_csv(latest_file)
        weather_code = int(df['weathercode (wmo code)'].iloc[-1])
        return jsonify({'weather_code': weather_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动Flask服务器...")
    print("项目根目录:", root_dir)
    print("上传目录:", os.path.abspath(UPLOAD_FOLDER))
    print("旧文件目录:", os.path.abspath(OLD_FILES_FOLDER))
    print("Python路径:", sys.path)
    app.run(debug=True, port=5000) 