import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义分析结果保存路径
ANALYSIS_DIR = 'weather_data/data_analysis'

def load_data(file_path):
    """加载天气数据"""
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    return df

def basic_statistics(df):
    """计算基本统计信息"""
    print("\n=== 基本统计信息 ===")
    print(df.describe())
    
    # 保存统计信息到文件
    stats_file = os.path.join(ANALYSIS_DIR, 'basic_statistics.txt')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== 基本统计信息 ===\n")
        f.write(df.describe().to_string())
    
    return df.describe()

def plot_temperature_trend(df):
    """绘制温度趋势图"""
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['temperature_2m (°C)'], label='温度')
    plt.title('温度变化趋势')
    plt.xlabel('时间')
    plt.ylabel('温度 (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'temperature_trend.png'))
    plt.close()

def plot_weather_distribution(df):
    """绘制天气类型分布"""
    plt.figure(figsize=(10, 6))
    weather_counts = df['weathercode (wmo code)'].value_counts()
    weather_counts.plot(kind='bar')
    plt.title('天气类型分布')
    plt.xlabel('天气类型')
    plt.ylabel('出现次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'weather_distribution.png'))
    plt.close()

def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    plt.figure(figsize=(10, 8))
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'correlation_heatmap.png'))
    plt.close()

def plot_precipitation_distribution(df):
    """绘制降水量分布"""
    plt.figure(figsize=(12, 6))
    plt.hist(df['rain (mm)'], bins=50, alpha=0.7)
    plt.title('降水量分布')
    plt.xlabel('降水量 (mm)')
    plt.ylabel('频次')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'precipitation_distribution.png'))
    plt.close()

def plot_wind_analysis(df):
    """绘制风速和风向分析"""
    # 风速分布
    plt.figure(figsize=(12, 6))
    plt.hist(df['windspeed_10m (m/s)'], bins=50, alpha=0.7)
    plt.title('风速分布')
    plt.xlabel('风速 (m/s)')
    plt.ylabel('频次')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'wind_speed_distribution.png'))
    plt.close()
    
    # 云量分布
    plt.figure(figsize=(12, 6))
    plt.hist(df['cloudcover (%)'], bins=50, alpha=0.7)
    plt.title('云量分布')
    plt.xlabel('云量 (%)')
    plt.ylabel('频次')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cloud_cover_distribution.png'))
    plt.close()

def seasonal_analysis(df):
    """季节性分析"""
    # 添加季节列
    df['season'] = df.index.month % 12 // 3 + 1
    seasons = {1: '冬季', 2: '春季', 3: '夏季', 4: '秋季'}
    df['season_name'] = df['season'].map(seasons)
    
    # 按季节分组统计
    seasonal_stats = df.groupby('season_name').agg({
        'temperature_2m (°C)': ['mean', 'std'],
        'relativehumidity_2m (%)': ['mean', 'std'],
        'rain (mm)': ['mean', 'sum'],
        'windspeed_10m (m/s)': ['mean', 'std'],
        'cloudcover (%)': ['mean', 'std']
    })
    
    # 保存季节性统计信息
    seasonal_file = os.path.join(ANALYSIS_DIR, 'seasonal_statistics.txt')
    with open(seasonal_file, 'w', encoding='utf-8') as f:
        f.write("=== 季节性统计信息 ===\n")
        f.write(seasonal_stats.to_string())
    
    return seasonal_stats

def main():
    # 创建分析结果目录
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # 加载数据
    data_path = 'weather_data/data_w/1y.csv'
    df = load_data(data_path)
    
    # 执行各项分析
    basic_statistics(df)
    plot_temperature_trend(df)
    plot_weather_distribution(df)
    plot_correlation_heatmap(df)
    plot_precipitation_distribution(df)
    plot_wind_analysis(df)
    seasonal_stats = seasonal_analysis(df)
    
    print(f"\n分析完成！结果已保存到 {ANALYSIS_DIR} 目录。")

if __name__ == '__main__':
    main() 