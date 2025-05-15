import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.weather_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_columns = [
            'temperature_2m (°C)',
            'relativehumidity_2m (%)',
            'rain (mm)',
            'surface_pressure (hPa)',
            'cloudcover (%)',
            'windspeed_10m (m/s)'
        ]
        self.target_columns = [
            'temperature_2m (°C)',
            'relativehumidity_2m (%)',
            'rain (mm)',
            'weathercode (wmo code)'
        ]
        
        # 保存标准化参数
        self.means = None
        self.stds = None
        
        # 更新天气代码映射
        self.weather_codes = [0, 1, 2, 3, 51, 53, 55, 61, 63, 65]
        # 添加天气代码到连续索引的映射
        self.weather_code_to_index = {code: idx for idx, code in enumerate(self.weather_codes)}
        self.index_to_weather_code = {idx: code for code, idx in self.weather_code_to_index.items()}
        
        self.rain_thresholds = [
            (5.0, 65),    # 大雨 (>5mm)
            (2.0, 63),    # 中雨 (2-5mm)
            (0.5, 61),    # 小雨 (0.5-2mm)
            (0.2, 55),    # 重度毛毛雨 (0.2-0.5mm)
            (0.1, 53),    # 中度毛毛雨 (0.1-0.2mm)
            (0.01, 51),   # 轻度毛毛雨 (0.01-0.1mm)
            (0.0, 3),     # 阴天
            (0.0, 2),     # 阴天
            (0.0, 1),     # 晴天/多云
            (0.0, 0)      # 晴天/多云
        ]
        
        # 添加天气代码描述映射
        self.weather_code_desc = {
            0: "晴天",
            1: "较为晴朗",
            2: "局部多云",
            3: "阴天",
            51: "轻度毛毛雨",
            53: "中度毛毛雨",
            55: "重度毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨"
        }
        
        # 云量阈值
        self.cloud_thresholds = [
            (90.0, 3),    # 阴天多云 (>90%)
            (70.0, 2),    # 阴天 (70-90%)
            (30.0, 1),    # 多云 (30-70%)
            (0.0, 0)      # 晴天 (0-30%)
        ]
        
        # 预先拟合天气编码器
        self.weather_encoder.fit(self.weather_codes)
        print("初始化完成：已预设天气编码器")
    
    def fit(self, df):
        """在训练数据上拟合编码器和标准化器"""
        # 创建时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # 对降水量进行对数变换
        df['rain (mm)'] = np.log1p(df['rain (mm)'])
        
        # 将天气代码转换为连续索引
        df['weathercode (wmo code)'] = df['weathercode (wmo code)'].map(self.weather_code_to_index)
        
        # 批量创建滞后特征
        lag_features = []
        for col in self.feature_columns:
            for lag in [1, 24, 48]:
                lag_feature = df[col].shift(lag)
                lag_feature.name = f'{col}_lag_{lag}'
                lag_features.append(lag_feature)
        
        # 批量创建滚动统计特征
        rolling_features = []
        for col in self.feature_columns:
            rolling_mean = df[col].rolling(window=24).mean()
            rolling_mean.name = f'{col}_rolling_mean_24'
            rolling_std = df[col].rolling(window=24).std()
            rolling_std.name = f'{col}_rolling_std_24'
            rolling_features.extend([rolling_mean, rolling_std])
        
        # 使用pd.concat一次性合并所有特征
        all_features = pd.concat([df] + lag_features + rolling_features, axis=1)
        
        # 删除包含NaN的行
        all_features = all_features.dropna()
        
        # 确保数据不为空
        if len(all_features) == 0:
            raise ValueError("处理后的数据为空，请检查输入数据的有效性")
        
        # 拟合标准化器
        numeric_features = [col for col in all_features.columns if col not in ['weathercode (wmo code)']]
        self.scaler.fit(all_features[numeric_features])
        
        # 保存标准化参数
        self.means = self.scaler.mean_
        self.stds = self.scaler.scale_
        
        # 拟合天气编码器
        all_weather_codes = np.array(self.weather_codes)
        current_codes = all_features['weathercode (wmo code)'].astype(int).unique()
        all_weather_codes = np.unique(np.concatenate([all_weather_codes, current_codes]))
        self.weather_encoder.fit(all_weather_codes)
        
        self.is_fitted = True
        return all_features
    
    def determine_weather_code(self, row):
        """根据降雨量和云量确定天气代码"""
        rain = row['rain (mm)']
        cloud = row['cloudcover (%)']
        
        # 如果有降雨，根据降雨量确定天气代码
        if rain > 0:
            for threshold, code in reversed(self.rain_thresholds):
                if rain >= threshold:
                    return self.weather_code_to_index[code]
        
        # 如果没有降雨，根据云量确定天气代码
        for threshold, code in reversed(self.cloud_thresholds):
            if cloud >= threshold:
                return self.weather_code_to_index[code]
        
        return 0  # 默认返回晴天（索引0）
    
    def adjust_weather_code(self, row):
        """使用新的天气代码确定方法"""
        return self.determine_weather_code(row)
    
    def get_weather_description(self, code):
        """获取天气代码对应的描述"""
        return self.weather_code_desc.get(code, "未知天气")
    
    def preprocess_data(self, df):
        """预处理数据，使用已拟合的编码器和标准化器"""
        if not self.is_fitted:
            raise RuntimeError("编码器和标准化器尚未拟合，请先调用fit方法")
        
        # 创建时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # 对降水量进行对数变换
        df['rain (mm)'] = np.log1p(df['rain (mm)'])
        
        # 批量创建滞后特征
        lag_features = []
        for col in self.feature_columns:
            for lag in [1, 24, 48]:
                lag_feature = df[col].shift(lag)
                lag_feature.name = f'{col}_lag_{lag}'
                lag_features.append(lag_feature)
        
        # 批量创建滚动统计特征
        rolling_features = []
        for col in self.feature_columns:
            rolling_mean = df[col].rolling(window=24).mean()
            rolling_mean.name = f'{col}_rolling_mean_24'
            rolling_std = df[col].rolling(window=24).std()
            rolling_std.name = f'{col}_rolling_std_24'
            rolling_features.extend([rolling_mean, rolling_std])
        
        # 使用pd.concat一次性合并所有特征
        all_features = pd.concat([df] + lag_features + rolling_features, axis=1)
        
        # 删除包含NaN的行
        all_features = all_features.dropna()
        
        # 确保数据不为空
        if len(all_features) == 0:
            raise ValueError("处理后的数据为空，请检查输入数据的有效性")
        
        # 标准化数值特征
        numeric_features = [col for col in all_features.columns if col not in ['weathercode (wmo code)']]
        all_features[numeric_features] = self.scaler.transform(all_features[numeric_features])
        
        return all_features
    
    def create_sequences(self, df, input_days=7, output_days=3):
        # 获取所有特征列（排除目标列）
        feature_columns = [col for col in df.columns if col not in self.target_columns]
        
        # 创建序列
        sequences = []
        targets = []
        
        input_hours = input_days * 24
        output_hours = output_days * 24
        
        # 计算需要的数据长度
        required_length = input_hours + output_hours
        
        # 使用滑动窗口创建多个序列
        for i in range(len(df) - required_length + 1):
            # 获取输入序列
            seq = df[feature_columns].iloc[i:i+input_hours].values
            sequences.append(seq)
            
            # 获取目标序列
            target = df[self.target_columns].iloc[i+input_hours:i+input_hours+output_hours].values
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def inverse_transform_weather(self, encoded_weather):
        """将连续索引转换回原始天气代码"""
        if isinstance(encoded_weather, torch.Tensor):
            encoded_weather = encoded_weather.cpu().numpy()
        return np.array([self.index_to_weather_code[idx] for idx in encoded_weather])
    
    def inverse_transform_features(self, scaled_features):
        """反标准化特征"""
        if self.means is None or self.stds is None:
            raise RuntimeError("标准化参数未保存，请先调用fit方法")
        
        # 确保输入是numpy数组
        if isinstance(scaled_features, torch.Tensor):
            scaled_features = scaled_features.cpu().numpy()
        
        # 手动进行反标准化
        original_features = scaled_features * self.stds + self.means
        return original_features
    
    def inverse_transform_targets(self, scaled_targets):
        """反标准化目标值"""
        if self.means is None or self.stds is None:
            raise RuntimeError("标准化参数未保存，请先调用fit方法")
        
        # 确保输入是numpy数组
        if isinstance(scaled_targets, torch.Tensor):
            scaled_targets = scaled_targets.cpu().numpy()
        
        # 获取目标列的均值和标准差
        target_means = self.means[:len(self.target_columns)-1]  # 排除天气代码
        target_stds = self.stds[:len(self.target_columns)-1]
        
        # 确保输入和目标维度匹配
        if scaled_targets.ndim == 1:
            scaled_targets = scaled_targets.reshape(1, -1)
        
        # 手动进行反标准化
        original_targets = scaled_targets * target_stds + target_means
        
        # 确保温度和湿度在合理范围内
        original_targets[:, 0] = np.clip(original_targets[:, 0], -20, 40)  # 温度范围
        original_targets[:, 1] = np.clip(original_targets[:, 1], 0, 100)   # 湿度范围
        
        # 对降水量进行指数变换
        original_targets[:, 2] = np.expm1(original_targets[:, 2])
        
        return original_targets

    def adjust_predicted_weather(self, weather_code, rain, cloud):
        """根据预测的降水量和云量调整天气代码"""
        # 如果有降雨，根据降雨量确定天气代码
        if isinstance(rain, (np.ndarray, list)):
            if len(rain.shape) > 0:
                rain = rain[0]
            rain = float(rain)
        if isinstance(cloud, (np.ndarray, list)):
            if len(cloud.shape) > 0:
                cloud = cloud[0]
            cloud = float(cloud)
            
        # 将天气代码转换为原始代码
        original_code = self.index_to_weather_code[weather_code]
            
        # 检查天气代码和降雨量是否一致
        is_rainy_weather = original_code in [51, 53, 55, 61, 63, 65]
        
        # 如果天气代码表示有雨但实际没有雨，或者天气代码表示无雨但实际有雨，则进行调整
        if (is_rainy_weather and rain <= 0) or (not is_rainy_weather and rain > 0):
            if rain > 0:
                for threshold, code in reversed(self.rain_thresholds):
                    if rain >= threshold:
                        return self.weather_code_to_index[code]
            else:
                for threshold, code in reversed(self.cloud_thresholds):
                    if cloud >= threshold:
                        return self.weather_code_to_index[code]
        
        return weather_code  # 如果天气代码和降雨量一致，保持原始预测的天气代码

class WeatherDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx] 