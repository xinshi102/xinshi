# new_models 模块说明文档

## 1. 文件结构
```
new_models/
├── weather_model.py    # 核心模型定义
├── weather_processor.py # 数据预处理和特征工程
├── train.py           # 模型训练脚本
└── predict.py         # 模型预测脚本
```

## 2. 各文件详细说明

### 2.1 weather_model.py
这是模型的核心定义文件，定义了`WeatherLSTM`类。

主要组件：
1. **LSTM层**：
   ```python
   self.lstm = nn.LSTM(
       input_size=input_size,
       hidden_size=hidden_size,
       num_layers=1,
       batch_first=True
   )
   ```
   - 用于捕获时间序列的长期依赖关系
   - 单层LSTM，避免梯度消失问题

2. **序列压缩层**：
   ```python
   self.sequence_compressor = nn.Sequential(
       nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2),
       nn.ReLU(),
       nn.Conv1d(hidden_size, hidden_size, kernel_size=13, stride=1, padding=6)
   )
   ```
   - 将输入序列压缩为固定长度
   - 第一层：捕捉短周期模式
   - 第二层：捕捉长周期模式

3. **预测头**：
   - 天气分类器：预测天气类型
   - 温度回归器：预测温度
   - 湿度回归器：预测湿度
   - 降水回归器：预测降水量

### 2.2 weather_processor.py
负责数据预处理和特征工程。

主要功能：
1. **特征工程**：
   ```python
   def fit(self, df):
       # 创建时间特征
       df['hour'] = df.index.hour
       df['day_of_week'] = df.index.dayofweek
       df['month'] = df.index.month
       
       # 创建滞后特征
       for col in self.feature_columns:
           for lag in [1, 24, 48]:
               lag_feature = df[col].shift(lag)
   ```
   - 时间特征提取
   - 滞后特征创建
   - 滚动统计特征计算

2. **数据标准化**：
   ```python
   def preprocess_data(self, df):
       # 标准化数值特征
       numeric_features = [col for col in all_features.columns if col not in ['weathercode (wmo code)']]
       all_features[numeric_features] = self.scaler.transform(all_features[numeric_features])
   ```
   - 数值特征标准化
   - 天气代码编码

### 2.3 train.py
负责模型训练流程。

主要步骤：
1. **数据准备**：
   ```python
   def main():
       # 加载数据
       data = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
       
       # 数据预处理
       processor = WeatherDataProcessor()
       processed_data = processor.fit(data)
   ```

2. **模型训练**：
   ```python
   def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
       # 训练循环
       for epoch in range(num_epochs):
           # 前向传播
           weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
           
           # 计算损失
           loss = weather_loss + temp_loss + humidity_loss + precip_loss
   ```
   - 多任务学习
   - 损失函数组合
   - 验证集评估

### 2.4 predict.py
负责模型预测和结果可视化。

主要功能：
1. **预测流程**：
   ```python
   def make_predictions():
       # 加载模型和数据
       model = WeatherLSTM(...)
       df = pd.read_csv(data_path)
       
       # 预处理
       processed_data = processor.preprocess_data(df)
       
       # 预测
       weather_pred, temp_pred, humidity_pred, precip_pred = model(input_sequence)
   ```

2. **结果可视化**：
   ```python
   def main():
       # 绘制预测结果
       plt.figure(figsize=(15, 10))
       plt.subplot(3, 1, 1)
       plt.plot(weather_codes, label='预测天气代码')
   ```
   - 天气代码预测图
   - 温度预测图
   - 湿度预测图

## 3. 工作流程

1. **数据准备阶段**：
   - 加载原始数据
   - 使用`weather_processor.py`进行特征工程
   - 数据标准化和编码

2. **训练阶段**：
   - 使用`train.py`训练模型
   - 保存训练好的模型和处理器

3. **预测阶段**：
   - 使用`predict.py`加载模型
   - 预处理新数据
   - 进行预测
   - 可视化结果

## 4. 关键参数说明

1. **模型参数**：
   - `input_size`: 输入特征维度
   - `hidden_size`: LSTM隐藏层大小
   - `num_layers`: LSTM层数
   - `output_size`: 输出序列长度
   - `weather_categories`: 天气类型数量

2. **数据参数**：
   - `input_days`: 输入序列长度（天）
   - `output_days`: 预测序列长度（天）
   - `lag_features`: 滞后特征时间跨度
   - `rolling_window`: 滚动统计窗口大小

## 5. 使用建议

1. **数据准备**：
   - 确保数据时间连续性
   - 检查异常值和缺失值
   - 保证数据质量

2. **模型训练**：
   - 选择合适的超参数
   - 监控训练过程
   - 保存最佳模型

3. **预测应用**：
   - 确保输入数据格式正确
   - 检查预测结果合理性
   - 必要时进行后处理 