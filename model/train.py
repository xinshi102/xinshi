import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from weather_processor import WeatherDataProcessor, WeatherDataset
from weather_model import WeatherLSTM
import pickle
from datetime import datetime
import torch.nn.functional as F
# 设置编码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def adjust_task_weights(losses, current_weights, min_weight=0.1, max_weight=0.35):
    """调整任务权重，添加权重上下限"""
    # 计算相对损失
    relative_losses = [loss / (current_weights[i] + 1e-8) for i, loss in enumerate(losses)]
    total_relative_loss = sum(relative_losses)
    
    # 计算新的权重，确保在上下限之间
    new_weights = [max(min_weight, min(max_weight, loss / total_relative_loss)) for loss in relative_losses]
    
    # 归一化权重
    weight_sum = sum(new_weights)
    new_weights = [w / weight_sum for w in new_weights]
    
    return new_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # 创建TensorBoard日志目录
    log_dir = 'runs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    
    train_losses = []
    val_losses = []
    
    # 早停参数
    patience = 10
    min_delta = 0.001
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 初始任务权重
    task_weights = {
        'weather':0.15,
        'temp': 0.2,
        'humidity': 0.2,
        'precip': 0.25,
        'weather_precip_correlation': 0.2
    }
    
    # 权重调整参数
    weight_clip_min = 0.1
    weight_clip_max = 10.0
    weight_adjust_rate = 0.1
    
    # 获取天气代码列表
    processor = WeatherDataProcessor()
    weather_codes = processor.weather_codes
    
    # 定义雨天天气代码
    rain_codes = [51, 53, 55, 61, 63, 65]
    rain_indices = [weather_codes.index(code) for code in rain_codes]
    
    # 定义降水量区间和对应的权重
    precip_ranges = [
        (0.0, 0.01, 1.0),    # 无雨
        (0.01, 0.1, 1.2),    # 毛毛雨
        (0.1, 0.5, 1.5),     # 小雨
        (0.5, 2.5, 2.0),     # 中雨
        (2.5, 10.0, 2.5),    # 大雨
        (10.0, float('inf'), 3.0)  # 暴雨
    ]
    
    def get_precip_weight(precip_value):
        """根据降水量获取权重"""
        for low, high, weight in precip_ranges:
            if low <= precip_value < high:
                return weight
        return 1.0
    
    def adjust_task_weights(task_losses, current_weights):
        """安全地调整任务权重"""
        # 计算每个任务的损失比例
        total_loss = sum(task_losses.values())
        if total_loss <= 0:
            return current_weights.copy()
            
        loss_ratios = {task: loss / total_loss for task, loss in task_losses.items()}
        new_weights = current_weights.copy()
        
        # 根据损失比例和当前权重的差异来调整
        for task in new_weights:
            # 计算目标权重（基于损失比例）
            target_weight = loss_ratios[task]
            
            # 如果当前权重与目标权重差异较大，则进行调整
            if abs(current_weights[task] - target_weight) > 0.05:  # 差异阈值
                # 缓慢向目标权重移动
                new_weights[task] = current_weights[task] + (target_weight - current_weights[task]) * 0.1
        
        # 确保权重在合理范围内
        for task in new_weights:
            new_weights[task] = max(0.15, min(0.35, new_weights[task]))
        
        # 归一化权重
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            for task in new_weights:
                new_weights[task] = new_weights[task] / weight_sum
        
        return new_weights
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        task_losses = {
            'weather': 0.0,
            'temp': 0.0,
            'humidity': 0.0,
            'precip': 0.0,
            'weather_precip_correlation': 0.0
        }
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
            
            # 获取目标值
            weather_target = targets[:, :, 3].long()
            temp_target = targets[:, :, 0]
            humidity_target = targets[:, :, 1]
            precip_target = targets[:, :, 2]
            
            # 计算各项损失
            # 1. 天气分类损失
            weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), 
                                      weather_target.view(-1))
            
            # 2. 温度损失（添加范围约束）
            temp_loss = criterion[1](temp_pred.squeeze(), temp_target) * 0.1  # 缩放温度损失
            temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.01
            
            # 3. 湿度损失（添加范围约束）
            humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target) * 0.1  # 缩放湿度损失
            humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.01
            
            # 4. 降水损失（带权重）
            precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target.view(-1)],
                                        device=device)
            precip_loss = criterion[1](precip_pred.squeeze(), precip_target) * precip_weights.mean() * 0.1  # 缩放降水损失
            
            # 5. 天气和降水量的关联性损失
            weather_probs = F.softmax(weather_pred, dim=-1)
            rain_probs = torch.sum(torch.stack([weather_probs[:, :, i] for i in rain_indices]), dim=0)
            
            # 计算期望的雨天概率
            expected_rain_probs = torch.zeros_like(precip_target, device=device)
            batch_size, seq_len = precip_target.shape
            
            precip_vals = precip_target.view(-1)
            expected_probs = torch.zeros_like(precip_vals, device=device)
            
            heavy_rain_mask = precip_vals >= 10.0
            light_rain_mask = (precip_vals > 0.0) & (precip_vals < 10.0)
            
            expected_probs[heavy_rain_mask] = 1.0
            expected_probs[light_rain_mask] = torch.minimum(
                precip_vals[light_rain_mask] / 10.0 + 0.3,
                torch.tensor(1.0, device=device)
            )
            
            expected_rain_probs = expected_probs.view(batch_size, seq_len)
            correlation_loss = F.mse_loss(rain_probs, expected_rain_probs) * 0.1  # 缩放相关性损失
            
            # 更新任务损失统计
            task_losses['weather'] += weather_loss.item()
            task_losses['temp'] += temp_loss.item()
            task_losses['humidity'] += humidity_loss.item()
            task_losses['precip'] += precip_loss.item()
            task_losses['weather_precip_correlation'] += correlation_loss.item()
            
            # 计算总损失
            loss = (weather_loss * task_weights['weather'] +
                   temp_loss * task_weights['temp'] +
                   humidity_loss * task_weights['humidity'] +
                   precip_loss * task_weights['precip'] +
                   correlation_loss * task_weights['weather_precip_correlation'] +
                   0.01 * (temp_range_penalty + humidity_range_penalty))  # 降低范围惩罚的权重
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 记录到TensorBoard
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Training/Total_Loss', loss.item(), step)
                writer.add_scalar('Training/Weather_Loss', weather_loss.item(), step)
                writer.add_scalar('Training/Temp_Loss', temp_loss.item(), step)
                writer.add_scalar('Training/Humidity_Loss', humidity_loss.item(), step)
                writer.add_scalar('Training/Precip_Loss', precip_loss.item(), step)
                writer.add_scalar('Training/Correlation_Loss', correlation_loss.item(), step)
        
        # 计算平均损失
        for task in task_losses:
            task_losses[task] /= batch_count
        
        # 安全地调整任务权重
        if epoch > 0 and epoch % 5 == 0:
            task_weights = adjust_task_weights(task_losses, task_weights)
        
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_task_losses = {
            'weather': 0.0,
            'temp': 0.0,
            'humidity': 0.0,
            'precip': 0.0,
            'weather_precip_correlation': 0.0
        }
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
                
                weather_target = targets[:, :, 3].long()
                temp_target = targets[:, :, 0]
                humidity_target = targets[:, :, 1]
                precip_target = targets[:, :, 2]
                
                # 计算验证集损失
                weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), 
                                          weather_target.view(-1))
                temp_loss = criterion[1](temp_pred.squeeze(), temp_target) * 0.1  # 缩放温度损失
                humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target) * 0.1  # 缩放湿度损失
                
                precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target.view(-1)],
                                            device=device)
                precip_loss = criterion[1](precip_pred.squeeze(), precip_target) * precip_weights.mean() * 0.1  # 缩放降水损失
                
                # 计算天气和降水量的关联性损失
                weather_probs = F.softmax(weather_pred, dim=-1)
                rain_probs = torch.sum(torch.stack([weather_probs[:, :, i] for i in rain_indices]), dim=0)
                
                expected_rain_probs = torch.zeros_like(precip_target, device=device)
                batch_size, seq_len = precip_target.shape
                
                precip_vals = precip_target.view(-1)
                expected_probs = torch.zeros_like(precip_vals, device=device)
                
                heavy_rain_mask = precip_vals >= 10.0
                light_rain_mask = (precip_vals > 0.0) & (precip_vals < 10.0)
                
                expected_probs[heavy_rain_mask] = 1.0
                expected_probs[light_rain_mask] = torch.minimum(
                    precip_vals[light_rain_mask] / 10.0 + 0.3,
                    torch.tensor(1.0, device=device)
                )
                
                expected_rain_probs = expected_probs.view(batch_size, seq_len)
                correlation_loss = F.mse_loss(rain_probs, expected_rain_probs) * 0.1  # 缩放相关性损失
                
                # 添加范围约束惩罚
                temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.01
                humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.01
                
                loss = (weather_loss * task_weights['weather'] +
                       temp_loss * task_weights['temp'] +
                       humidity_loss * task_weights['humidity'] +
                       precip_loss * task_weights['precip'] +
                       correlation_loss * task_weights['weather_precip_correlation'] +
                       0.01 * (temp_range_penalty + humidity_range_penalty))  # 使用与训练阶段相同的范围惩罚权重
                
                val_loss += loss.item()
                val_task_losses['weather'] += weather_loss.item()
                val_task_losses['temp'] += temp_loss.item()
                val_task_losses['humidity'] += humidity_loss.item()
                val_task_losses['precip'] += precip_loss.item()
                val_task_losses['weather_precip_correlation'] += correlation_loss.item()
                val_batch_count += 1
        
        # 计算验证集平均损失
        for task in val_task_losses:
            val_task_losses[task] /= val_batch_count
        
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                model.load_state_dict(best_model_state)
                break
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {avg_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        print("任务权重:", {k: f"{v:.2f}" for k, v in task_weights.items()})
        print("---")
    
    writer.close()
    return train_losses, val_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    current_dir = os.path.dirname(os.path.abspath(__file__))#获取当前文件所在目录的绝对路径
    data_path = os.path.join(current_dir, '..', 'data', 'data_w', '5y.csv')  #构建数据文件的绝对路径
    print(f"\n正在加载数据文件: {data_path}")
    data = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"原始数据形状: {data.shape}")
    print(f"数据列名: {data.columns.tolist()}")
    
    # 数据预处理
    print("\n开始数据预处理...")
    processor = WeatherDataProcessor()
    print("正在拟合处理器...")
    processed_data = processor.fit(data)  # 在训练数据上拟合编码器和标准化器
    print(f"拟合后的数据形状: {processed_data.shape}")
    print("正在预处理数据...")
    processed_data = processor.preprocess_data(data)
    print(f"预处理后的数据形状: {processed_data.shape}")
    
    # 打印数据形状信息
    print("\n数据预处理后的形状信息：")
    print(f"原始数据形状: {data.shape}")
    print(f"处理后的数据形状: {processed_data.shape}")
    print(f"特征列数: {len(processed_data.columns)}")
    print("\n特征列名：")
    for col in processed_data.columns:
        print(f"- {col}")
    
    # 创建序列
    print("\n正在创建序列...")
    sequences, targets = processor.create_sequences(processed_data, input_days=7, output_days=3)
    
    # 打印序列形状信息
    print("\n序列形状信息：")
    print(f"输入序列形状: {sequences.shape}")
    print(f"目标序列形状: {targets.shape}")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(sequences))
    X_train, X_val = sequences[:train_size], sequences[train_size:]
    y_train, y_val = targets[:train_size], targets[train_size:]
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 模型参数
    input_size = 36  # 特征维度
    hidden_size = 96
    num_layers = 1
    output_size = 72  # 3天 * 24小时
    weather_categories = 10  # 更新天气类别数量为10种
    
    # 初始化模型
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size, weather_categories).to(device)
    
    # 定义损失函数
    criterion = [
        nn.CrossEntropyLoss(),  # 天气分类损失
        nn.MSELoss()  # 回归任务损失
    ]
    
    # 定义优化器，使用固定学习率0.0001
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练模型
    num_epochs = 100
    print("\n开始训练模型...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # 保存模型
    model_path = os.path.join(current_dir, '..', 'data', 'weather_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存处理器
    processor_path = os.path.join(current_dir, '..', 'data', 'weather_processor.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"\n处理器已保存到: {processor_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    print("\n训练完成！")

if __name__ == '__main__':
    main() 