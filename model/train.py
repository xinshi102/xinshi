import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
<<<<<<< HEAD
# from torch.utils.tensorboard import SummaryWriter  # 移除TensorBoard相关导入
=======
from torch.utils.tensorboard import SummaryWriter
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
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
<<<<<<< HEAD
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负

def adjust_task_weights(task_losses, current_weights, prev_losses=None):
    """
    考虑损失趋势的情况下调整多任务损失
    task_losses: 当前各任务损失
    current_weights: 当前各任务权重
    prev_losses: 上一轮各任务损失
    :return: 新的权重字典
    """
    total_loss = sum(task_losses.values())#总损失和
    if total_loss <= 0:
        return current_weights.copy()#避免除0或者无效调整
    loss_ratios = {task: loss / total_loss for task, loss in task_losses.items()}#计算每个任务损失占总损失的比例
    new_weights = current_weights.copy()#复制权重，防止修改原始数据
    max_adjustment = 0.02  # 调整最大比例
    if prev_losses is not None:# 计算损失变化趋势，非首次调整使用
        loss_changes = {
            task: (task_losses[task] - prev_losses[task]) / (prev_losses[task] + 1e-8)#采用相对变化
            for task in task_losses#将task_loss中的每个元素赋值给task
        }
        for task in new_weights:
            target_weight = loss_ratios[task]#计算该任务理想权重比例
            current_weight = current_weights[task]
            # 损失增加，降低权重
            if loss_changes[task] > 0.05:#如果损失增加%5，降低权重
                adjustment = min(max_adjustment, current_weight * 0.1)#不超过最大调整值
                new_weights[task] = current_weight - adjustment
            # 损失减少，适当增加权重
            elif loss_changes[task] < -0.05:
=======
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def adjust_task_weights(task_losses, current_weights, prev_losses=None):
    """安全地调整任务权重，考虑损失变化趋势"""
    # 计算每个任务的损失比例
    total_loss = sum(task_losses.values())
    if total_loss <= 0:
        return current_weights.copy()
        
    loss_ratios = {task: loss / total_loss for task, loss in task_losses.items()}
    new_weights = current_weights.copy()
    
    # 限制每次调整的最大幅度
    max_adjustment = 0.02  # 降低到2%，使调整更平缓
    
    if prev_losses is not None:
        # 计算损失变化趋势
        loss_changes = {
            task: (task_losses[task] - prev_losses[task]) / (prev_losses[task] + 1e-8)
            for task in task_losses
        }
        
        # 根据损失变化趋势调整权重
        for task in new_weights:
            target_weight = loss_ratios[task]
            current_weight = current_weights[task]
            
            # 如果损失增加，降低该任务的权重
            if loss_changes[task] > 0.05:  # 损失增加超过5%
                adjustment = min(max_adjustment, current_weight * 0.1)  # 最多降低当前权重的10%
                new_weights[task] = current_weight - adjustment
            # 如果损失减少，适当增加权重
            elif loss_changes[task] < -0.05:  # 损失减少超过5%
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
                adjustment = min(max_adjustment, abs(target_weight - current_weight))
                new_weights[task] = current_weight + adjustment
            # 损失变化不大，微调
            else:
                adjustment = min(max_adjustment * 0.5, abs(target_weight - current_weight))
                if target_weight > current_weight:
                    new_weights[task] = current_weight + adjustment
                else:
                    new_weights[task] = current_weight - adjustment
    else:
<<<<<<< HEAD
        # 首次调整，调整减半
=======
        # 首次调整，使用较小的调整幅度
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        for task in new_weights:
            target_weight = loss_ratios[task]
            current_weight = current_weights[task]
            adjustment = min(max_adjustment * 0.5, abs(target_weight - current_weight))
            if target_weight > current_weight:
                new_weights[task] = current_weight + adjustment
            else:
                new_weights[task] = current_weight - adjustment
<<<<<<< HEAD
    # 权重范围限制
    for task in new_weights:
        new_weights[task] = max(0.15, min(0.35, new_weights[task]))
    # 归一化
=======
    
    # 确保权重在合理范围内
    for task in new_weights:
        new_weights[task] = max(0.15, min(0.35, new_weights[task]))
    
    # 归一化权重
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    weight_sum = sum(new_weights.values())
    if weight_sum > 0:
        for task in new_weights:
            new_weights[task] = new_weights[task] / weight_sum
<<<<<<< HEAD
    return new_weights


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型主循环，包含多任务损失、动态权重调整、早停、TensorBoard日志等
    """
    # 创建TensorBoard日志目录
    # log_dir = 'runs'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # writer = SummaryWriter(log_dir)
    train_losses = []
    val_losses = []
    # 早停参数
    patience = 10
    min_delta = 0.001
    best_val_loss = float('inf')#使用无穷大作为初始值，任何损失都会比它小
    patience_counter = 0
    best_model_state = None
    # 初始任务权重
    task_weights = {
        'weather': 0.25,
=======
    
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
    
    # 设置更合理的初始权重
    task_weights = {
        'weather': 0.25,  # 略微提高初始权重
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        'temp': 0.20,
        'humidity': 0.20,
        'precip': 0.20,
        'weather_precip_correlation': 0.15
    }
<<<<<<< HEAD
    # 保存最近几个epoch的损失
    recent_losses = {task: [] for task in task_weights}
    loss_window_size = 3#滑动窗口保存最近三个epoch的损失
    prev_epoch_losses = None#保存上一轮各个任务的损失值
    # 获取天气代码列表
    processor = WeatherDataProcessor()
    weather_codes = processor.weather_codes
    # 定义雨天天气代码
    rain_codes = [51, 53, 55, 61, 63, 65]
    rain_indices = [weather_codes.index(code) for code in rain_codes]
    # 降水量区间权重 - 降低权重避免过拟合
    def get_precip_weight(precip_value):
        # 更温和的权重策略，避免过拟合
        if precip_value > 5.0:
            return 8.0  # 大雨权重
        elif precip_value > 1.0:
            return 5.0  # 中雨权重
        elif precip_value > 0.1:
            return 3.0  # 小雨权重
        elif precip_value > 0:
            return 2.0  # 微量降水权重
        else:
            return 1.0  # 无降水权重
    # 训练主循环
    for epoch in range(num_epochs):
        model.train()#训练模式
        total_loss = 0
        task_losses = {k: 0.0 for k in task_weights}#字典记录每个任务的累计损失
        batch_count = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()#梯度清0
=======
    
    # 保存最近几个epoch的损失
    recent_losses = {task: [] for task in task_weights}
    loss_window_size = 3  # 保存最近7个epoch的损失
    prev_epoch_losses = None  # 用于记录上一个epoch的损失
    
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
        return 3.0 if precip_value > 0 else 1.0
    
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
            
            # 打印批次信息
            if batch_idx == 0:
                print(f"训练批次 {batch_idx} 形状: 输入={inputs.shape}, 目标={targets.shape}")
            
            optimizer.zero_grad()
            
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            # 前向传播
            weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
            
            # 获取目标值
            weather_target = targets[:, :, 3].long()
<<<<<<< HEAD
            temp_target = targets[:, :, 0]#取出所有样本和时间步的特征
            humidity_target = targets[:, :, 1]
            precip_target = targets[:, :, 2]
            
            # 调试信息（仅在第一个batch打印）
            if batch_idx == 0 and epoch == 0:
                print(f"调试信息 - 输入维度: {inputs.shape}")
                print(f"调试信息 - 目标维度: {targets.shape}")
                print(f"调试信息 - 天气预测维度: {weather_pred.shape}")
                print(f"调试信息 - 温度预测维度: {temp_pred.shape}")
                print(f"调试信息 - 湿度预测维度: {humidity_pred.shape}")
                print(f"调试信息 - 降水预测维度: {precip_pred.shape}")
                print(f"调试信息 - 降水目标维度: {precip_target.shape}")
                print(f"调试信息 - 降水预测squeeze后维度: {precip_pred.squeeze().shape}")
                print(f"调试信息 - 降水预测squeeze后维数: {precip_pred.squeeze().dim()}")
            # 1. 天气分类损失，多类别，降低维度一次性计算所有样本的损失，调用交叉熵损失函数
            weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), weather_target.view(-1))
            # 2. 温度损失，回归，范围惩罚
            temp_loss = criterion[1](temp_pred.squeeze(), temp_target)#调用均方误差函数，去掉形状中多余的1维度
            temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.1
            # 3. 湿度损失，回归，范围惩罚
            humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target)
            humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.1
            # 4. 降水损失，回归，使用Huber损失更鲁棒
            # 确保预测和目标的维度匹配
            precip_pred_squeezed = precip_pred.squeeze()
            
            # 确保两者都是相同的形状
            if precip_pred_squeezed.dim() == 3:
                precip_pred_squeezed = precip_pred_squeezed.squeeze(-1)
            
            # 将两者都展平为一维
            precip_pred_flat = precip_pred_squeezed.contiguous().view(-1)
            precip_target_flat = precip_target.contiguous().view(-1)
            
            # 确保长度一致
            min_len = min(len(precip_pred_flat), len(precip_target_flat))
            precip_pred_flat = precip_pred_flat[:min_len]
            precip_target_flat = precip_target_flat[:min_len]
            
            # 计算权重
            precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target_flat], device=device)
            
            # 使用Huber损失替代MSE，对异常值更鲁棒
            precip_loss = F.smooth_l1_loss(precip_pred_flat, precip_target_flat, reduction='none')
            precip_loss = (precip_loss * precip_weights).mean()  # 应用权重
            # 5. 天气和降水量的关联性损失
            weather_probs = F.softmax(weather_pred, dim=-1)
            rain_probs = torch.sum(torch.stack([weather_probs[:, :, i] for i in rain_indices]), dim=0)#降雨概率提取，将所有样本的降雨概率相加得到序列
            expected_rain_probs = torch.zeros_like(precip_target, device=device)
            batch_size, seq_len = precip_target.shape#获取目标值的形状
            precip_vals = precip_target.view(-1)#将目标值转换成1维
            expected_probs = torch.zeros_like(precip_vals, device=device)#创建与目标值相同形状的0张量
            heavy_rain_mask = precip_vals >= 5.0#大雨阈值为5mm
            light_rain_mask = (precip_vals > 0.0) & (precip_vals < 5.0)
            expected_probs[heavy_rain_mask] = 1.0
            expected_probs[light_rain_mask] = torch.minimum(
                precip_vals[light_rain_mask] / 5.0 + 0.3,#归一化降雨量0~1，添加基础概率0.3，使小雨的概率在0.3和1.0之间
                torch.tensor(1.0, device=device)#创建值为1.0的张量
            )
            expected_rain_probs = expected_probs.view(batch_size, seq_len)#1维度再转2维，与rain_prob保持一致
            correlation_loss = F.mse_loss(rain_probs, expected_rain_probs)#用mse计算模型预测的降雨概率和根据降雨量推算的概率之间的差异
            # 统计各项损失
=======
            temp_target = targets[:, :, 0]
            humidity_target = targets[:, :, 1]
            precip_target = targets[:, :, 2]
            
            # 计算各项损失
            # 1. 天气分类损失
            weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), 
                                      weather_target.view(-1))
            
            # 2. 温度损失（添加范围约束）
            temp_loss = criterion[1](temp_pred.squeeze(), temp_target)  # 移除0.1缩放
            temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.1  # 改为0.1
            
            # 3. 湿度损失（添加范围约束）
            humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target)  # 移除0.1缩放
            humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.1  # 改为0.1
            
            # 4. 降水损失（带权重）
            precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target.view(-1)],
                                        device=device)
            precip_loss = criterion[1](precip_pred.squeeze(), precip_target) * precip_weights.mean()  # 移除0.1缩放
            
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
            correlation_loss = F.mse_loss(rain_probs, expected_rain_probs)  # 移除0.1缩放
            
            # 更新任务损失统计
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            task_losses['weather'] += weather_loss.item()
            task_losses['temp'] += temp_loss.item()
            task_losses['humidity'] += humidity_loss.item()
            task_losses['precip'] += precip_loss.item()
<<<<<<< HEAD
            task_losses['weather_precip_correlation'] += correlation_loss.item()#损失转为python 标量，累加到字典中
            # 总损失
=======
            task_losses['weather_precip_correlation'] += correlation_loss.item()
            
            # 计算总损失
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            loss = (weather_loss * task_weights['weather'] +
                   temp_loss * task_weights['temp'] +
                   humidity_loss * task_weights['humidity'] +
                   precip_loss * task_weights['precip'] +
                   correlation_loss * task_weights['weather_precip_correlation'] +
<<<<<<< HEAD
                   0.1 * (temp_range_penalty + humidity_range_penalty))#总损失权重
            # 反向传播
            loss.backward()#计算梯度，使用链式法则
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()#更新模型参数，使用优化器的规则(adam)
            total_loss += loss.item()#累加损失
            batch_count += 1
            # TensorBoard记录
            # if batch_idx % 100 == 0:
            #     step = epoch * len(train_loader) + batch_idx
            #     writer.add_scalar('Training/Total_Loss', loss.item(), step)
            #     writer.add_scalar('Training/Weather_Loss', weather_loss.item(), step)
            #     writer.add_scalar('Training/Temp_Loss', temp_loss.item(), step)
            #     writer.add_scalar('Training/Humidity_Loss', humidity_loss.item(), step)
            #     writer.add_scalar('Training/Precip_Loss', precip_loss.item(), step)
            #     writer.add_scalar('Training/Correlation_Loss', correlation_loss.item(), step)
=======
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
        
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        # 计算平均损失
        for task in task_losses:
            task_losses[task] /= batch_count
            recent_losses[task].append(task_losses[task])
            if len(recent_losses[task]) > loss_window_size:
<<<<<<< HEAD
                recent_losses[task].pop(0)#将本次的损失值加到列表中，让列表只保留最近3个轮次的损失值
        # 动态调整任务权重
        if epoch > 0 and epoch % 3 == 0:
            avg_recent_losses = {task: np.mean(losses) for task, losses in recent_losses.items()}#平均三次
            task_weights = adjust_task_weights(avg_recent_losses, task_weights, prev_epoch_losses)#动态调整
            prev_epoch_losses = avg_recent_losses.copy()
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        # 验证阶段
        model.eval()#评估模式
        val_loss = 0
        val_task_losses = {k: 0.0 for k in task_weights}#初始化累加器
        val_batch_count = 0
=======
                recent_losses[task].pop(0)
        
        # 安全地调整任务权重，使用最近几个epoch的平均损失
        if epoch > 0 and epoch % 3 == 0:  # 降低调整频率
            # 计算最近几个epoch的平均损失
            avg_recent_losses = {task: np.mean(losses) for task, losses in recent_losses.items()}
            task_weights = adjust_task_weights(avg_recent_losses, task_weights, prev_epoch_losses)
            prev_epoch_losses = avg_recent_losses.copy()  # 保存当前epoch的损失用于下次比较
        
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
        
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
<<<<<<< HEAD
                weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)#前向传播预测
=======
                
                weather_pred, temp_pred, humidity_pred, precip_pred = model(inputs)
                
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
                weather_target = targets[:, :, 3].long()
                temp_target = targets[:, :, 0]
                humidity_target = targets[:, :, 1]
                precip_target = targets[:, :, 2]
<<<<<<< HEAD
                # 验证集损失
                weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), weather_target.view(-1))
                temp_loss = criterion[1](temp_pred.squeeze(), temp_target)
                humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target)
                # 验证集降水量损失计算
                precip_pred_squeezed = precip_pred.squeeze()
                
                # 确保两者都是相同的形状
                if precip_pred_squeezed.dim() == 3:
                    precip_pred_squeezed = precip_pred_squeezed.squeeze(-1)
                
                # 将两者都展平为一维
                precip_pred_flat = precip_pred_squeezed.contiguous().view(-1)
                precip_target_flat = precip_target.contiguous().view(-1)
                
                # 确保长度一致
                min_len = min(len(precip_pred_flat), len(precip_target_flat))
                precip_pred_flat = precip_pred_flat[:min_len]
                precip_target_flat = precip_target_flat[:min_len]
                
                precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target_flat], device=device)
                # 验证集也使用Huber损失
                precip_loss = F.smooth_l1_loss(precip_pred_flat, precip_target_flat, reduction='none')
                precip_loss = (precip_loss * precip_weights).mean()
                # 天气和降水量的关联性损失
                weather_probs = F.softmax(weather_pred, dim=-1)#分类天气得类别概率
                rain_probs = torch.sum(torch.stack([weather_probs[:, :, i] for i in rain_indices]), dim=0)#下雨类别的概率相加得到下雨概率
                expected_rain_probs = torch.zeros_like(precip_target, device=device)
                batch_size, seq_len = precip_target.shape
                precip_vals = precip_target.view(-1)
                expected_probs = torch.zeros_like(precip_vals, device=device)
                heavy_rain_mask = precip_vals >= 5.0
                light_rain_mask = (precip_vals > 0.0) & (precip_vals < 10.0)#线性增加期望概率
                expected_probs[heavy_rain_mask] = 1.0
                expected_probs[light_rain_mask] = torch.minimum(
                    precip_vals[light_rain_mask] / 5.0 + 0.3,#0.3为概率最小值，避免模型在轻度降雨时不预测降雨
                    torch.tensor(1.0, device=device)
                )
                expected_rain_probs = expected_probs.view(batch_size, seq_len)#1维转2维
                correlation_loss = F.mse_loss(rain_probs, expected_rain_probs)#下雨概率和期望下雨概率之间的差异
                # 范围惩罚
                temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.1
                humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.1
=======
                
                # 计算验证集损失
                weather_loss = criterion[0](weather_pred.view(-1, weather_pred.size(-1)), 
                                          weather_target.view(-1))
                temp_loss = criterion[1](temp_pred.squeeze(), temp_target) 
                humidity_loss = criterion[1](humidity_pred.squeeze(), humidity_target) 
                
                # 确保维度匹配
                precip_pred = precip_pred.squeeze()
                if precip_pred.size(0) != precip_target.size(0):
                    precip_pred = precip_pred[:precip_target.size(0)]
                
                precip_weights = torch.tensor([get_precip_weight(p.item()) for p in precip_target.view(-1)],
                                            device=device)
                precip_loss = criterion[1](precip_pred, precip_target) * precip_weights.mean() 
                
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
                correlation_loss = F.mse_loss(rain_probs, expected_rain_probs)  # 移除0.1缩放
                
                # 添加范围约束惩罚
                temp_range_penalty = torch.mean(F.relu(temp_pred - 40.0) + F.relu(-20.0 - temp_pred)) * 0.1  # 改为0.1
                humidity_range_penalty = torch.mean(F.relu(humidity_pred - 100.0) + F.relu(-humidity_pred)) * 0.1  # 改为0.1
                
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
                loss = (weather_loss * task_weights['weather'] +
                       temp_loss * task_weights['temp'] +
                       humidity_loss * task_weights['humidity'] +
                       precip_loss * task_weights['precip'] +
                       correlation_loss * task_weights['weather_precip_correlation'] +
<<<<<<< HEAD
                       0.01 * (temp_range_penalty + humidity_range_penalty))#计算本batch损失，后累加
=======
                       0.01 * (temp_range_penalty + humidity_range_penalty))  # 使用与训练阶段相同的范围惩罚权重
                
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
                val_loss += loss.item()
                val_task_losses['weather'] += weather_loss.item()
                val_task_losses['temp'] += temp_loss.item()
                val_task_losses['humidity'] += humidity_loss.item()
                val_task_losses['precip'] += precip_loss.item()
<<<<<<< HEAD
                val_task_losses['weather_precip_correlation'] += correlation_loss.item()#总差异
                val_batch_count += 1
        # 计算验证集平均损失
        for task in val_task_losses:
            val_task_losses[task] /= val_batch_count
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
=======
                val_task_losses['weather_precip_correlation'] += correlation_loss.item()
                val_batch_count += 1
        
        # 计算验证集平均损失
        for task in val_task_losses:
            val_task_losses[task] /= val_batch_count
        
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        # 早停检查
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
<<<<<<< HEAD
=======
            
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
            if patience_counter >= patience:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                model.load_state_dict(best_model_state)
                break
<<<<<<< HEAD
=======
        
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {avg_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        print("任务权重:", {k: f"{v:.2f}" for k, v in task_weights.items()})
        print("---")
<<<<<<< HEAD
    # writer.close()  # 移除TensorBoard关闭
    return train_losses, val_losses


def main():
    """
    训练主入口：加载数据、预处理、建模、训练、保存模型与处理器、绘制损失曲线
    """
    # 设置随机种子，保证实验可复现
    torch.manual_seed(42)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'data_w', '2y.csv')
=======
    
    writer.close()
    return train_losses, val_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    current_dir = os.path.dirname(os.path.abspath(__file__))#获取当前文件所在目录的绝对路径
    data_path = os.path.join(current_dir, '..', 'data', 'data_w', '2y.csv')  #构建数据文件的绝对路径
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    print(f"\n正在加载数据文件: {data_path}")
    data = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    # 数据预处理
    print("\n开始数据预处理...")
    processor = WeatherDataProcessor()
<<<<<<< HEAD
    processed_data = processor.fit(data)#拟合编码器和标准器
    processed_data = processor.preprocess_data(data)#预处理
    # 创建序列
    print("\n正在创建序列...")
    sequences, targets = processor.create_sequences(processed_data, input_days=7, output_days=3)
=======
    print("正在拟合处理器...")
    processed_data = processor.fit(data)  # 在训练数据上拟合编码器和标准化器
    print("正在预处理数据...")
    processed_data = processor.preprocess_data(data)  
    # 创建序列
    print("\n正在创建序列...")
    sequences, targets = processor.create_sequences(processed_data, input_days=7, output_days=3)
    
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    # 划分训练集和验证集
    train_size = int(0.8 * len(sequences))
    X_train, X_val = sequences[:train_size], sequences[train_size:]
    y_train, y_val = targets[:train_size], targets[train_size:]
<<<<<<< HEAD
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))#数据转为浮点型
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#打乱数据顺序，提升泛化能力？有必要吗，我们是时序数据
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型参数
    input_size = 36
    hidden_size = 128
    num_layers = 2
    output_size = 72
    weather_categories = 10
    # 初始化模型
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size, weather_categories).to(device)
    # 定义损失函数
    criterion = [
        nn.CrossEntropyLoss(),
        nn.MSELoss()
    ]
    # 定义优化器 - 降低学习率提高稳定性
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
=======
     
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)   
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用cpu
    # 模型参数
    input_size = 36  # 特征维度
    hidden_size = 128
    num_layers = 2  # 从1改为2
    output_size = 72  # 3天 * 24小时
    weather_categories = 10  # 更新天气类别数量为10种
   
    # 初始化模型
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size, weather_categories).to(device)    
    # 定义损失函数
    criterion = [
        nn.CrossEntropyLoss(),  # 天气分类损失
        nn.MSELoss()  # 回归任务损失
    ]  
    # 定义优化器，使用固定学习率0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    # 训练模型
    num_epochs = 100
    print("\n开始训练模型...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
<<<<<<< HEAD
    )
    # 保存模型
    model_path = os.path.join(current_dir, '..', 'data', 'weather_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
=======
    ) 
    # 保存模型
    model_path = os.path.join(current_dir, '..', 'data', 'weather_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")   
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    # 保存处理器
    processor_path = os.path.join(current_dir, '..', 'data', 'weather_processor.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"\n处理器已保存到: {processor_path}")
<<<<<<< HEAD
=======
    
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
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
<<<<<<< HEAD
=======
    
>>>>>>> 69f64fa2f4f09ebc088dc7a8e174736a027c9345
    print("\n训练完成！")

if __name__ == '__main__':
    main() 