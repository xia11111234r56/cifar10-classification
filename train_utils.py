"""
GPU优化训练工具
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import time
import gc

def get_local_dataloaders(config):
    """从本地文件夹加载数据"""
    print(f"加载训练集: {config.train_data_path}")
    print(f"加载测试集: {config.test_data_path}")
    
    # 检查路径
    if not os.path.exists(config.train_data_path):
        print(f"错误: 训练集路径不存在: {config.train_data_path}")
        return None, None, None, None
    
    # 数据增强（训练集）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载训练集
    train_dataset = ImageFolder(config.train_data_path, transform=train_transform)
    
    # 加载测试集
    test_dataset = ImageFolder(config.test_data_path, transform=test_transform)
    
    # 划分训练集和验证集
    train_size = int((1 - config.validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器（使用pin_memory加速GPU）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True  # GPU加速
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"训练集: {train_size} 张图片")
    print(f"验证集: {val_size} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"类别: {config.class_names}")
    
    return train_loader, val_loader, test_loader, config.class_names

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps=2):
    """训练一个epoch（带梯度累积）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps  # 归一化损失
        
        # 反向传播
        loss.backward()
        
        # 每accumulation_steps步更新一次（节省显存）
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item() * accumulation_steps:.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    # 处理剩余的梯度
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def test_model(model, test_loader, device, classes):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每个类别的统计
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_acc = 100. * correct / total
    
    # 每个类别的准确率
    class_acc = []
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_acc.append(acc)
        else:
            class_acc.append(0)
    
    return overall_acc, class_acc

def train_model(model, model_name, train_loader, val_loader, config, classes):
    """训练一个模型"""
    print(f"\n训练模型: {model_name}")
    print(f"模型结构: {model.__class__.__name__}")
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 移动模型到设备
    model = model.to(config.device)
    
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate,
                          weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # 最佳模型跟踪
    best_val_acc = 0.0
    best_model_state = None
    
    # 早停参数
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(config.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch, config.accumulation_steps
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"   ↑ 新的最佳验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   → 早停触发，停止训练")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试模型（使用验证集作为最终评估）
    test_acc, class_acc = test_model(model, val_loader, config.device, classes)
    
    train_time = time.time() - start_time
    
    print(f"\n训练完成 - {model_name}:")
    print(f"  训练时间: {train_time/60:.1f}分钟")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  最终验证准确率: {test_acc:.2f}%")
    
    return model, history, test_acc, class_acc, train_time, params

def plot_training_history(history, model_name, save_dir):
    """绘制训练历史图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{model_name}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def save_results_summary(results, config):
    """保存结果汇总表"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    summary = []
    for model_name, result in results.items():
        summary.append({
            'Model': model_name,
            'Parameters': result['params'],
            'Best_Val_Acc': f"{result['test_acc']:.2f}%",
            'Train_Time(min)': f"{result['train_time']/60:.1f}",
            'Epochs_Trained': len(result['history']['train_loss'])
        })
    
    df = pd.DataFrame(summary)
    
    # 按准确率排序
    df = df.sort_values('Best_Val_Acc', ascending=False)
    
    # 保存为CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{config.save_dir}/results_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # 打印表格
    print("\n" + "="*70)
    print("实验结果汇总:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    print(f"结果已保存到: {csv_path}")
    
    return df, csv_path