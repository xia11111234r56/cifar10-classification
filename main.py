"""
主运行文件 - 按4类网络依次训练
"""
import torch
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from config import config
from models import get_model
from train_utils import (
    get_local_dataloaders,
    train_model,
    plot_training_history,
    save_results_summary
)

def print_experiment_info():
    """打印实验信息：展示实验配置、硬件环境和训练计划"""
    print("="*70)  # 打印分隔线，增强可读性
    print("CIFAR-10 图像分类实验")  # 显示实验名称和数据集
    print("="*70)
    print(f"实验要求: 包含4类神经网络")  # 显示实验任务要求
    print(f"  1. 全连接神经网络 (FCN)")  # 第一类网络：全连接神经网络
    print(f"  2. 卷积神经网络 (CNN)")  # 第二类网络：卷积神经网络
    print(f"  3. 循环神经网络 (RNN)")  # 第三类网络：循环神经网络
    print(f"  4. Transformer注意力网络")  # 第四类网络：Transformer网络
    print("="*70)
    print(f"硬件配置:")  # 硬件环境信息标题
    print(f"  设备: {config.device}")  # 显示当前设备（CPU或CUDA）
    # 如果CUDA可用，显示GPU名称；否则显示CPU
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    # 如果CUDA可用，计算并显示显存大小（GB）；否则不显示
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB" if torch.cuda.is_available() else "")
    print(f"训练配置:")  # 训练参数配置标题
    print(f"  批大小: {config.batch_size}")  # 显示批处理大小
    print(f"  训练轮数: {config.epochs}")  # 显示训练总轮数
    print(f"  学习率: {config.learning_rate}")  # 显示初始学习率
    print("="*70)  # 打印分隔线
    
    print("按网络类型依次训练:")  # 训练顺序说明标题
    # 遍历配置中的网络类型字典，依次打印每类网络包含的模型
    for net_type, models in config.network_types.items():
        # 将模型列表连接为字符串，格式如："FCN: DeepFCN"
        print(f"  {net_type}: {', '.join(models)}")
    print("="*70)  # 打印结束分隔线

def train_by_network_type(train_loader, val_loader, classes):
    """按网络类型依次训练：FCN→CNN→RNN→Transformer"""
    results = {}  # 初始化结果字典，用于存储所有网络类型的训练结果
    total_start_time = time.time()  # 记录总体训练开始时间
    
    # 按网络类型顺序训练：遍历配置中的网络类型字典
    for net_type, model_names in config.network_types.items():
        print(f"\n{'='*70}")  # 打印分隔线，突出当前网络类型
        print(f"开始训练 {net_type} 类网络")  # 显示当前训练的网络类型
        print(f"{'='*70}")
        
        type_start_time = time.time()  # 记录当前网络类型的训练开始时间
        type_results = {}  # 初始化当前网络类型的结果字典
        
        # 遍历当前网络类型下的所有模型名称
        for model_name in model_names:
            try:  # 使用try-except块捕获单个模型训练异常
                print(f"\n▶ 训练模型: {model_name}")  # 显示当前训练模型名称
                print(f"{'-'*50}")  # 打印分隔线
                
                # 获取模型实例：通过get_model函数根据模型名称创建模型
                model = get_model(model_name, len(classes))
                
                # 训练模型：调用train_model函数执行完整训练流程
                trained_model, history, test_acc, class_acc, train_time, params = train_model(
                    model, model_name, train_loader, val_loader, config, classes
                )
                
                # 绘制图表：保存训练过程中的损失和准确率曲线图
                history_plot = plot_training_history(history, f"{net_type}_{model_name}", config.save_dir)
                
                # 保存结果：将训练结果存储到字典中
                type_results[model_name] = {
                    'model': trained_model,  # 训练完成的模型实例
                    'test_acc': test_acc,  # 最终验证准确率
                    'class_acc': class_acc,  # 每个类别的准确率
                    'params': params,  # 模型参数量
                    'train_time': train_time,  # 训练耗时（秒）
                    'history': history,  # 训练历史记录（损失和准确率）
                    'history_plot': history_plot,  # 训练曲线图保存路径
                    'network_type': net_type  # 所属网络类型
                }
                
                print(f"✓ {model_name} 训练完成")  # 显示训练成功信息
                
            except Exception as e:  # 捕获训练过程中的异常
                print(f"✗ {model_name} 训练失败: {e}")  # 显示训练失败信息
        
        type_time = time.time() - type_start_time  # 计算当前网络类型的训练总时间
        print(f"\n{net_type} 类网络训练完成!")  # 显示当前网络类型训练完成
        print(f"训练时间: {type_time/60:.1f}分钟")  # 显示训练时间（转换为分钟）
        print(f"包含模型: {list(type_results.keys())}")  # 显示成功训练的模型列表
        
        # 保存该类型的结果：将当前网络类型的结果添加到总结果字典
        results[net_type] = type_results
    
    total_time = time.time() - total_start_time  # 计算总体训练时间
    
    return results, total_time  # 返回所有训练结果和总训练时间

def analyze_results_by_type(results):
    """按网络类型分析结果：结构化展示各网络类型下每个模型的性能指标"""
    print("\n" + "="*70)  # 打印分隔线，突出结果分析开始
    print("按网络类型分析结果")  # 结果显示标题
    print("="*70)
    
    all_results = []  # 初始化列表，用于收集所有模型的详细结果，便于后续转换为DataFrame
    
    # 遍历results字典，net_type为网络类型，type_results为该类型下所有模型的结果字典
    for net_type, type_results in results.items():
        print(f"\n{net_type} 类网络:")  # 打印当前网络类型名称
        print("-"*50)  # 打印分隔线
        
        # 遍历当前网络类型下的每个模型结果
        for model_name, result in type_results.items():
            print(f"  {model_name}:")  # 打印模型名称
            # 打印验证准确率，格式化为两位小数，带百分号
            print(f"    准确率: {result['test_acc']:.2f}%")
            # 打印参数量，使用千位分隔符格式化
            print(f"    参数量: {result['params']:,}")
            # 打印训练时间（转换为分钟），保留一位小数
            print(f"    训练时间: {result['train_time']/60:.1f}分钟")
            
            # 收集所有结果：将每个模型的详细指标添加到列表中，便于后续分析
            all_results.append({
                '网络类型': net_type,  # 网络类型标签
                '模型名称': model_name,  # 模型名称标签
                '准确率': result['test_acc'],  # 模型准确率
                '参数量': result['params'],  # 模型参数量
                '训练时间(分钟)': result['train_time']/60  # 训练时间（分钟）
            })
    
    return pd.DataFrame(all_results)  # 将列表转换为Pandas DataFrame，便于后续处理和分析

def plot_comparison(df, config):
    """绘制对比图表：创建四子图从多个维度对比模型性能"""
    os.makedirs(config.save_dir, exist_ok=True)  # 确保结果保存目录存在，不存在则创建
    
    # 按准确率排序：按准确率从高到低排序DataFrame，便于可视化对比
    df_sorted = df.sort_values('准确率', ascending=False)
    
    # 创建对比图表：创建2行2列的子图，总尺寸15×12英寸
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 所有模型准确率对比（左上子图）
    colors = plt.cm.Set3(range(len(df_sorted)))  # 使用Set3颜色映射为每个模型分配颜色
    bars1 = axes[0, 0].bar(range(len(df_sorted)), df_sorted['准确率'], color=colors)  # 创建柱状图
    axes[0, 0].set_xlabel('模型')  # 设置x轴标签
    axes[0, 0].set_ylabel('准确率 (%)')  # 设置y轴标签
    axes[0, 0].set_title('所有模型准确率对比')  # 设置子图标题
    axes[0, 0].set_xticks(range(len(df_sorted)))  # 设置x轴刻度位置
    # 设置x轴刻度标签：使用网络类型和模型名称作为复合标签，旋转45度，右对齐
    axes[0, 0].set_xticklabels([f"{row['网络类型']}\n{row['模型名称']}" for _, row in df_sorted.iterrows()], 
                               rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)  # 添加网格线，透明度0.3
    
    # 添加数值标签：在每个柱状图上方显示准确率数值
    for bar, acc in zip(bars1, df_sorted['准确率']):
        height = bar.get_height()  # 获取柱状图高度
        # 在柱状图上方添加文本，显示准确率（保留一位小数）
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. 按网络类型平均准确率（右上子图）
    type_avg = df.groupby('网络类型')['准确率'].mean()  # 按网络类型分组计算平均准确率
    colors = plt.cm.Pastel1(range(len(type_avg)))  # 使用Pastel1颜色映射
    bars2 = axes[0, 1].bar(range(len(type_avg)), type_avg.values, color=colors)  # 创建柱状图
    axes[0, 1].set_xlabel('网络类型')  # 设置x轴标签
    axes[0, 1].set_ylabel('平均准确率 (%)')  # 设置y轴标签
    axes[0, 1].set_title('按网络类型平均准确率')  # 设置子图标题
    axes[0, 1].set_xticks(range(len(type_avg)))  # 设置x轴刻度位置
    axes[0, 1].set_xticklabels(type_avg.index)  # 设置x轴刻度标签为网络类型名称
    axes[0, 1].grid(True, alpha=0.3)  # 添加网格线
    
    # 添加数值标签：在每个柱状图上方显示平均准确率
    for bar, acc in zip(bars2, type_avg.values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. 参数量对比（对数坐标）（左下子图）
    axes[1, 0].bar(range(len(df_sorted)), df_sorted['参数量'], color=colors)  # 创建柱状图
    axes[1, 0].set_xlabel('模型')  # 设置x轴标签
    axes[1, 0].set_ylabel('参数量')  # 设置y轴标签
    axes[1, 0].set_title('模型参数量对比')  # 设置子图标题
    axes[1, 0].set_xticks(range(len(df_sorted)))  # 设置x轴刻度位置
    # 设置x轴刻度标签：使用网络类型和模型名称作为复合标签
    axes[1, 0].set_xticklabels([f"{row['网络类型']}\n{row['模型名称']}" for _, row in df_sorted.iterrows()], 
                               rotation=45, ha='right')
    axes[1, 0].set_yscale('log')  # 设置y轴为对数坐标，便于展示参数量级的差异
    axes[1, 0].grid(True, alpha=0.3)  # 添加网格线
    
    # 4. 准确率 vs 训练时间散点图（右下子图）
    scatter = axes[1, 1].scatter(df['训练时间(分钟)'], df['准确率'], 
                                 c=range(len(df)), cmap='viridis', s=100, alpha=0.7)  # 创建散点图
    axes[1, 1].set_xlabel('训练时间 (分钟)')  # 设置x轴标签
    axes[1, 1].set_ylabel('准确率 (%)')  # 设置y轴标签
    axes[1, 1].set_title('准确率 vs 训练时间')  # 设置子图标题
    axes[1, 1].grid(True, alpha=0.3)  # 添加网格线
    
    # 添加标签：在每个散点旁标注模型名称
    for i, row in df.iterrows():
        axes[1, 1].annotate(f"{row['模型名称']}", 
                           (row['训练时间(分钟)'], row['准确率']),  # 标签位置
                           textcoords="offset points",  # 文本坐标偏移模式
                           xytext=(0,10),  # 文本偏移量（向上偏移10点）
                           ha='center',  # 水平居中对齐
                           fontsize=8)  # 字体大小
    
    plt.tight_layout()  # 自动调整子图布局，防止标签重叠
    
    # 保存图表：使用时间戳生成唯一文件名，保存为PNG格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{config.save_dir}/model_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')  # 保存图像，DPI150，裁剪空白
    plt.close()  # 关闭图形，释放内存
    
    print(f"\n对比图表已保存到: {plot_path}")  # 打印图表保存路径
    
    return plot_path  # 返回保存的图表文件路径

def save_detailed_results(df, config):
    """保存详细结果：将DataFrame结果保存为CSV文件，并在控制台格式化输出"""
    os.makedirs(config.save_dir, exist_ok=True)  # 创建结果保存目录，如果目录已存在则不会引发错误
    
    # 保存为CSV：生成带有时间戳的唯一文件名，避免覆盖先前的结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成格式为"年月日_时分秒"的时间戳
    csv_path = f"{config.save_dir}/detailed_results_{timestamp}.csv"  # 构建CSV文件完整路径
    df.to_csv(csv_path, index=False)  # 将DataFrame保存为CSV文件，不包含行索引
    
    # 打印汇总表：在控制台以格式化表格形式展示所有结果
    print("\n" + "="*70)  # 打印分隔线
    print("实验结果详细汇总")  # 结果汇总标题
    print("="*70)
    print(df.to_string(index=False))  # 将DataFrame转换为字符串格式并打印，不显示行索引
    print("="*70)  # 打印结束分隔线
    print(f"详细结果已保存到: {csv_path}")  # 显示CSV文件保存路径
    
    return csv_path  # 返回CSV文件路径，便于后续引用

def main():
    """主函数：实验入口点，控制整个实验流程的执行顺序"""
    # 打印实验信息：显示实验配置、硬件环境和训练计划
    print_experiment_info()
    
    # 检查GPU：检测CUDA是否可用，决定使用GPU还是CPU进行训练
    if not torch.cuda.is_available():  # 如果CUDA不可用
        print("警告: GPU不可用，将使用CPU训练（训练会较慢）")  # 显示警告信息
    else:  # 如果CUDA可用
        print(f"✓ GPU加速已启用")  # 显示GPU加速已启用
    
    # 加载数据：调用数据加载函数获取训练、验证、测试数据加载器和类别信息
    print("\n正在加载数据集...")  # 显示加载数据进度
    train_loader, val_loader, test_loader, classes = get_local_dataloaders(config)
    
    # 检查数据加载是否成功：如果train_loader为None，说明数据加载失败
    if train_loader is None:
        print("数据加载失败，请检查数据路径！")  # 显示错误信息
        return  # 退出主函数
    
    # 按网络类型依次训练：尝试执行主要训练流程，包含异常处理
    try:
        # 调用训练函数，按网络类型顺序训练所有模型
        results, total_time = train_by_network_type(train_loader, val_loader, classes)
        
        # 检查是否有模型训练成功：如果results为空字典，表示没有模型训练成功
        if not results:
            print("没有模型训练成功")  # 显示错误信息
            return  # 退出主函数
        
        # 分析结果：按网络类型组织并分析训练结果
        df = analyze_results_by_type(results)
        
        # 绘制对比图表：生成四子图对比分析图表
        plot_path = plot_comparison(df, config)
        
        # 保存详细结果：将结果保存为CSV文件
        csv_path = save_detailed_results(df, config)
        
        # 输出最佳模型：从DataFrame中找到准确率最高的模型
        best_model = df.loc[df['准确率'].idxmax()]  # 使用idxmax找到准确率最大值的索引
        
        # 实验总结输出：显示完整实验总结信息
        print("\n" + "="*70)  # 打印分隔线
        print("🎉 实验完成!")  # 显示实验完成信息（带庆祝表情）
        print("="*70)
        print("实验总结:")  # 实验总结标题
        print(f"  总训练时间: {total_time/60:.1f}分钟")  # 显示总训练时间（转换为分钟）
        print(f"  训练模型总数: {len(df)}")  # 显示成功训练的模型数量
        print(f"  网络类型: {len(config.network_types)}类")  # 显示涉及的网络类型数量
        print()  # 空行分隔
        print("最佳模型:")  # 最佳模型信息标题
        print(f"  网络类型: {best_model['网络类型']}")  # 显示最佳模型的网络类型
        print(f"  模型名称: {best_model['模型名称']}")  # 显示最佳模型名称
        print(f"  准确率: {best_model['准确率']:.2f}%")  # 显示最佳模型准确率
        print(f"  参数量: {best_model['参数量']:,}")  # 显示最佳模型参数量（千位分隔）
        print(f"  训练时间: {best_model['训练时间(分钟)']:.1f}分钟")  # 显示最佳模型训练时间
        print()  # 空行分隔
        print("结果文件:")  # 结果文件信息标题
        print(f"  对比图表: {plot_path}")  # 显示对比图表保存路径
        print(f"  详细数据: {csv_path}")  # 显示详细数据CSV文件路径
        print("="*70)  # 打印结束分隔线
        
        return results, df  # 返回训练结果和DataFrame，便于后续分析或调试
        
    except KeyboardInterrupt:  # 捕获用户中断（Ctrl+C）异常
        print("\n\n实验被用户中断")  # 显示实验被用户中断信息
    except Exception as e:  # 捕获其他所有异常
        print(f"\n\n实验出错: {e}")  # 显示错误信息
        import traceback  # 导入traceback模块
        traceback.print_exc()  # 打印详细的异常堆栈信息，便于调试
    
    return None, None  # 如果发生异常，返回None值

if __name__ == "__main__":
    main()