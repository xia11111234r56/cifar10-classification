"""
配置文件 - 针对RTX 3050 4GB优化
"""
import torch
import os

class Config:
    # 数据路径：默认用仓库内的相对目录，可用环境变量覆盖
    # 目录需为 ImageFolder 结构：data/train/<类别>/xxx.png
    train_data_path = os.environ.get("CIFAR10_TRAIN_DIR", "./data/train")
    test_data_path = os.environ.get("CIFAR10_TEST_DIR", "./data/test")
    
    # GPU设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU优化参数（针对4GB显存）
    batch_size = 32
    num_workers = 2
    epochs = 25  # 适当减少
    
    # 验证集划分（10%训练集作为验证集）
    validation_split = 0.1
    
    # 4类网络及其包含的模型
    network_types = {
        "FCN": ["DeepFCN"],  # 全连接神经网络
        "CNN": ["LeNet5", "MiniVGG"],  # 卷积神经网络（2种结构）
        "RNN": ["CNNLSTM"],  # 循环神经网络
        "Transformer": ["CNNTransformer"]  # Transformer注意力网络
    }
    
    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 保存路径
    save_dir = "./results_gpu"
    save_figures = True
    
    # 梯度累积步数（防止显存不足）
    accumulation_steps = 2
    
    # 学习率
    learning_rate = 0.001

config = Config()