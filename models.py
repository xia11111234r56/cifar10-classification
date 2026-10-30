"""
5个核心神经网络模型定义
针对RTX 3050 4GB显存优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFCN(nn.Module):
    """深度全连接网络（带BN）- 预计GPU时间: 12-15分钟"""
    def __init__(self, input_dim=3072, hidden_dims=[512, 256, 128], num_classes=10):
        super(DeepFCN, self).__init__()  # 调用父类nn.Module的初始化方法
        
        # 初始化网络层列表，用于存储所有网络层
        layers = []
        prev_dim = input_dim  # 前一层维度，初始为输入维度3072（32*32*3）
        
        # 循环构建隐藏层：每个隐藏层包含线性层、批量归一化、ReLU激活和Dropout
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层：将前一层的输出维度映射到当前隐藏层维度
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # 批量归一化层：加速训练收敛，提高训练稳定性
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU激活函数：引入非线性，增强模型表达能力
            layers.append(nn.ReLU())
            # Dropout层：随机丢弃30%神经元，防止过拟合
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim  # 更新前一层维度为当前隐藏层维度
        
        # 添加输出层：将最后一个隐藏层的输出映射到类别数（10个类别）
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 使用Sequential容器按顺序组合所有层，形成完整网络
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 前向传播函数：定义数据在网络中的流向
        x = x.view(x.size(0), -1)  # 将输入图像展平为一维向量，保持batch维度不变
        return self.model(x)  # 依次通过所有网络层，得到最终输出

class LeNet5(nn.Module):
    """LeNet-5 CNN结构 - 预计GPU时间: 10-15分钟"""
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()  # 调用父类nn.Module的初始化方法
        # 第一个卷积层：输入通道3（RGB图像），输出通道6，卷积核大小5×5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # 第一个池化层：2×2最大池化，步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层：输入通道6，输出通道16，卷积核大小5×5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二个池化层：2×2最大池化，步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)       
        # 第一个全连接层：输入维度16*5*5（经过两次池化后的特征图尺寸），输出维度120
        self.fc1 = nn.Linear(16*5*5, 120)
        # 第二个全连接层：输入维度120，输出维度84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层（输出层）：输入维度84，输出维度num_classes（10个类别）
        self.fc3 = nn.Linear(84, num_classes)       
        # ReLU激活函数实例，避免重复创建
        self.relu = nn.ReLU()  
    def forward(self, x):
        # 第一层卷积+激活+池化
        x = self.relu(self.conv1(x))  # 卷积后应用ReLU激活
        x = self.pool1(x)            # 最大池化降维
        
        # 第二层卷积+激活+池化
        x = self.relu(self.conv2(x))  # 卷积后应用ReLU激活
        x = self.pool2(x)            # 最大池化降维
        
        # 展平操作：将多维特征图转换为一维向量，保留batch维度
        x = x.view(x.size(0), -1)
        
        # 第一个全连接层+激活
        x = self.relu(self.fc1(x))
        # 第二个全连接层+激活
        x = self.relu(self.fc2(x))
        # 第三个全连接层（输出层），不使用激活函数（配合交叉熵损失函数）
        x = self.fc3(x)
        return x


class MiniVGG(nn.Module):
    """微型VGG网络（4GB显存优化版）- 预计GPU时间: 20-25分钟"""
    def __init__(self, num_classes=10):
        super(MiniVGG, self).__init__()  # 继承nn.Module基类
        
        # Block 1（减小通道数以适应4GB显存）：两个3×3卷积层，通道数3→32→32，保持空间尺寸不变（padding=1）
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 第一卷积层
        self.bn1_1 = nn.BatchNorm2d(32)  # 批量归一化层，加速收敛
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 第二卷积层
        self.bn1_2 = nn.BatchNorm2d(32)  # 批量归一化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2最大池化，空间尺寸减半
        
        # Block 2：两个3×3卷积层，通道数32→64→64
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第三卷积层
        self.bn2_1 = nn.BatchNorm2d(64)  # 批量归一化层
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 第四卷积层
        self.bn2_2 = nn.BatchNorm2d(64)  # 批量归一化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2最大池化
        
        # Block 3：两个3×3卷积层，通道数64→128→128
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 第五卷积层
        self.bn3_1 = nn.BatchNorm2d(128)  # 批量归一化层
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 第六卷积层
        self.bn3_2 = nn.BatchNorm2d(128)  # 批量归一化层
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2最大池化
        
        # 全连接层：将特征图展平后通过三个全连接层进行分类
        self.fc1 = nn.Linear(128*4*4, 256)  # 输入维度128×4×4（经过三次池化后尺寸），输出256维
        self.bn_fc1 = nn.BatchNorm1d(256)  # 全连接层批量归一化
        self.dropout1 = nn.Dropout(0.5)  # 50% Dropout防止过拟合
        self.fc2 = nn.Linear(256, 128)  # 第二全连接层
        self.bn_fc2 = nn.BatchNorm1d(128)  # 批量归一化
        self.dropout2 = nn.Dropout(0.5)  # 50% Dropout
        self.fc3 = nn.Linear(128, num_classes)  # 输出层，10个类别
        
        self.relu = nn.ReLU()  # ReLU激活函数实例化，避免重复创建
    
    def forward(self, x):
        # Block 1前向传播：卷积→批归一化→激活→卷积→批归一化→激活→池化
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        # Block 2前向传播
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        # Block 3前向传播
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        # 展平特征图：将四维张量[B, C, H, W]转为二维张量[B, C*H*W]
        x = x.view(x.size(0), -1)
        # 全连接层前向传播：线性变换→批归一化→激活→Dropout
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)  # 输出层，不使用激活函数（配合交叉熵损失）
        return x


class CNNLSTM(nn.Module):
    """CNN + LSTM混合网络 - 预计GPU时间: 15-20分钟"""
    def __init__(self, num_classes=10):
        super(CNNLSTM, self).__init__()  # 继承nn.Module基类
        
        # CNN特征提取部分：使用Sequential容器顺序组织CNN层
        self.cnn = nn.Sequential(
            # 第一卷积层：输入3通道（RGB），输出32通道，3×3卷积核，保持空间尺寸（padding=1）
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # 批量归一化，加速训练收敛
            nn.ReLU(),           # ReLU激活函数，引入非线性
            nn.MaxPool2d(2, 2),  # 2×2最大池化，空间尺寸减半（32×32→16×16）
            
            # 第二卷积层：输入32通道，输出64通道，3×3卷积核
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),           # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 2×2最大池化，空间尺寸再减半（16×16→8×8）
        )
        
        # LSTM部分：处理CNN提取的特征序列
        self.lstm = nn.LSTM(
            input_size=64*8*8,    # 输入特征维度：64通道×8×8空间尺寸=4096维
            hidden_size=128,      # LSTM隐藏状态维度：128
            num_layers=2,         # LSTM堆叠层数：2层
            batch_first=True,     # 输入输出张量格式为[batch, seq_len, features]
            dropout=0.3           # 层间Dropout率：0.3，防止过拟合
        )
        
        # 全连接分类层：将LSTM输出映射到类别空间
        self.fc = nn.Linear(128, num_classes)  # 输入128维，输出10个类别
    
    def forward(self, x):
        batch_size = x.size(0)  # 获取当前批次大小
        
        # CNN特征提取：输入[B,3,32,32] → 输出[B,64,8,8]
        cnn_features = self.cnn(x)
        
        # 特征图重塑：将四维特征图转换为三维序列[B,1,4096]
        # -1表示自动计算该维度大小（保持总元素数不变）
        cnn_features = cnn_features.view(batch_size, 1, -1)
        
        # LSTM处理：输入[B,1,4096] → 输出lstm_out[B,1,128]
        # lstm_out包含所有时间步的输出，hidden/cell为最后时间步的隐藏/细胞状态
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # 取最后一个时间步的输出：[B,1,128]中取索引-1 → [B,128]
        last_output = lstm_out[:, -1, :]
        
        # 分类输出：[B,128] → [B,10]
        output = self.fc(last_output)
        
        return output


class CNNTransformer(nn.Module):
    """CNN + Transformer混合网络 - 预计GPU时间: 18-25分钟"""
    def __init__(self, num_classes=10):
        super(CNNTransformer, self).__init__()  # 调用父类nn.Module初始化
        
        # CNN特征提取部分：使用Sequential容器顺序组织两层卷积块
        self.cnn = nn.Sequential(
            # 第一层卷积：输入3通道（RGB），输出64通道，3×3卷积核，padding=1保持尺寸
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 批量归一化层，稳定训练过程
            nn.ReLU(),           # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 2×2最大池化，尺寸减半（32×32→16×16）
            
            # 第二层卷积：输入64通道，输出128通道，3×3卷积核
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # 批量归一化
            nn.ReLU(),            # ReLU激活
            nn.MaxPool2d(2, 2),   # 2×2最大池化，尺寸再减半（16×16→8×8）
        )
        
        # Transformer编码器（简化版，减少内存占用）
        # 创建Transformer编码器层，定义基本参数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,           # 输入特征维度（与CNN输出通道数一致）
            nhead=4,               # 注意力头数，4头注意力
            dim_feedforward=256,   # 前馈网络隐藏层维度
            dropout=0.1,           # Dropout率，防止过拟合
            batch_first=True       # 输入输出格式为[batch, seq, feature]
        )
        # 使用单个Transformer编码器层构建Transformer编码器
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 全连接分类层：将Transformer输出映射到类别空间
        self.fc = nn.Linear(128, num_classes)  # 输入128维，输出10个类别
    
    def forward(self, x):
        batch_size = x.size(0)  # 获取批次大小
        x = self.cnn(x)  # CNN特征提取：[B, 3, 32, 32] → [B, 128, 8, 8]
        
        # 将特征图转换为序列：将空间维度展平为序列长度
        batch_size, channels, height, width = x.size()  # 获取特征图形状
        x = x.view(batch_size, channels, -1)  # 展平空间维度：[B, 128, 64]
        x = x.transpose(1, 2)  # 调整维度顺序：[B, 64, 128]，64为序列长度，128为特征维度
        
        # Transformer处理：通过注意力机制建模全局依赖关系
        x = self.transformer(x)  # [B, 64, 128] → [B, 64, 128]
        
        # 全局平均池化：沿序列维度取平均，聚合所有位置信息
        x = x.mean(dim=1)  # [B, 64, 128] → [B, 128]
        
        # 分类：通过全连接层输出分类结果
        output = self.fc(x)  # [B, 128] → [B, 10]
        
        return output


def get_model(model_name, num_classes=10):
    """根据名称获取模型实例"""
    models = {
        "DeepFCN": DeepFCN,
        "LeNet5": LeNet5,
        "MiniVGG": MiniVGG,
        "CNNLSTM": CNNLSTM,
        "CNNTransformer": CNNTransformer
    }
    
    if model_name in models:
        return models[model_name](num_classes=num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")