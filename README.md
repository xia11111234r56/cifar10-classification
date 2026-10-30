# CIFAR-10 图像分类实验

在 CIFAR-10 数据集上，用四类神经网络完成图像分类，并对比它们的准确率、参数量与训练时间。

| 网络类型 | 模型 | 准确率 | 参数量 | 训练时间 |
| --- | --- | ---: | ---: | ---: |
| FCN | DeepFCN | 51.84% | 1.74M | 14.8 min |
| CNN | LeNet5 | 60.80% | 62K | 13.4 min |
| CNN | MiniVGG | **87.90%** | 847K | 17.9 min |
| RNN | CNNLSTM | 74.24% | 2.32M | 13.5 min |
| Transformer | CNNTransformer | 79.52% | 210K | 12.4 min |

> 结果来自单卡 RTX 3050（4GB），完整图表见 [`results_gpu/`](results_gpu/)。

## 项目结构

```
cifar10_experiment/
├── config.py          # 训练参数、模型列表、数据路径
├── models.py          # 5 个模型定义 + get_model 工厂
├── train_utils.py     # 数据加载、训练/验证、绘图
├── main.py            # 实验入口
├── requirements.txt
└── results_gpu/       # 训练曲线、准确率对比图与 CSV
```

## 环境

```bash
pip install -r requirements.txt
```

## 数据准备

数据采用 `ImageFolder` 结构，每个类别一个子目录：

```
data/
├── train/
│   ├── airplane/ ...
│   └── truck/ ...
└── test/
    ├── airplane/ ...
    └── truck/ ...
```

默认读取仓库下的 `./data/train` 和 `./data/test`，也可用环境变量覆盖：

```bash
# Linux / macOS
export CIFAR10_TRAIN_DIR=/path/to/train
export CIFAR10_TEST_DIR=/path/to/test

# Windows PowerShell
$env:CIFAR10_TRAIN_DIR="D:\path\to\train"
$env:CIFAR10_TEST_DIR="D:\path\to\test"
```

## 运行

```bash
python main.py
```

训练结果（曲线图、对比图、`detailed_results_*.csv`）会写入 `results_gpu/`。

## 模型

- **DeepFCN**：4 层全连接（1024→512→256→128），带 BatchNorm 与 Dropout
- **LeNet5**：经典卷积网络的小幅改造版
- **MiniVGG**：3 个 VGG 风格卷积块 + 全连接分类头
- **CNNLSTM**：卷积特征提取 + LSTM 序列建模
- **CNNTransformer**：卷积特征 + Transformer 编码器
