# 基于卷积神经网络的无线通信信号自动调制识别

**Automatic Modulation Classification of Wireless Communication Signals Based on Convolutional Neural Networks**

## 1. 项目简介

本项目基于 PyTorch 实现无线通信信号自动调制识别，使用卷积神经网络（CNN）对无线通信 IQ 信号进行分类。

项目主要研究不同信噪比（SNR）条件下，CNN 对不同调制方式信号的识别能力，并通过调整 Dropout 参数，对模型的训练效果和泛化性能进行对比分析。

本项目基于开源项目进行修改和实验扩展。

## 2. 数据集

本项目使用 RadioML2016.10a 数据集。

数据集包含：

- 调制类别：11 类
- 信噪比范围：-20 dB 至 18 dB
- SNR 间隔：2 dB
- 每组数据：1000 条信号
- 单条 IQ 信号形状：2 × 128
- 数据集总样本数：220000

将下载的数据文件放置在项目根目录的 `data` 文件夹中，文件名为：

`RML2016.10a_dict.pkl`

目录结构：

```text
project/
├── data/
│   └── RML2016.10a_dict.pkl
```

数据集文件不包含在本项目仓库中。

## 3. 实验环境

- 操作系统：Windows
- Python：3.11
- 深度学习框架：PyTorch
- GPU：NVIDIA GeForce RTX 3070 Ti Laptop GPU
- CUDA：通过 PyTorch CUDA 版本调用 GPU

## 4. 安装依赖

安装项目所需依赖：

```bash
python -m pip install -r requirements.txt
```

注意：PyTorch 的安装版本需要根据计算机的 CPU、GPU 和 CUDA 环境选择。

## 5. 模型训练

本项目主要使用 VT-CNN2 模型进行调制识别实验。

在项目根目录打开终端，执行：

```bash
python train.py --model vtcnn --epochs 10 --batch_size 64 --split 0.8
```

训练过程中会记录训练损失、验证损失和准确率等指标。

模型权重保存在 `models` 文件夹中，训练日志保存在 `logs` 文件夹中。

## 6. 模型评估

### 6.1 不同 SNR 条件下的识别准确率

运行：

```bash
python evaluate.py
```

该程序统计不同信噪比条件下的识别准确率，并生成相应的 CSV 数据文件。

### 6.2 混淆矩阵分析

运行：

```bash
python evaluate_confusion.py
```

该程序计算各调制类别的识别情况，并生成混淆矩阵 CSV 文件和可视化图片。

### 6.3 绘制 SNR 准确率曲线

运行：

```bash
python plot_snr_curve.py
```

## 7. 实验结果文件

项目中保存了以下实验结果：

- `training_log_10epochs.csv`：基础训练记录
- `training_log_dropout03.csv`：Dropout=0.3 训练记录
- `training_log_dropout05.csv`：Dropout=0.5 训练记录
- `snr_accuracy_dropout03.csv`：Dropout=0.3 的 SNR 测试结果
- `snr_accuracy_dropout05.csv`：Dropout=0.5 的 SNR 测试结果
- `confusion_matrix_dropout03.csv`：Dropout=0.3 的混淆矩阵
- `confusion_matrix_dropout05.csv`：Dropout=0.5 的混淆矩阵

对应的 PNG 图片也保存在项目根目录中。

## 8. 项目结构

```text
pytorch-modulation-recognition/
├── data/                         # 数据集（不上传）
├── logs/                         # TensorBoard 训练日志
├── models/                       # 模型权重
├── torch_modulation_recognition/ # 数据处理、模型及指标定义
├── train.py                      # 模型训练
├── evaluate.py                   # SNR 评估
├── evaluate_confusion.py         # 混淆矩阵计算
├── plot_snr_curve.py             # SNR 曲线绘制
├── read_training_log.py          # 训练日志读取
├── requirements.txt              # Python 依赖
├── README.md                     # 项目说明
└── LICENSE                       # 开源许可证
```

## 9. 原始项目来源与致谢

本项目基于以下开源项目修改：

https://github.com/isaaccorley/pytorch-modulation-recognition

感谢原作者提供的 PyTorch 调制识别实现。

原项目采用 MIT License，本项目保留原始 LICENSE 文件及相关版权信息。