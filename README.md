# 基于卷积神经网络的无线通信信号自动调制识别

**Automatic Modulation Classification of Wireless Communication Signals Based on Convolutional Neural Networks**

GitHub: https://github.com/chuanchuan6396-dot/CNN-Modulation-Recognition

## 1. 项目简介

本项目基于 PyTorch 实现无线通信信号自动调制识别，使用卷积神经网络（CNN）对无线通信 IQ 信号进行分类。

项目主要研究不同信噪比（SNR）条件下，CNN 对不同调制方式信号的识别能力，并通过调整 Dropout 参数，对模型的训练效果和泛化性能进行对比分析。

本项目基于开源项目进行修改，并针对 RadioML2016.10a 数据集进行了适配和实验扩展。

## 2. 数据集

本项目使用 RadioML2016.10a 数据集。

数据集包含：

- 调制类别：11 类
- 信噪比范围：-20 dB 至 18 dB
- SNR 间隔：2 dB
- 每组数据：1000 条信号
- 单条 IQ 信号形状：2 × 128
- 数据集总样本数：220000

数据文件名称：

`RML2016.10a_dict.pkl`

将数据文件放置在项目根目录的 `data` 文件夹中：

```text
project/
├── data/
│   └── RML2016.10a_dict.pkl
```

数据集文件较大，因此不包含在本项目 GitHub 仓库中。

## 3. 实验环境

- 操作系统：Windows
- Python：3.11.9
- 深度学习框架：PyTorch
- GPU：NVIDIA GeForce RTX 3070 Ti Laptop GPU
- CUDA：通过 PyTorch CUDA 版本调用 GPU

## 4. 安装依赖

在项目根目录打开终端，执行：

```bash
python -m pip install -r requirements.txt
```

注意：PyTorch 的安装版本需要根据计算机的 CPU、GPU 和 CUDA 环境选择。

## 5. 模型与对比实验

本项目主要使用 VT-CNN2 卷积神经网络进行调制识别实验。

为研究 Dropout 参数对模型性能的影响，设置三组对比实验：

| 实验组 | Dropout |
|---|---:|
| 实验一 | 0.0 |
| 实验二 | 0.3 |
| 实验三 | 0.5 |

三组实验保持模型结构及其他训练设置一致，仅改变 Dropout 参数，以便比较不同 Dropout 设置下的模型表现。

训练数据与验证数据按照 8:2 的比例划分。

模型权重保存在 `models` 文件夹中：

- `vtcnn.pt`：Dropout=0.0
- `vtcnn_dropout03.pt`：Dropout=0.3
- `vtcnn_dropout05.pt`：Dropout=0.5

训练日志保存在 `logs` 文件夹中。

训练命令示例：

```bash
python train.py --model vtcnn --epochs 10 --batch_size 64 --split 0.8
```

当前训练脚本配置对应 Dropout=0.5 的训练实验。其他两组实验的模型权重也保存在仓库中。

## 6. 模型评估

### 6.1 统一评估

运行：

```bash
python evaluate_all.py
```

该程序使用相同的验证集，对三组模型进行统一评估，统计：

- 各模型总体分类准确率
- 不同 SNR 条件下的识别准确率
- 各调制类别的识别情况
- 三组模型的混淆矩阵

### 6.2 混淆矩阵核查

运行：

```bash
python check_confusion.py
```

该程序计算各调制类别的识别准确率，并检查混淆矩阵计算出的总体准确率是否与统一评估结果一致。

### 6.3 其他评估程序

项目还保留了原有评估程序：

```bash
python evaluate.py
python evaluate_confusion.py
python plot_snr_curve.py
```

分别用于 SNR 评估、混淆矩阵计算和 SNR 准确率曲线绘制。

## 7. 实验结果

### 7.1 不同 Dropout 参数下的总体准确率

三组模型在相同验证集上的总体准确率如下：

| 模型 | Dropout | 总体准确率 |
|---|---:|---:|
| VT-CNN2 | 0.0 | 50.57% |
| VT-CNN2 | 0.3 | 53.21% |
| VT-CNN2 | 0.5 | 50.65% |

其中，Dropout=0.3 的模型取得了最高的总体准确率。

### 7.2 不同 SNR 条件下的识别结果

项目对 -20 dB 至 18 dB 范围内的 20 个 SNR 档位进行了评估。

相关结果文件：

- `unified_snr_accuracy.csv`：三组模型在不同 SNR 下的准确率数据。
- `unified_snr_accuracy.png`：不同 SNR 条件下的准确率对比曲线。

### 7.3 混淆矩阵结果

三组模型均生成了混淆矩阵 CSV 文件和对应的 PNG 可视化图片：

- `unified_confusion_dropout00.csv`
- `unified_confusion_dropout00.png`
- `unified_confusion_dropout03.csv`
- `unified_confusion_dropout03.png`
- `unified_confusion_dropout05.csv`
- `unified_confusion_dropout05.png`

### 7.4 其他实验结果文件

- `unified_evaluation_summary.csv`：统一评估总体结果。
- `training_log_10epochs.csv`：基础训练记录。
- `training_log_dropout03.csv`：Dropout=0.3 训练记录。

## 8. 项目结构

```text
CNN-Modulation-Recognition/
├── data/                              # 数据集（不上传）
├── logs/                              # TensorBoard 训练日志
├── models/                            # 三组模型权重
│   ├── vtcnn.pt
│   ├── vtcnn_dropout03.pt
│   └── vtcnn_dropout05.pt
├── torch_modulation_recognition/      # 数据处理、模型及指标定义
├── train.py                           # 模型训练
├── evaluate.py                        # SNR 评估
├── evaluate_confusion.py              # 混淆矩阵计算
├── evaluate_all.py                    # 三组模型统一评估
├── check_confusion.py                 # 混淆矩阵核查
├── plot_snr_curve.py                  # SNR 曲线绘制
├── read_training_log.py               # 训练日志读取
├── requirements.txt                   # Python 依赖
├── unified_evaluation_summary.csv     # 总体评估结果
├── unified_snr_accuracy.csv          # SNR 准确率数据
├── unified_snr_accuracy.png          # SNR 准确率曲线
├── README.md                          # 项目说明
└── LICENSE                            # 开源许可证
```

## 9. 原始项目来源与致谢

本项目基于以下开源项目修改：

https://github.com/isaaccorley/pytorch-modulation-recognition

感谢原作者提供的 PyTorch 调制识别实现。

原项目采用 MIT License，本项目保留原始 LICENSE 文件及相关版权信息。
