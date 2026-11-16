
import os
import csv
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split, Subset
import torch_modulation_recognition as tmr


# =========================
# 1. 实验参数
# =========================

BATCH_SIZE = 64
SPLIT = 0.8
N_CLASSES = 11
DROPOUT = 0.5

MODEL_PATH = os.path.join("models", "vtcnn_dropout05.pt")
OUTPUT_CSV = "snr_accuracy_dropout05.csv"
OUTPUT_PNG = "snr_accuracy_curve.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 固定随机种子
# =========================

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================
# 3. 创建模型
# 注意：必须在 random_split 之前创建
# =========================

print("正在初始化模型...")

net = tmr.models.VT_CNN2(
    n_classes=N_CLASSES,
    dropout=DROPOUT
)


# =========================
# 4. 加载数据集并复现验证集
# =========================

print("正在读取数据集...")

dataset = tmr.data.RadioML2016()

total = len(dataset)

lengths = [
    int(total * SPLIT),
    total - int(total * SPLIT)
]

train_set, val_set = random_split(dataset, lengths)

print("数据集总数：", total)
print("验证集样本数：", len(val_set))


# =========================
# 5. 加载最佳模型
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "没有找到模型文件：" + MODEL_PATH
    )

print("正在加载模型权重...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

# 兼容不同形式的模型权重文件
if isinstance(checkpoint, dict):
    for key in ["state_dict", "model_state_dict", "network", "model"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

# 处理可能存在的 DataParallel 前缀
if isinstance(checkpoint, dict):
    checkpoint = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }

net.load_state_dict(checkpoint)

net = net.to(DEVICE)
net.eval()

print("模型加载成功！")
print("当前设备：", DEVICE)


# =========================
# 6. 按SNR计算验证集准确率
# =========================

results = []

print("\n========== 不同SNR下的识别准确率 ==========")

for snr in tmr.data.SNRS:

    # 找到验证集中属于当前SNR的样本
    snr_indices = [
        idx for idx in val_set.indices
        if dataset.y[idx][1] == snr
    ]

    snr_set = Subset(dataset, snr_indices)

    loader = DataLoader(
        snr_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    correct = 0
    total_samples = 0

    with torch.no_grad():
        for signals, labels in loader:

            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(signals)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (
        correct / total_samples * 100
        if total_samples > 0 else 0
    )

    results.append({
        "SNR": snr,
        "Samples": total_samples,
        "Accuracy": accuracy
    })

    print(
        f"SNR = {snr:3d} dB | "
        f"样本数 = {total_samples:5d} | "
        f"准确率 = {accuracy:.2f}%"
    )


# =========================
# 7. 计算整体准确率
# =========================

all_samples = sum(item["Samples"] for item in results)

overall_correct = sum(
    item["Accuracy"] * item["Samples"] / 100
    for item in results
)

overall_accuracy = (
    overall_correct / all_samples * 100
    if all_samples > 0 else 0
)

print("\n整体验证集准确率：{:.2f}%".format(overall_accuracy))


# =========================
# 8. 保存CSV结果
# =========================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["SNR", "Samples", "Accuracy"]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nCSV结果已保存：", OUTPUT_CSV)


# =========================
# 9. 绘制SNR准确率曲线
# =========================

try:
    import matplotlib.pyplot as plt

    snrs = [item["SNR"] for item in results]
    accuracies = [item["Accuracy"] for item in results]

    plt.figure(figsize=(10, 6))

    plt.plot(
        snrs,
        accuracies,
        marker="o",
        linewidth=2
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Modulation Recognition Accuracy vs SNR")

    plt.xticks(snrs)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print("SNR准确率曲线已保存：", OUTPUT_PNG)

except ImportError:
    print("没有安装 matplotlib，暂时跳过曲线绘制。")
    print("CSV结果已经保存，可以后续再绘图。")


print("\n========== SNR测试完成 ==========")
