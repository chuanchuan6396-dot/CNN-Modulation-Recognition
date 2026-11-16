
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import torch_modulation_recognition as tmr


# ==========================================
# 1. 实验参数
# ==========================================

BATCH_SIZE = 256
SPLIT = 0.8
N_CLASSES = 11

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODELS = [
    {
        "dropout": 0.0,
        "path": "models/vtcnn.pt",
        "name": "Dropout=0.0",
        "tag": "00"
    },
    {
        "dropout": 0.3,
        "path": "models/vtcnn_dropout03.pt",
        "name": "Dropout=0.3",
        "tag": "03"
    },
    {
        "dropout": 0.5,
        "path": "models/vtcnn_dropout05.pt",
        "name": "Dropout=0.5",
        "tag": "05"
    }
]

CLASS_NAMES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM"
]


# ==========================================
# 2. 固定随机种子
# 与 train.py 保持一致
# ==========================================

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================
# 3. 初始化参考模型
# 必须在数据集划分之前
# ==========================================

print("正在初始化参考模型...")

reference_model = tmr.models.VT_CNN2(
    n_classes=N_CLASSES,
    dropout=0.5
)


# ==========================================
# 4. 加载数据集并划分验证集
# 与 train.py 使用相同的顺序
# ==========================================

print("正在读取数据集...")

dataset = tmr.data.RadioML2016()

total_size = len(dataset)
train_size = int(total_size * SPLIT)
val_size = total_size - train_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size]
)

print("数据集总数：", total_size)
print("训练集样本数：", len(train_set))
print("验证集样本数：", len(val_set))


# 获取验证集每条样本对应的 SNR
val_indices = val_set.indices

val_snrs = np.array([
    dataset.y[idx][1]
    for idx in val_indices
])

# 验证集 DataLoader
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 5. 检查模型文件
# ==========================================

for item in MODELS:
    if not os.path.exists(item["path"]):
        raise FileNotFoundError(
            "没有找到模型文件：" + item["path"]
        )

print("\n三个模型文件均已找到。")


# ==========================================
# 6. 统一评估三组模型
# ==========================================

summary_results = []
snr_results = []

for item in MODELS:

    dropout = item["dropout"]
    model_path = item["path"]
    model_name = item["name"]
    tag = item["tag"]

    print("\n" + "=" * 50)
    print("开始评估：", model_name)
    print("=" * 50)

    # 初始化模型
    model = tmr.models.VT_CNN2(
        n_classes=N_CLASSES,
        dropout=dropout
    ).to(DEVICE)

    # 加载模型权重
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("模型权重加载成功：", model_path)

    # 混淆矩阵
    confusion_matrix = np.zeros(
        (N_CLASSES, N_CLASSES),
        dtype=np.int64
    )

    # 保存预测结果
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    # 在同一验证集上进行预测
    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch_y.cpu().numpy()
            )

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 总体准确率
    overall_accuracy = correct / total * 100

    print(
        "\n验证集总体准确率："
        "{:.2f}%".format(overall_accuracy)
    )

    summary_results.append({
        "Dropout": dropout,
        "Model": model_path,
        "Samples": total,
        "Accuracy": overall_accuracy
    })

    # ======================================
    # 7. 计算各 SNR 等级准确率
    # ======================================

    print("\n各 SNR 等级准确率：")

    for snr in tmr.data.SNRS:

        mask = (val_snrs == snr)

        snr_labels = all_labels[mask]
        snr_predictions = all_predictions[mask]

        snr_total = len(snr_labels)

        snr_correct = np.sum(
            snr_labels == snr_predictions
        )

        snr_accuracy = (
            snr_correct / snr_total * 100
            if snr_total > 0 else 0
        )

        print(
            "SNR = {:3d} dB | "
            "样本数 = {:5d} | "
            "准确率 = {:.2f}%".format(
                snr,
                snr_total,
                snr_accuracy
            )
        )

        snr_results.append({
            "SNR": snr,
            "Dropout": dropout,
            "Samples": snr_total,
            "Accuracy": snr_accuracy
        })

    # ======================================
    # 8. 计算混淆矩阵
    # ======================================

    for true_label, pred_label in zip(
        all_labels,
        all_predictions
    ):
        confusion_matrix[
            true_label,
            pred_label
        ] += 1

    # 保存混淆矩阵 CSV
    csv_path = (
        "unified_confusion_dropout"
        + tag + ".csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["True/Pred"] + CLASS_NAMES
        )

        for i in range(N_CLASSES):
            writer.writerow(
                [CLASS_NAMES[i]]
                + confusion_matrix[i].tolist()
            )

    # 计算归一化混淆矩阵
    row_sums = confusion_matrix.sum(
        axis=1,
        keepdims=True
    )

    normalized_matrix = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros(
            (N_CLASSES, N_CLASSES),
            dtype=float
        ),
        where=row_sums != 0
    ) * 100

    # 绘制混淆矩阵
    plt.figure(figsize=(11, 9))

    plt.imshow(
        normalized_matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=100
    )

    plt.colorbar(label="Accuracy (%)")

    plt.xticks(
        range(N_CLASSES),
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(N_CLASSES),
        CLASS_NAMES
    )

    plt.xlabel("Predicted Modulation")
    plt.ylabel("True Modulation")
    plt.title(
        "Normalized Confusion Matrix - "
        + model_name
    )

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            plt.text(
                j,
                i,
                "{:.1f}".format(
                    normalized_matrix[i, j]
                ),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()

    png_path = (
        "unified_confusion_dropout"
        + tag + ".png"
    )

    plt.savefig(
        png_path,
        dpi=300
    )

    plt.close()

    print("混淆矩阵已保存：", csv_path)
    print("混淆矩阵图片已保存：", png_path)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ==========================================
# 9. 保存总体准确率
# ==========================================

with open(
    "unified_evaluation_summary.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Dropout",
            "Model",
            "Samples",
            "Accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(summary_results)


# ==========================================
# 10. 保存 SNR 对照结果
# ==========================================

with open(
    "unified_snr_accuracy.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "SNR",
            "Dropout",
            "Samples",
            "Accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(snr_results)


# ==========================================
# 11. 绘制三组模型 SNR 对比曲线
# ==========================================

plt.figure(figsize=(10, 6))

for item in MODELS:

    current = [
        row for row in snr_results
        if row["Dropout"] == item["dropout"]
    ]

    current.sort(
        key=lambda x: x["SNR"]
    )

    snrs = [
        row["SNR"]
        for row in current
    ]

    accuracies = [
        row["Accuracy"]
        for row in current
    ]

    plt.plot(
        snrs,
        accuracies,
        marker="o",
        linewidth=2,
        label=item["name"]
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("CNN Modulation Recognition Accuracy vs SNR")

plt.xticks(tmr.data.SNRS)
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "unified_snr_accuracy.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. 输出最终结果
# ==========================================

print("\n" + "=" * 50)
print("三组模型统一评估完成！")
print("=" * 50)

for item in summary_results:
    print(
        "Dropout={:.1f} | "
        "总体准确率={:.2f}%".format(
            item["Dropout"],
            item["Accuracy"]
        )
    )

print("\n结果文件：")
print("1. unified_evaluation_summary.csv")
print("2. unified_snr_accuracy.csv")
print("3. unified_snr_accuracy.png")
print("4. unified_confusion_dropout00.csv/png")
print("5. unified_confusion_dropout03.csv/png")
print("6. unified_confusion_dropout05.csv/png")

print("\n所有实验结果均来自同一份验证集。")
