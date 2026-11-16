

import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import torch_modulation_recognition as tmr


# 1. 固定随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 加载数据集
print("正在加载数据集……")

dataset = tmr.data.RadioML2016()

total_size = len(dataset)
train_size = int(total_size * 0.8)
val_size = total_size - train_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size]
)

print("验证集样本数：", len(val_set))


# 3. 加载 Dropout=0.5 模型
print("正在加载 Dropout=0.5 模型……")

model = tmr.models.VT_CNN2(
    n_classes=11,
    dropout=0.5
).to(device)

checkpoint = torch.load(
    "models/vtcnn_dropout05.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("模型加载成功！")


# 4. 准备调制类别
class_names = [
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

num_classes = len(class_names)

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)


# 5. 在验证集上进行预测
print("开始计算混淆矩阵……")

val_loader = DataLoader(
    val_set,
    batch_size=256,
    shuffle=False
)

with torch.no_grad():
    for batch_x, batch_y in val_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        predictions = torch.argmax(outputs, dim=1)

        for true_label, pred_label in zip(
            batch_y.cpu().numpy(),
            predictions.cpu().numpy()
        ):
            confusion_matrix[true_label, pred_label] += 1

print("混淆矩阵计算完成！")


# 6. 计算每类识别准确率
print("\n各调制类别识别准确率：")

for i in range(num_classes):

    total = confusion_matrix[i].sum()

    if total > 0:
        accuracy = confusion_matrix[i, i] / total * 100
    else:
        accuracy = 0

    print(
        f"{class_names[i]:8s}: "
        f"{accuracy:.2f}% "
        f"({confusion_matrix[i, i]}/{total})"
    )

overall_accuracy = (
    np.trace(confusion_matrix)
    / confusion_matrix.sum()
    * 100
)

print(f"\n验证集总体准确率：{overall_accuracy:.2f}%")


# 7. 保存 CSV
with open(
    "confusion_matrix_dropout05.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow(["True/Pred"] + class_names)

    for i in range(num_classes):
        writer.writerow(
            [class_names[i]] +
            confusion_matrix[i].tolist()
        )

print("CSV已保存：confusion_matrix_dropout05.csv")


# 8. 绘制混淆矩阵
row_sums = confusion_matrix.sum(axis=1, keepdims=True)

normalized_matrix = np.divide(
    confusion_matrix,
    row_sums,
    out=np.zeros_like(confusion_matrix, dtype=float),
    where=row_sums != 0
) * 100

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
    range(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    class_names
)

plt.xlabel("Predicted Modulation")
plt.ylabel("True Modulation")
plt.title("Normalized Confusion Matrix")

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            f"{normalized_matrix[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix_dropout05.png",
    dpi=300
)

plt.show()

print("图片已保存：confusion_matrix_dropout05.png")
print("\n混淆矩阵实验全部完成！")
