
import os
import glob
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = os.path.join("logs", "vtcnn")

# 文件夹名称对应的实验指标
metric_folders = {
    "train_loss": "loss_loss",
    "val_loss": "loss_val_loss",
    "train_acc": "acc_acc",
    "val_acc": "acc_val_acc",
    "train_top3": "top3_top3",
    "val_top3": "top3_val_top3",
    "train_top5": "top5_top5",
    "val_top5": "top5_val_top5",
    "epoch": "epoch_epoch"
}

data = {}

for name, folder_name in metric_folders.items():
    folder = os.path.join(root, folder_name)
    files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))

    if not files:
        print("没有找到：", folder_name)
        continue

    latest_file = max(files, key=os.path.getmtime)

    ea = EventAccumulator(latest_file)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])

    if not tags:
        continue

    events = ea.Scalars(tags[0])
    data[name] = [e.value for e in events]

# 按轮数整理
count = len(data.get("epoch", []))

if count == 0:
    count = max((len(v) for v in data.values()), default=0)

columns = list(metric_folders.keys())

print("\n========== 10轮训练完整结果 ==========")
print("轮数 | 训练损失 | 验证损失 | 训练准确率 | 验证准确率")

rows = []

for i in range(count):
    row = {}

    for name in columns:
        values = data.get(name, [])
        row[name] = values[i] if i < len(values) else ""

    epoch = row["epoch"] if row["epoch"] != "" else i + 1

    print(
        f"{int(epoch):2} | "
        f"{row['train_loss']!s:10.8} | "
        f"{row['val_loss']!s:10.8} | "
        f"{row['train_acc']!s:10.8} | "
        f"{row['val_acc']!s:10.8}"
    )

    rows.append(row)

# 保存完整结果
if rows:
    with open("training_log_10epochs.csv", "w",
              newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n完整结果已保存：training_log_10epochs.csv")
