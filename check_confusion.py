
import csv
import numpy as np

# 11种调制类别
class_names = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK",
    "CPFSK", "GFSK", "PAM4", "QAM16",
    "QAM64", "QPSK", "WBFM"
]

# 三组模型及其统一评估准确率
models = [
    ("Dropout=0.0", "unified_confusion_dropout00.csv", 50.57),
    ("Dropout=0.3", "unified_confusion_dropout03.csv", 53.21),
    ("Dropout=0.5", "unified_confusion_dropout05.csv", 50.65)
]

for model_name, filename, expected_acc in models:

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    # 读取混淆矩阵
    with open(filename, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 去掉第一行表头和第一列类别名称
    matrix = np.array(
        [[int(value) for value in row[1:]] for row in rows[1:]],
        dtype=np.int64
    )

    # 每一类识别准确率
    row_totals = matrix.sum(axis=1)
    correct = np.diag(matrix)

    class_accuracy = correct / row_totals * 100

    print("\n各类别识别准确率：")

    for name, accuracy, total in zip(
        class_names, class_accuracy, row_totals
    ):
        print(
            f"{name:8s} | "
            f"正确样本数：{int(matrix[class_names.index(name), class_names.index(name)]):4d} | "
            f"总样本数：{int(total):4d} | "
            f"准确率：{accuracy:.2f}%"
        )

    # 混淆矩阵计算总体准确率
    total_correct = np.trace(matrix)
    total_samples = matrix.sum()
    overall_acc = total_correct / total_samples * 100

    print("\n混淆矩阵总样本数：", int(total_samples))
    print("正确分类样本数：", int(total_correct))
    print(f"混淆矩阵总体准确率：{overall_acc:.2f}%")
    print(f"统一评估总体准确率：{expected_acc:.2f}%")

    if abs(overall_acc - expected_acc) < 0.02:
        print("核查结果：通过")
    else:
        print("核查结果：不一致，请进一步检查")
