import csv
import matplotlib.pyplot as plt

# 读取之前保存的SNR测试结果
file_path = "snr_accuracy_results.csv"

snr_values = []
accuracy_values = []

with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    print("CSV列名：", reader.fieldnames)

    for row in reader:
        # 自动识别SNR和准确率所在的列
        snr_key = next(
            (k for k in row if "snr" in k.lower()), None
        )
        acc_key = next(
            (k for k in row if "acc" in k.lower()), None
        )

        if snr_key is None or acc_key is None:
            raise ValueError("无法识别SNR或准确率列，请把上面显示的CSV列名发给我。")

        snr_values.append(float(row[snr_key]))
        accuracy_values.append(float(row[acc_key]))

# 按SNR从低到高排序
data = sorted(zip(snr_values, accuracy_values))
snr_values, accuracy_values = zip(*data)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(snr_values, accuracy_values, marker="o", linewidth=2)

plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("CNN Modulation Recognition Accuracy under Different SNRs")
plt.xticks(snr_values)
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("snr_accuracy_curve.png", dpi=300)
plt.show()

print("SNR曲线绘制成功！")
print("图片已保存为：snr_accuracy_curve.png")