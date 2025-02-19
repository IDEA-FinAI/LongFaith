import numpy as np
import matplotlib.pyplot as plt

# 测试集名称
categories = ['MuSiQue', '2Wiki', 'HotpotQA', 'Qasper(S)', 'MFQA-En(S)', 'MuSiQue(M)', '2Wiki(M)', 'HotpotQA(M)']

# 不同方法在各个测试集上的F1得分
methods = {
    "LongAlpaca": [21.6, 47.8, 32.7, 5.7, 5.8, 8.5, 25.4, 12.5],
    "LongAlign": [24.8, 55.6, 51.0, 6.5, 10.7, 15.0, 33.4, 35.8],
    "MuSiQue-Attribute": [13.9, 23.9, 20.2, 10.0, 8.3, 15.2, 21.2, 25.6],
    "LongMIT": [4.9, 3.3, 10.1, 9.5, 5.6, 7.5, 3.6, 23.7],
    "LongReward-SFT": [6.2, 23.3, 15.6, 2.6, 0.5, 1.1, 6.6, 8.9],
    "SeaLong-SFT": [31.3, 55.8, 59.4, 14.5, 18.6, 24.1, 34.1, 37.3],
    "LongFaith-SFT": [56.8, 73.8, 70.5, 36.9, 47.0, 50.1, 63.9, 53.1],
    "LongReward-PO": [3.3, 14.3, 8.9, 1.6, 0.1, 0.0, 4.4, 3.3],
    "SeaLong-PO": [30.2, 50.1, 58.3, 17.1, 20.1, 18.1, 34.0, 40.2],
    "LongFaith-PO": [60.5, 68.0, 65.4, 38.1, 46.7, 50.2, 73.7, 55.6]
}

fig, ax = plt.subplots(figsize=(9, 5), subplot_kw=dict(polar=True))

# 更新角度数组
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # 闭合圆形

# 为每个方法绘制雷达图
for method, scores in methods.items():
    scores = scores + scores[:1]  # 将第一项值添加到列表末尾，使得图形闭合
    ax.plot(angles, scores, label=method, linewidth=2, linestyle='solid')

# 设置雷达图的标签
ax.set_yticklabels([])  # 去掉雷达图的y轴标签
ax.set_xticks(angles[:-1])  # 去掉最后一项角度
ax.set_xticklabels(categories, fontsize=12)  # 增加类别标签字体大小

# 设置字体大小
plt.xticks(fontsize=18)  # 类别标签字体大小
plt.yticks(fontsize=18)  # 可以控制y轴标签大小，如果你有需要

# 添加图例并调整位置
plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), fontsize=18, title='Datasets', title_fontsize=18)

# 显示雷达图
plt.tight_layout()
plt.show()

# 保存图像
plt.savefig('longfaith_radar.png', bbox_inches='tight')
