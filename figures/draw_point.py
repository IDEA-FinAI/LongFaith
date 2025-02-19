import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['MuSiQue', '2Wiki', 'Hotpot']
# 每个数据集的QA EM和Attribution F1数据，F1数据已乘以100
mu_sique_em = [39.0, 40.6, 44.6, 48.0, 39.4, 46.6, 47.6, 53.2]
mu_sique_f1 = [75.9, 75.4, 77.9, 78.9, 73.6, 75.8, 76.4, 79.1]

two_wiki_em = [49.0, 55.4, 56.6, 54.2, 52.0, 59.0, 57.8, 57.0]
two_wiki_f1 = [88.7, 89.6, 91.4, 92.0, 88.6, 88.7, 90.0, 92.0]

hotpot_em = [54.2, 53.6, 56.4, 60.8, 53.2, 58.6, 59.6, 58.6]
hotpot_f1 = [89.1, 89.5, 91.0, 91.2, 89.4, 90.2, 90.8, 90.9]

# 创建图形，三张并排的子图
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# 绘制MuSiQue的散点图 (绿色)
axes[0].scatter(mu_sique_f1, mu_sique_em, color='green', label='MuSiQue', marker='o')

# 计算并绘制线性拟合线
p_mu_sique = np.polyfit(mu_sique_f1, mu_sique_em, 1)
poly_mu_sique = np.poly1d(p_mu_sique)
axes[0].plot(mu_sique_f1, poly_mu_sique(mu_sique_f1), color='darkgreen', linestyle='-')

axes[0].set_title('MuSiQue - Performance', fontsize=20)
axes[0].set_xlabel('Attribution - F1 (%)', fontsize=20)
axes[0].set_ylabel('QA - EM (%)', fontsize=20)
axes[0].grid(True)
axes[0].legend(loc='lower right', fontsize=20)
axes[0].tick_params(axis='both', labelsize=20)

# 绘制2Wiki的散点图 (红色)
axes[1].scatter(two_wiki_f1, two_wiki_em, color='red', label='2WikiMHQA', marker='x')

# 计算并绘制线性拟合线
p_two_wiki = np.polyfit(two_wiki_f1, two_wiki_em, 1)
poly_two_wiki = np.poly1d(p_two_wiki)
axes[1].plot(two_wiki_f1, poly_two_wiki(two_wiki_f1), color='darkred', linestyle='-')

axes[1].set_title('2WikiMHQA - Performance', fontsize=20)
axes[1].set_xlabel('Attribution - F1 (%)', fontsize=20)
axes[1].set_ylabel('QA - EM (%)', fontsize=20)
axes[1].grid(True)
axes[1].legend(loc='lower right', fontsize=20)
axes[1].tick_params(axis='both', labelsize=20)

# 绘制Hotpot的散点图 (蓝色)
axes[2].scatter(hotpot_f1, hotpot_em, color='blue', label='HotpotQA', marker='^')

# 计算并绘制线性拟合线
p_hotpot = np.polyfit(hotpot_f1, hotpot_em, 1)
poly_hotpot = np.poly1d(p_hotpot)
axes[2].plot(hotpot_f1, poly_hotpot(hotpot_f1), color='darkblue', linestyle='-')

axes[2].set_title('HotpotQA - Performance', fontsize=20)
axes[2].set_xlabel('Attribution - F1 (%)', fontsize=20)
axes[2].set_ylabel('QA - EM (%)', fontsize=20)
axes[2].grid(True)
axes[2].legend(loc='lower right', fontsize=20)
axes[2].tick_params(axis='both', labelsize=20)

# 优化布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留点空间

# 保存图像
plt.savefig('longfaith_point.png')

# 显示图形
plt.show()
