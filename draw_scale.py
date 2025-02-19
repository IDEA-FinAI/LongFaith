import matplotlib.pyplot as plt

# 数据，所有数值乘以 100 以表示百分比
x = ["1024", "2048", "4096", "8192"]

# LongFaith-SFT 数据
musique_em_sft = [0.390, 0.406, 0.446, 0.480]
musique_f1_sft = [0.508, 0.540, 0.574, 0.599]

twowiki_em_sft = [0.490, 0.554, 0.566, 0.542]
twowiki_f1_sft = [0.606, 0.650, 0.669, 0.639]

hotpot_em_sft = [0.542, 0.536, 0.564, 0.608]
hotpot_f1_sft = [0.692, 0.695, 0.719, 0.742]

# LongFaith-PO 数据
musique_em_po = [0.394, 0.466, 0.476, 0.532]
musique_f1_po = [0.517, 0.592, 0.599, 0.635]

twowiki_em_po = [0.520, 0.590, 0.578, 0.570]
twowiki_f1_po = [0.614, 0.674, 0.676, 0.660]

hotpot_em_po = [0.532, 0.586, 0.596, 0.586]
hotpot_f1_po = [0.683, 0.728, 0.733, 0.729]

# LongBench-SFT 数据
LongBenchS_em_SFT = [0.112, 0.141, 0.129, 0.150]
LongBenchS_f1_SFT = [0.312, 0.384, 0.327, 0.376]

LongBenchM_em_SFT = [0.387, 0.430, 0.437, 0.450]
LongBenchM_f1_SFT = [0.485, 0.538, 0.541, 0.559]

# LongBench-PO 数据
LongBenchS_em_po = [0.104, 0.154, 0.182, 0.176]
LongBenchS_f1_po = [0.237, 0.353, 0.392, 0.416]

LongBenchM_em_po = [0.373, 0.437, 0.457, 0.477]
LongBenchM_f1_po = [0.474, 0.546, 0.564, 0.571]

# 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(12, 14))

# LongFaith-SFT 第一张图（EM 和 F1）
axs[0, 0].plot(x, [i * 100 for i in musique_em_sft], color='green', label='MuSiQue', marker='o')
axs[0, 0].plot(x, [i * 100 for i in twowiki_em_sft], color='red', label='2WikiMHQA', marker='o')
axs[0, 0].plot(x, [i * 100 for i in hotpot_em_sft], color='blue', label='HotpotQA', marker='o')
axs[0, 0].plot(x, [i * 100 for i in LongBenchS_em_SFT], color='purple', label='LongBench(S)', marker='o')
axs[0, 0].plot(x, [i * 100 for i in LongBenchM_em_SFT], color='orange', label='LongBench(M)', marker='o')
axs[0, 0].set_title('LongFaith-SFT - EM (%)', fontsize=20)
axs[0, 0].set_xlabel('# Examples', fontsize=20)
axs[0, 0].set_ylabel('EM (%)', fontsize=20)
axs[0, 0].legend(fontsize=16)
axs[0, 0].tick_params(axis='both', labelsize=20)
axs[0, 0].grid(True)

axs[0, 1].plot(x, [i * 100 for i in musique_f1_sft], color='green', label='MuSiQue', marker='o')
axs[0, 1].plot(x, [i * 100 for i in twowiki_f1_sft], color='red', label='2WikiMHQA', marker='o')
axs[0, 1].plot(x, [i * 100 for i in hotpot_f1_sft], color='blue', label='HotpotQA', marker='o')
axs[0, 1].plot(x, [i * 100 for i in LongBenchS_f1_SFT], color='purple', label='LongBench(S)', marker='o')
axs[0, 1].plot(x, [i * 100 for i in LongBenchM_f1_SFT], color='orange', label='LongBench(M)', marker='o')
axs[0, 1].set_title('LongFaith-SFT - F1 (%)', fontsize=20)
axs[0, 1].set_xlabel('# Examples', fontsize=20)
axs[0, 1].set_ylabel('F1 (%)', fontsize=20)
axs[0, 1].legend(fontsize=16)
axs[0, 1].tick_params(axis='both', labelsize=20)
axs[0, 1].grid(True)

# LongFaith-PO 第二张图（EM 和 F1）
axs[1, 0].plot(x, [i * 100 for i in musique_em_po], color='green', label='MuSiQue', marker='o')
axs[1, 0].plot(x, [i * 100 for i in twowiki_em_po], color='red', label='2WikiMHQA', marker='o')
axs[1, 0].plot(x, [i * 100 for i in hotpot_em_po], color='blue', label='HotpotQA', marker='o')
axs[1, 0].plot(x, [i * 100 for i in LongBenchS_em_po], color='purple', label='LongBench(S)', marker='o')
axs[1, 0].plot(x, [i * 100 for i in LongBenchM_em_po], color='orange', label='LongBench(M)', marker='o')
axs[1, 0].set_title('LongFaith-PO - EM (%)', fontsize=20)
axs[1, 0].set_xlabel('# Examples', fontsize=20)
axs[1, 0].set_ylabel('EM (%)', fontsize=20)
axs[1, 0].legend(fontsize=16)
axs[1, 0].tick_params(axis='both', labelsize=20)
axs[1, 0].grid(True)

axs[1, 1].plot(x, [i * 100 for i in musique_f1_po], color='green', label='MuSiQue', marker='o')
axs[1, 1].plot(x, [i * 100 for i in twowiki_f1_po], color='red', label='2WikiMHQA', marker='o')
axs[1, 1].plot(x, [i * 100 for i in hotpot_f1_po], color='blue', label='HotpotQA', marker='o')
axs[1, 1].plot(x, [i * 100 for i in LongBenchS_f1_po], color='purple', label='LongBench(S)', marker='o')
axs[1, 1].plot(x, [i * 100 for i in LongBenchM_f1_po], color='orange', label='LongBench(M)', marker='o')
axs[1, 1].set_title('LongFaith-PO - F1 (%)', fontsize=20)
axs[1, 1].set_xlabel('# Examples', fontsize=20)
axs[1, 1].set_ylabel('F1 (%)', fontsize=20)
axs[1, 1].legend(fontsize=16)
axs[1, 1].tick_params(axis='both', labelsize=20)
axs[1, 1].grid(True)

# # 新增 LongBench-SFT 图（EM 和 F1）
# axs[2, 0].plot(x, [i * 100 for i in LongBenchS_em_SFT], color='green', label='LongBench(S)', marker='o')
# axs[2, 0].plot(x, [i * 100 for i in LongBenchM_em_SFT], color='red', label='LongBench(M)', marker='o')
# axs[2, 0].set_title('LongFaith-SFT - EM (%)', fontsize=20)
# axs[2, 0].set_xlabel('# Examples', fontsize=20)
# axs[2, 0].set_ylabel('EM (%)', fontsize=20)
# axs[2, 0].legend(loc='lower right', fontsize=16)
# axs[2, 0].tick_params(axis='both', labelsize=20)
# axs[2, 0].grid(True)

# axs[2, 1].plot(x, [i * 100 for i in LongBenchS_f1_SFT], color='green', label='LongBench(S)', marker='o')
# axs[2, 1].plot(x, [i * 100 for i in LongBenchM_f1_SFT], color='red', label='LongBench(M)', marker='o')
# axs[2, 1].set_title('LongFaith-SFT - F1 (%)', fontsize=20)
# axs[2, 1].set_xlabel('# Examples', fontsize=20)
# axs[2, 1].set_ylabel('F1 (%)', fontsize=20)
# axs[2, 1].legend(loc='lower right', fontsize=16)
# axs[2, 1].tick_params(axis='both', labelsize=20)
# axs[2, 1].grid(True)

# # 新增 LongBench-PO 图（EM 和 F1）
# axs[3, 0].plot(x, [i * 100 for i in LongBenchS_em_po], color='green', label='LongBench(S)', marker='o')
# axs[3, 0].plot(x, [i * 100 for i in LongBenchM_em_po], color='red', label='LongBench(M)', marker='o')
# axs[3, 0].set_title('LongFaith-PO - EM (%)', fontsize=20)
# axs[3, 0].set_xlabel('# Examples', fontsize=20)
# axs[3, 0].set_ylabel('EM (%)', fontsize=20)
# axs[3, 0].legend(loc='lower right', fontsize=16)
# axs[3, 0].tick_params(axis='both', labelsize=20)
# axs[3, 0].grid(True)

# axs[3, 1].plot(x, [i * 100 for i in LongBenchS_f1_po], color='green', label='LongBench(S)', marker='o')
# axs[3, 1].plot(x, [i * 100 for i in LongBenchM_f1_po], color='red', label='LongBench(M)', marker='o')
# axs[3, 1].set_title('LongFaith-PO - F1 (%)', fontsize=20)
# axs[3, 1].set_xlabel('# Examples', fontsize=20)
# axs[3, 1].set_ylabel('F1 (%)', fontsize=20)
# axs[3, 1].legend(loc='lower right', fontsize=16)
# axs[3, 1].tick_params(axis='both', labelsize=20)
# axs[3, 1].grid(True)

# 调整布局
plt.tight_layout()

# 保存图像为longfaith_scale.png
plt.savefig('longfaith_scale.png')

# 显示图形
plt.show()
