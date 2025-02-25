import matplotlib.pyplot as plt
import numpy as np

# X and Y 值
x_axis_data = [10, 20, 30, 40, 50]
# y_axis_data1 = [0.705, 0.699, 0.706, 0.704, 0.706]
# y_axis_data2 = [0.701, 0.694, 0.709, 0.707, 0.700]
y_axis_data1 = [0.977, 0.972, 0.979, 0.980, 0.978]
y_axis_data2 = [0.970, 0.966, 0.977, 0.970, 0.974]

# 给Y轴/纵坐标划分等间隔：(0,101,10)，0到100，每隔10个划分一个刻度
yticks = np.arange(0.960, 0.980 + 0.005, 0.005)
# 设置X轴刻度
xticks = x_axis_data


# 添加Y轴网格线，并设置颜色为灰色，线宽为2
plt.grid(axis='y', color='#DDDDDD', linewidth=1.5)

# 画图  plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(x_axis_data, y_axis_data1, 's-', color='#0096FF', markersize=8, alpha=1, linewidth=1.5, label='RMSE(Movies->CDs)')
plt.plot(x_axis_data, y_axis_data2, 'o-', color='#ff8d00', markersize=8, alpha=1, linewidth=1.5, label='RMSE(Books->CDs)')


# 显示上面的label,放在右下方
plt.legend(loc='lower right')
plt.ylabel('MAE')

# 设置X轴和Y轴的刻度
plt.xticks(xticks)
plt.yticks(yticks)

plt.show()

'''
# 拓展实验
import matplotlib.pyplot as plt
import numpy as np

# 数据设置
datasets = ['Movie->CD', 'Book->CD']
backend_model = [1.009, 0.984]  # 有train_guide,无t
ddrm_model = [0.983, 0.968]  # 有t,无train_guide
additional_data = [0.972, 0.956]  # 最终

# 条形图的位置
bar_width = 0.15  # 条形图的宽度
spacing = 0.02  # 条形图之间的间隔
n_groups = len(datasets)  # 数据集的数量
n_bars = 3  # 每组的条形图数量
index = np.arange(n_groups)

# 绘制条形图
fig, ax = plt.subplots()

# 设置纵坐标的范围和刻度
ax.set_ylim(0.85, 1.10)  # 上限和下限
yticks = np.arange(0.85, 1.10, 0.05)  # 从0.90到1.15每隔0.05,不包括1.15.
ax.set_yticks(yticks)

# 绘制Backend model的条形图
bars1 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2, backend_model, bar_width, color='#4F7942', label='Backend model')

# 绘制DDRM的条形图
bars2 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2 + bar_width + spacing, ddrm_model, bar_width, color='#CD7F32', label='DDRM')

# 绘制额外的条形图（绿色）
bars3 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2 + 2 * (bar_width + spacing), additional_data, bar_width, color='#005CBF', label='Additional')

# 设置横坐标标签
ax.set_xticks(index)
ax.set_xticklabels(datasets)

# 添加图例
ax.legend()

# 添加标题和标签
ax.set_ylabel('RMSE')

# 显示图表
plt.show()

'''


'''
# 拓展实验
import matplotlib.pyplot as plt
import numpy as np
#  MAE

# 数据设置
datasets = ['Movie->CD', 'Book->CD']
backend_model = [0.756, 0.736]  # 有train_guide,无t
ddrm_model = [0.710, 0.714]  # 有t,无train_guide
additional_data = [0.694, 0.695]  # 最终

# 条形图的位置
bar_width = 0.15  # 条形图的宽度
spacing = 0.02  # 条形图之间的间隔
n_groups = len(datasets)  # 数据集的数量
n_bars = 3  # 每组的条形图数量
index = np.arange(n_groups)

# 绘制条形图
fig, ax = plt.subplots()

# 设置纵坐标的范围
ax.set_ylim(0.68, 0.78)
ax.set_yticks(np.arange(0.68, 0.80, 0.02))

# 绘制Backend model的条形图
bars1 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2, backend_model, bar_width, color='#4F7942', label='Backend model')

# 绘制DDRM的条形图
bars2 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2 + bar_width + spacing, ddrm_model, bar_width, color='#005CBF', label='DDRM')

# 绘制额外的条形图（绿色）
bars3 = ax.bar(index - (n_bars - 1) * (bar_width + spacing) / 2 + 2 * (bar_width + spacing), additional_data, bar_width, color='#CD7F32', label='Additional')

# 设置横坐标标签
ax.set_xticks(index)
ax.set_xticklabels(datasets)

# 添加图例
ax.legend()

# 添加标题和标签
ax.set_ylabel('MAE')

# 显示图表
plt.show()
'''
