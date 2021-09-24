import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
# from warnings import filterwarnings
import seaborn as sns


sns.set_style('darkgrid')
plt.rc('font', size=10)
# filterwarnings("ignore")

# ###########---------------set up and plot input data-----------------######################

base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 7, 1
length = [steps_day*5+steps_day, steps_week*5+steps_week]  # 代表每个序列的长度，分别为周、日序列的一年及两年

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)


##########################################################--构造加法周期性时间序列，模拟真实销售
# random.seed(0)
# np.random.seed(0)
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_add_actual = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_season_actual[i] = y_season_actual[i] + max(y_season_actual[i]) + 1  # 使y_season均为正
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3*np.random.standard_t(length[i]-1, length[i])  # normal(0, 1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i])*0.9] = 0  # 只保留随机数中的离群值
    y_input_add_actual[i] = 10*(y_level_actual[i] + y_trend_actual[i] + y_season_actual[i] + y_noise_actual[i])  # 假定各项以加法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i])-min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(f'第{i}条真实加法性序列最终极差：{max(y_input_add_actual[i]) - min(y_input_add_actual[i])}，均值：{np.mean(y_input_add_actual[i])}', '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_add_actual[i] = pd.Series(y_input_add_actual[i]).rename('y_input_add_actual')
    # y_input_add_actual[i][y_input_add_actual[i] < 0] = 0

# 绘制加法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('add_actual_pred: 14+7', figsize=(5, 10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_add_actual[0].plot(ax=ax1, legend=True)
y_level_actual[0].plot(ax=ax2, legend=True)
y_trend_actual[0].plot(ax=ax3, legend=True)
y_season_actual[0].plot(ax=ax4, legend=True)
y_noise_actual[0].plot(ax=ax5, legend=True)

plt.figure('add_actual_pred: 4+1', figsize=(5, 10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_add_actual[1].plot(ax=ax1, legend=True)
y_level_actual[1].plot(ax=ax2, legend=True)
y_trend_actual[1].plot(ax=ax3, legend=True)
y_season_actual[1].plot(ax=ax4, legend=True)
y_noise_actual[1].plot(ax=ax5, legend=True)

##########################################################--构造乘法周期性时间序列，模拟真实销售
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_mul_actual = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_season_actual[i] = y_season_actual[i] + max(y_season_actual[i]) + 1  # 使y_season均为正
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3*np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_actual[i] = (y_level_actual[i] + y_trend_actual[i]) * y_season_actual[i] * abs(y_noise_actual[i])  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(f'第{i}条真实乘法性序列最终极差：{max(y_input_mul_actual[i]) - min(y_input_mul_actual[i])}，均值：{np.mean(y_input_mul_actual[i])}', '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(y_input_mul_actual[i]).rename('y_input_mul_actual')
    # y_input_mul_actual[i][y_input_mul_actual[i] < 0] = 0

# 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('mul_actual_pred: 14+7', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_mul_actual[0].plot(ax=ax1, legend=True)
y_level_actual[0].plot(ax=ax2, legend=True)
y_trend_actual[0].plot(ax=ax3, legend=True)
y_season_actual[0].plot(ax=ax4, legend=True)
y_noise_actual[0].plot(ax=ax5, legend=True)

plt.figure('mul_actual_pred: 4+1', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_mul_actual[1].plot(ax=ax1, legend=True)
y_level_actual[1].plot(ax=ax2, legend=True)
y_trend_actual[1].plot(ax=ax3, legend=True)
y_season_actual[1].plot(ax=ax4, legend=True)
y_noise_actual[1].plot(ax=ax5, legend=True)


##########################################################--构造加法周期性时间序列，模拟预测销售
# random.seed(0)
# np.random.seed(0)
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_add_pred = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1/2 * np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_season_pred[i] = y_season_pred[i] + max(y_season_pred[i]) + 1  # 使y_season_pred均为正
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_pred[i])) + np.average(abs(y_season_pred[i])) + np.random.randint(-np.average(abs(y_season_pred[i])), np.average(abs(y_season_pred[i])))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i]-1, length[i])  # normal(0, 1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i])*0.9] = 0  # 只保留随机数中的离群值
    y_input_add_pred[i] = 10*(y_level_pred[i] + y_trend_pred[i] + y_season_pred[i] + y_noise_pred[i])  # 假定各项以加法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i])-min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测加法性序列最终极差：{max(y_input_add_pred[i]) - min(y_input_add_pred[i])}，均值：{np.mean(y_input_add_pred[i])}', '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_add_pred[i] = pd.Series(y_input_add_pred[i]).rename('y_input_add_pred')
    # y_input_add_pred[i][y_input_add_pred[i] < 0] = 0

# 绘制加法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('add_actual_pred: 14+7', figsize=(5, 10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_add_pred[0].plot(ax=ax1, legend=True)
y_level_pred[0].plot(ax=ax2, legend=True)
y_trend_pred[0].plot(ax=ax3, legend=True)
y_season_pred[0].plot(ax=ax4, legend=True)
y_noise_pred[0].plot(ax=ax5, legend=True)

plt.figure('add_actual_pred: 4+1', figsize=(5, 10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_add_pred[1].plot(ax=ax1, legend=True)
y_level_pred[1].plot(ax=ax2, legend=True)
y_trend_pred[1].plot(ax=ax3, legend=True)
y_season_pred[1].plot(ax=ax4, legend=True)
y_noise_pred[1].plot(ax=ax5, legend=True)

##########################################################--构造乘法周期性时间序列，模拟预测销售
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_mul_pred = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1/2 * np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_season_pred[i] = y_season_pred[i] + max(y_season_pred[i]) + 1  # 使y_season_pred均为正
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_pred[i])) + np.average(abs(y_season_pred[i])) + np.random.randint(-np.average(abs(y_season_pred[i])), np.average(abs(y_season_pred[i])))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布；使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_pred[i] = (y_level_pred[i] + y_trend_pred[i]) * y_season_pred[i] * abs(y_noise_pred[i])  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测乘法性序列最终极差：{max(y_input_mul_pred[i]) - min(y_input_mul_pred[i])}，均值：{np.mean(y_input_mul_pred[i])}', '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(y_input_mul_pred[i]).rename('y_input_mul_pred')
    # y_input_mul_pred[i][y_input_mul_pred[i] < 0] = 0

# 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('mul_actual_pred: 14+7', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_mul_pred[0].plot(ax=ax1, legend=True)
y_level_pred[0].plot(ax=ax2, legend=True)
y_trend_pred[0].plot(ax=ax3, legend=True)
y_season_pred[0].plot(ax=ax4, legend=True)
y_noise_pred[0].plot(ax=ax5, legend=True)

plt.figure('mul_actual_pred: 4+1', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_mul_pred[1].plot(ax=ax1, legend=True)
y_level_pred[1].plot(ax=ax2, legend=True)
y_trend_pred[1].plot(ax=ax3, legend=True)
y_season_pred[1].plot(ax=ax4, legend=True)
y_noise_pred[1].plot(ax=ax5, legend=True)



def dyn_df_weighted(df, type, w=None, r=3/2, d=1, initial=2):
    """
    传入df，根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，将df各列与权重相乘再相加，得到一条最终序列。
    :param df: 需要加权的各列组成的df
    :param type: 采用几何级数或算数级数加权，type = 'geometric'或'arithmetic'
    :param w: 权重系数可人为指定，默认为None
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: 将df各列分别乘以权重w再相加得到一条最终的加权序列
    """
    if w is None:
        w = list()
        if type == 'geometric':
            for i in range(len(df.columns)):
                w.append((1 / r) ** i)
        elif type == 'arithmetic':
            for i in range(len(df.columns)):
                w.append(1 / (initial + d * i))
        else:
            raise Exception('if w=None, type must be one of geometric or arithmetic')
        w = np.array(w) / sum(w)
    elif (w is not None) and (len(w) == len(df.columns)):
        w = np.array(w) / sum(w)
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度相等')
    print(w, sum(w))
    return np.matmul(df.values, w)


def dyn_seri_weighted(seri, type, w=None, r=2, d=1, initial=1):
    """
    传入seri，根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，将seri各点与权重相乘再相加，得到一条最终seri。
    :param seri: 需要加权的一维数组
    :param type: 采用几何级数或算数级数加权，type = 'geometric'或'arithmetic'
    :param w: 权重系数可人为指定，默认为None
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: 返回seri各点与权重w相乘再相加，得到的一条最终加权序列
    """
    if w is None:
        w = list()
        if type == 'geometric':
            for i in range(len(seri)):
                w.append((1 / r) ** i)
        elif type == 'arithmetic':
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))
        else:
            raise Exception('if w=None, type must be one of geometric or arithmetic')
        w = np.array(w) / sum(w)
    elif (w is not None) and (len(w) == len(seri)):
        w = np.array(w) / sum(w)
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度相等')
    print(w, sum(w))
    return np.dot(seri, w)


k_mul = np.array(range(7))
y_all = [[]] * len(k_mul)

for i in range(0, len(k_mul)):
    y_all[i] = np.sin(random.choices(range(0, 100), k=20))

y_all = pd.DataFrame(y_all).T
y_all.fillna(value=0, inplace=True)

print(dyn_df_weighted(y_all, type='geometric'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], type='geometric'), '\n')

print(dyn_df_weighted(y_all, type='arithmetic'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], type='arithmetic'), '\n')

print(dyn_df_weighted(y_all, w=np.ones(len(k_mul)), type='arithmetic'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul)), type='geometric'), '\n')


if sum(pd.Series(y[-43:]).notnull()) >= 7:  # 当滑动窗口内的真实序列至少有7个点，才进行波动性的统计判断

    y = y[-43:][y[-43:] > 0]  # 认为真实序列中为nan和非正的点为异常点，将其排除不进行波动性计算
    yhat = yhat[-43:][y.index]  # 预测序列也排除真实值为异常点的对应点
    # 将y和yhat分解为level, trend, weekly seasonality和error
    y1, yhat1 = y.diff(periods=1), yhat.diff(periods=1)  # 将真实序列和预测序列的level项去掉，剩trend, seasonality, error
    y1, yhat1 = y1[y1.notnull()], yhat1[yhat1.notnull()]

    y7, yhat7 = y.diff(periods=7), yhat.diff(periods=7)  # 将真实序列和预测序列的level, seasonality项去掉，剩trend，error
    y7, yhat7 = y7[y7.notnull()], yhat7[yhat7.notnull()]

    y1_1, yhat1_1 = y1.diff(periods=1), yhat1.diff(periods=1)  # 将真实序列和预测序列的level, trend项去掉，剩seasonality, error
    y1_1, yhat1_1 = y1_1[y1_1.notnull()], yhat1_1[yhat1_1.notnull()]

    y7_1, yhat7_1 = y7.diff(periods=1), yhat7.diff(periods=1)  # 将真实序列和预测序列的level, seasonality, trend项去掉，剩error
    y7_1, yhat7_1 = y7_1[y7_1.notnull()], yhat7_1[yhat7_1.notnull()]


y_input_mul_actual[0]


def dynamic_bounds(yhat, y):
    """
    :param yhat: 预测序列5不带索引时的全长序列
    :param y: 真实序列不带索引时的全长序列，起点与预测序列5相同
    :return: 下一个预测点的最终值（正常值）
    """
    y, yhat = pd.Series(y), pd.Series(yhat)
    y = y[-14:]
    yhat = yhat[min(y.index): max(y.index)+2]  # 使yhat的索引起点与y的索引起点相同，而yhat的索引终点比y的索引终点多1.


    if sum(y.notnull()) >= 7:
        yhat1, yhat7 = yhat.diff(periods=1), yhat.diff(periods=7)


    normal_value = 1
    return normal_value
