import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


freq = 'D'
t0 = '2020-01-01'
data_length = 7*10
num_ts = 3
period = 7
fit_series, origin_series = [], []
time_ticks = np.array(range(data_length))
index = pd.date_range(t0, periods=data_length, freq='D')
np.random.seed(0)
level = 10 * np.random.rand()


for k in range(num_ts):
    # generate fitting data
    np.random.seed(k+10)
    seas_amplitude_fit = (0.1 + 0.3*np.random.rand()) * level
    sig_fit = 0.05 * level  # noise parameter (constant in time)
    source_fit = level + seas_amplitude_fit * np.sin(time_ticks * (2 * np.pi) / period)
    noise_fit = 0.1 * sig_fit * np.random.randn(data_length)
    data_fit = source_fit + noise_fit
    fit_series.append(pd.Series(data=data_fit, index=index))
    # generate origin data
    np.random.seed(k+10)
    seas_amplitude_origin = (0.2 + 0.3 * np.random.rand()) * level
    sig_origin = 0.5 * level  # noise parameter (constant in time)
    source_origin = level + seas_amplitude_origin * np.sin(time_ticks * (2 * np.pi) / period)
    noise_origin = 0.5 * sig_origin * np.random.randn(data_length)
    data_origin = source_origin + noise_origin
    origin_series.append(pd.Series(data=data_origin, index=index))
    # plot the two contrast series
    plt.figure()
    fit_series[k].plot()
    origin_series[k].plot()
    plt.title('contrast fit_series and origin series')
    plt.show()


def MAE(Y, y):
    """
    param：
        Y: 原始序列（假定波动较大）
        y: 拟合序列（假定波动较小）
    return：
        MAE值，该值的大小与两条序列间平均偏差程度成正比，该值越大，平均偏差程度越大；
        两序列间的残差（特别是残差的离群值）对MAE的影响比LMAE大，比EMAE小。
    """

    Y, y = np.array(Y), np.array(y)
    mae = sum(abs(Y - y)) / len(Y)

    return mae


def LMAE(Y, y, a=np.exp(1)):
    """
    param：
        Y: 原始序列（假定波动较大）
        y: 拟合序列（假定波动较小）
        a: 对数的底数，大于1，作用于换底公式，使所有对数函数为单调递增函数；
        该值越大，则两序列间的残差（特别是残差的离群值）对LMAE返回值影响的弱化作用越明显。
    return：
        对数MAE值，该值的大小与两条序列间平均偏差程度成正比，该值越大，平均偏差程度越大；
        但两序列间的残差（特别是残差的离群值）对LMAE的影响比MAE小。
    """

    Y, y = np.array(Y), np.array(y)
    Y[Y < 0] = 0  # 使对数的真数≥1，从而使所有对数值非负，便于统一比较。
    y[y < 0] = 0
    lmae = sum(abs(np.log(abs(Y+1)) / np.log(a) - np.log(abs(y+1)) / np.log(a))) / len(Y)

    return lmae


def EMAE(Y, y, a=1.2):
    """
    param：
        Y: 原始序列（假定波动较大）
        y: 拟合序列（假定波动较小）
        a: 指数的自变量，≥1，该值越大，则两序列间的残差（特别是残差的离群值）对EMAE返回值影响的强化作用越明显；
        当a=1时，EMAE化简为MAE。
    return：
        指数MAE值，该值的大小与两条序列间平均偏差程度成正比，该值越大，平均偏差程度越大；
        且两序列间的残差（特别是残差的离群值）对EMAE的影响比MAE大。
    """

    Y, y = np.array(Y), np.array(y)
    Y[Y < 0] = 0  # 使指数的底数≥1，则所有指数均为递增函数
    y[y < 0] = 0
    emae = sum(abs((Y+1)**a - (y+1)**a)) / len(Y)

    return emae


a, b, c = [], [], []
for k in range(num_ts):
    a.append(MAE(origin_series[k], fit_series[k]))
    b.append(LMAE(origin_series[k], fit_series[k]))
    c.append(EMAE(origin_series[k], fit_series[k]))
print(' 每对序列的MAE：', '\n', a, '\n', '每对序列的LMAE：', '\n', b, '\n', '每对序列的EMAE：', '\n', c, '\n')
print(' EMAE - MAE', '\n', np.array(c)-np.array(a), '\n', 'MAE - LMAE', '\n', np.array(a)-np.array(b), '\n')
print(' MAE与LMAE间的相关程度：', '\n', pearsonr(a, b), '\n', 'MAE与EMAE间的相关程度：', '\n', pearsonr(a, c), '\n',
      'LMAE与EMAE间的相关程度：', '\n', pearsonr(c, b), '\n')
