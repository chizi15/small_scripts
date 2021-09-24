import matplotlib.pyplot as plt
import numpy as np
import copy
plt.style.use('seaborn-white')


x = [np.linspace(0, 35, 36), np.linspace(0, 9, 10), np.linspace(0, 19, 20)]
# x_plot使绘图时的x取值比多项式拟合时，自变量x的取值更密集，以观察到可能出现的振荡
x_plot = [np.linspace(0, 35, 36*5), np.linspace(0, 9, 10*5), np.linspace(0, 19, 20*5)]
y = [np.random.rand(len(x[0])), np.random.randn(len(x[1]))*10, np.random.randn(len(x[2]))*10]
y[0][int(len(x[0])/2-2):int(len(x[0])/2+2)] = 10
n = [[2, 3, 4], [4, 9, 19], [9, 19, 39]]
weights = [(max(y[i])*2-y[i])/sum(max(y[i])*2-y[i]) for i in range(len(x))]  # y[i]越大，weights越小；让离群点占较小权重，正常点占较大权重。

c = copy.deepcopy(n)  # c与n具有相同的结构；一定要用copy.deepcopy，不能用浅拷贝copy，否则后面对c的赋值也会改变n
p = copy.deepcopy(n)  # p与n具有相同的结构
for i in range(len(x)):
    for j in range(len(n[i])):
        c[i][j] = np.polyfit(x[i], y[i], n[i][j], full=True, w=weights[i])
        p[i][j] = np.poly1d(c[i][j][0])
        print('序列长度{0}个点，拟合次数为{1}，即目标函数未知数个数为{2}，即目标函数对自变量偏导数的方程个数为{2}：'.format(len(x[i]), n[i][j], n[i][j]+1))
        print('拟合函数的系数（次数由高到低）：', '\n', c[i][j][0])
        print('拟合值与实际值的MAPE：', sum(abs((p[i][j](x[i])-y[i])/y[i]))/len(x[i]))
        print('目标函数对自变量偏导数方程组的系数矩阵的秩：', c[i][j][2])
        print('目标函数对自变量偏导数方程组的系数矩阵的奇异值：', '\n', c[i][j][3])
        print('拟合的相关条件数：', c[i][j][4], '\n')
    plt.figure(figsize=(10, 6))
    plt.scatter(x[i], y[i], facecolors='none', edgecolor='darkblue', label='original_data')
    plt.plot(x[i], y[i], color='darkblue')
    plt.plot(x_plot[i], p[i][0](x_plot[i]), '-', color='g', label='polynomial of degree {}'.format(n[i][0]))
    plt.plot(x_plot[i], p[i][1](x_plot[i]), '--', color='orange', label='polynomial of degree {}'.format(n[i][1]))
    plt.plot(x_plot[i], p[i][2](x_plot[i]), color='magenta', label='polynomial of degree {}'.format(n[i][2]))
    plt.ylim(min(y[i])-abs(min(y[i])/2), max(y[i])*1.5)  # 使y坐标的上下限关于y的最大最小值对称，且不受多项式拟合函数的振荡值影响
    plt.legend()
    plt.title('series length: {0}, degree of polyfit: {1}, {2}, {3}'.format(len(x[i]), n[i][0], n[i][1], n[i][2]))


#########################################################################
def polynomial_all(data, n):  # data为df['amou_crct']
    """
    param：
        data：理论销售金额，根据节日效应期长度，向左右各取2倍，共5倍长度，若遇节日则剔除补足，并在data全长内剔除周日效应
        n: 拟合次数
    return：
        fitts：经多项式拟合后的各点拟合值
    """
    if len(data) <= 3:
        raise Exception('历史数据过少，不能计算有效的节日效应系数')
    else:
        Y = np.array(data)
        X = np.linspace(1, len(Y), len(Y))  # 等差数列
        weights = (max(Y) * 2 - Y) / sum(max(Y) * 2 - Y)

        C = np.polyfit(X, Y, n, full=True, w=weights)  # C:多项式拟合的系数，其对应的次数从高到低排列
        p = np.poly1d(C[0])
        fitts = p(X)

    return fitts


x1 = list(range(1, 26))
y1 = np.random.rand(len(x1))
y1[int(len(x1)/2-2):int(len(x1)/2+1)] = 5
n = 3

fitts = polynomial_all(y1, n)

plt.figure('序列长度{0}个点，拟合次数为{1}'.format(len(x1), n))
plt.plot(x1, y1, '.', x1, fitts, '-')
plt.ylim(min(y1)-abs(min(y1)/10), max(y1)+max(y1)/10)
