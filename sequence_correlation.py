from scipy.stats import pearsonr, spearmanr, kendalltau, weightedtau, pointbiserialr
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score, r2_score
from matplotlib import pyplot as plt


# test for correlation coefficient, mutual_info_score, r2_score
x = np.linspace((-np.pi)*2, np.pi*2, 100, endpoint=True)
y1, y2 = np.sin(x), np.sin(x-np.pi/16)
# y1 = np.random.rand(10)
# y2 = np.random.rand(10)


def correlation(vector1, vector2):
    """
    scipy.stats.pearsonr == scipy.stats.pointbiserialr == np.corrcoef，三者等价。
    """

    print('vector1 and vector1 similarity -- correlation coefficient in scipy.stats:')
    print('{0} and {1} pearsonr: {2}'.format('vector1', 'vector1', pearsonr(vector1, vector1)[0]))
    print('{0} and {1} pointbiserialr: {2}'.format('vector1', 'vector1', pointbiserialr(vector1, vector1)[0]))
    print('{0} and {1} spearmanr: {2}'.format('vector1', 'vector1', spearmanr(vector1, vector1)[0]))
    print('{0} and {1} kendalltau: {2}'.format('vector1', 'vector1', kendalltau(vector1, vector1)[0]))
    print('{0} and {1} weightedtau: {2}'.format('vector1', 'vector1', weightedtau(vector1, vector1)[0]))

    print('vector1 and vector1 similarity -- correlation coefficient in numpy:')
    print('{0} and {1} pearsonr: {2}'.format('vector1', 'vector1', np.corrcoef(vector1, vector1)[0, 1]))

    print('vector1 and vector1 similarity -- mutual information score in sklearn.metrics:')
    print('{0} and {1} mutual_info_score: {2}'.format('vector1', 'vector1', mutual_info_score(vector1, vector1)))
    print('{0} and {1} adjusted_mutual_info_score: {2}'.format('vector1', 'vector1', adjusted_mutual_info_score(vector1, vector1)))
    print('{0} and {1} normalized_mutual_info_score: {2}'.format('vector1', 'vector1', normalized_mutual_info_score(vector1, vector1)))

    print('vector1 and vector1 similarity -- coefficient of determination in sklearn.metrics:')
    print('{0} and {1} coefficient of determination: {2}'.format('vector1', 'vector1', r2_score(vector1, vector1)))

    print('----------------------------------')

    print('vector1 and vector2 similarity -- correlation coefficient in scipy.stats:')
    print('{0} and {1} pearsonr: {2}'.format('vector1', 'vector2', pearsonr(vector1, vector2)[0]))
    print('{0} and {1} pointbiserialr: {2}'.format('vector1', 'vector2', pointbiserialr(vector1, vector2)[0]))
    print('{0} and {1} spearmanr: {2}'.format('vector1', 'vector2', spearmanr(vector1, vector2)[0]))
    print('{0} and {1} kendalltau: {2}'.format('vector1', 'vector2', kendalltau(vector1, vector2)[0]))
    print('{0} and {1} weightedtau: {2}'.format('vector1', 'vector2', weightedtau(vector1, vector2)[0]))

    print('vector1 and vector2 similarity -- correlation coefficient in numpy:')
    print('{0} and {1} pearsonr: {2}'.format('vector1', 'vector2', np.corrcoef(vector1, vector2)[0, 1]))

    print('vector1 and vector2 similarity -- mutual information score in sklearn.metrics:')
    print('{0} and {1} mutual_info_score: {2}'.format('vector1', 'vector2', mutual_info_score(vector1, vector2)))
    print('{0} and {1} adjusted_mutual_info_score: {2}'.format('vector1', 'vector2', adjusted_mutual_info_score(vector1, vector2)))
    print('{0} and {1} normalized_mutual_info_score: {2}'.format('vector1', 'vector2', normalized_mutual_info_score(vector1, vector2)))

    print('vector1 and vector2 similarity -- coefficient of determination in sklearn.metrics:')
    print('{0} and {1} coefficient of determination: {2}'.format('vector1', 'vector2', r2_score(vector1, vector2)))


def sequence_similarity(vector1, vector2):
    """vector1 and vector2 are both 1d data"""
    similarity = (pearsonr(vector1, vector2)[0] + spearmanr(vector1, vector2)[0] + kendalltau(vector1, vector2)[0] + weightedtau(vector1, vector2)[0] + r2_score(vector1, vector2)) / 5
    return similarity


correlation(y1, y2)
print(sequence_similarity(y1, y2))
# plt.figure('1')
# plt.plot(x, y1, c='g', marker='s', label='sin(x)')
# plt.plot(x, y2, c='r', marker='o', label='sin_shift(x)')
# plt.legend(loc='upper left', numpoints=2)
plt.figure('2')
plt.plot(y1, y2, 'bo')




from scipy.stats import hmean, gmean, normaltest
from sklearn.metrics import mean_squared_error


a = np.random.rand(1000) + 1e-3
print('调和平均:{:.3f}，几何平均:{:.3f}，算术平均:{:.3f}，均方根:{:.3f}'.format(hmean(a), gmean(a), a.mean(), mean_squared_error(a, np.zeros(len(a)), squared=False)), '\n'
    '调和平均 < 几何平均 < 算术平均 < 均方根:', hmean(a) < gmean(a) < a.mean() < mean_squared_error(a, np.zeros(len(a)), squared=False))


pts = 1000
np.random.seed(28041990)
a = np.random.normal(0, 1, size=pts)
b = np.random.normal(2, 1, size=pts)
x = np.concatenate((a, b))
k1, p1 = normaltest(a)
k2, p2 = normaltest(x)
alpha = 1e-3
print("p1 = {:g}".format(p1))
print("p2 = {:g}".format(p2))
if p1 < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
if p2 < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")


import statistics as st
import math
import numpy as np
import random
import sklearn
from scipy import stats


st.median_low([1, 3, 5, 7])
st.mean([1, 2, 3, 4, 4])
random.random()

# check scikit-learn version
print(sklearn.__version__)


# math.log(1.5**2, 1.5)
# np.log(1.5**2)/np.log(1.5)


# 几何平均数用于求连乘样本的均值，算术平均数用于求连加样本的均值
r1, r2, r3 = 1.05, 1.03, 1.022  # 各年利率
n1, n2, n3 = 1.5, 2.5, 1  # 各年利率持续时间

G = (r1**n1*r2**n2*r3**n3)**(1/(n1+n2+n3))
r_avg = G - 1
print('用几何级数计算平均年利率的误差：', (1+r_avg)**(n1+n2+n3) - r1**n1*r2**n2*r3**n3)  # 几何平均数的n次方等于总量

A = (n1*r1+n2*r2+n3*r3)/(n1+n2+n3)
r_avg = A - 1
print('用算术级数计算平均年利率的误差：', (1+r_avg)*(n1+n2+n3) - r1**n1*r2**n2*r3**n3)  # 算术平均数的n倍等于总量

