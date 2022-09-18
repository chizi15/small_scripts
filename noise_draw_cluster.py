import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


n = 1  # 选择主分类类型
n_clusters = 3  # 选择亚分类类别数
m = 10**2  # 设置外层次数，m越大得到的平均比例越可信；如当m=10**6时，得到的平均比例就可作为蒙特卡洛过程的一次试验；
# 再进行多次试验如1000次，就是蒙特卡洛模拟。
ratio = 1  # 设置抽取正态噪音时，正态分布标准差的倍数。当m足够大，得到的比例可信时，调节不同的倍数，就可用于敏感性分析。

type_list = ['铅钡', '高钾']  # 初始主分类
type = type_list[n]
iris = pd.read_excel(r'G:/coding/functions/raw_pca_data.xlsx', sheet_name=type)
X = iris.iloc[:, 19:23]  # 获取主成分0，1，2，3四列所有行的数据。

# 对主成分数据做minmax归一化
mM = MinMaxScaler()
X = mM.fit_transform(X)
X = pd.DataFrame(X)

label_list = []
if type == type_list[0]:
    estimator = KMeans(n_clusters=n_clusters, random_state=300)  # 构造聚类器
    estimator.fit(X)  # 聚类
    label_train = estimator.labels_  # 获取训练后的聚类标签
    label_train = pd.DataFrame(label_train)
    print('观察聚类标签是否降序排列，如不是须替换为降序类别标签，并以0为终点', '\n', label_train)
    # 铅钡的数据进去fit之后出来的类别标签直接就是降序210，与循环使index的迭代逻辑一致，就可以直接使用
    label_list.append(label_train)
    j = 0
    for _ in range(m):
        index = 0
        for i in range(n_clusters - 1, -1, -1):
            for j in X.columns:
                std_ = np.std(X[label_train.values == i][j], ddof=1)
                err = np.random.normal(0, ratio * std_)
                X.loc[index: X[label_train.values == i][j].index[-1], j] \
                    = X[label_train.values == i][j] + err  # 加噪音测试聚类的模型的鲁棒性
            index = X[label_train.values == i][j].index[-1] + 1
        label_pred = estimator.predict(X)  # 此处的X是加了噪音的，不等于fit时的X了
        label_pred = pd.DataFrame(label_pred)
        label_list.append(label_pred)
    data = pd.concat([label for label in label_list], axis=1)
    data.to_excel(r'G:/张驰资料/others/数模竞赛/2022-C/labels.xlsx')

else:
    estimator = KMeans(n_clusters=n_clusters, random_state=300)  # 构造聚类器
    estimator.fit(X)  # 聚类
    label_train = estimator.labels_  # 获取聚类标签
    label_train = pd.DataFrame(label_train)
    print('观察聚类标签是否降序排列，如不是须替换为降序类别标签，并以0为终点', '\n', label_train)
    # 高钾的数据进去fit之后出来的类别标签不是降序，是021，为了使index在循环的时候正确，与铅钡保持一致，应替换为降序类别标签210
    label_train.replace({0: 'a2', 2: 'a1', 1: 'a0'}, inplace=True)
    label_train.replace({'a2': 2, 'a1': 1, 'a0': 0}, inplace=True)
    label_list.append(label_train)
    j = 0
    for _ in range(m):
        index = 0
        for i in range(n_clusters - 1, -1, -1):
            for j in X.columns[:]:
                std_ = np.std(X[label_train.values == i][j], ddof=1)
                err = np.random.normal(0, ratio * std_)
                X.loc[index: X[label_train.values == i][j].index[-1], j] = X[label_train.values == i][j] + err
            index = X[label_train.values == i][j].index[-1] + 1
        label_pred = estimator.predict(X)  # 获取预测后的聚类标签；此处的X是加了噪音的，不等于fit时的X
        label_pred = pd.DataFrame(label_pred)
        # 前面fit的同一个estimator做预测时出来的类别标签还是一样的顺序，即021，也要替换为降序210，
        # df拼接的时候label_pred和label_train在索引上才保持一致
        label_pred.replace({0: 'a2', 2: 'a1', 1: 'a0'}, inplace=True)
        label_pred.replace({'a2': 2, 'a1': 1, 'a0': 0}, inplace=True)
        label_list.append(label_pred)
    data = pd.concat([label for label in label_list], axis=1)
    data.to_excel(r'G:/张驰资料/others/数模竞赛/2022-C/labels.xlsx')
