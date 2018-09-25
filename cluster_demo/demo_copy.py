from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import metrics

# 导入数据
filename = 'wine.data'
names = ['class', 'Alcohol', 'MalicAcid', 'Ash', 'AlclinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids',
         'NonflayanoidPhenols', 'Proanthocyanins', 'ColorIntensiyt', 'Hue', 'OD280/OD315', 'Proline']
dataset = read_csv(filename, names=names)
print('type of dataset is:', type(dataset))
print('dataset.shape is:', dataset.shape)
print('dataset is')
print(dataset)
print()

dataset['class'] = dataset['class'].replace(to_replace=[1, 2, 3], value=[0, 1, 2])
array = dataset.values
X = array[:, 1:14]
y = array[:, 0]
print('type of X is:', type(X))
print('shape of X is:', X.shape)
print('X is')
print(X)
print('type of y is:', type(y))
print('shape of y is:', y.shape)
print('y is')
print(y)
print()

# 数据降维
pca = PCA()

# X_scale = StandardScaler().fit_transform(X)
# print('type of X_scale.shape is:', type(X_scale.shape))
# print('X_scale.shape is:', X_scale.shape)
# print('type of X_scale is:', type(X_scale))
# print('X_scale is')
# print(X_scale)
# print()

X_reduce = pca.fit_transform(scale(X))
print('type of X_reduce.shape is:', type(X_reduce.shape))
print('X_reduce.shape is:', X_reduce.shape)
print('type of X_reduce is:', type(X_reduce))
print('X_reduce is')
print(X_reduce)
print()

# 模型训练
model = KMeans(n_clusters=3)
model.fit(X_reduce)
labels = model.labels_
centers = model.cluster_centers_

# print('type of model.transform(X_reduce) is:', type(model.transform(X_reduce)))
# print('shape of model.transform(X_reduce) is:', model.transform(X_reduce).shape)
# print('model.transform(X_reduce) is')
# print(model.transform(X_reduce))
print('type of labels is:', type(labels))
print('shape of labels is:', labels.shape)
print('labels are')
print(labels)
print()

# 输出模型的准确度
print('%0.3f %0.3f %.3f %.3f %.3f' %
      (metrics.homogeneity_score(y, labels),
       metrics.completeness_score(y, labels),
       metrics.v_measure_score(y, labels),
       metrics.adjusted_rand_score(y, labels),
       metrics.adjusted_mutual_info_score(y,  labels)))
# 轮廓到中心的距离
       # metrics.silhouette_score(X_reduce, labels)))

# 绘制模型的分布图
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], X_reduce[:, 2], c=labels.astype(np.float))
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', color='red')
plt.show()
