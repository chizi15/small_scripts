# 导入类库
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import set_option

# 导入数据
filename = 'iris.data.csv'
names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
print()
print('"dataset"的数据类型是:', type(dataset))

# 显示数据维度
print()
l, r = dataset.shape
print('"dataset.shape"的类型是：{0}，行数的类型是：{1}，列数的类型是：{2}'
      .format(type(dataset.shape), type(l), type(r)))
print('数据维度: {0}行，{1}列' .format(l, r))
print()

# 查看数据的前10行
set_option('display.width', 250)
print('查看数据的前10行')
print(dataset.head(10))
print('查看数据的后10行')
print(dataset.tail(10))
print()

# 统计描述数据信息
print('统计描述数据信息')
print(dataset.describe())
print()

# 分类分布情况
print('查看“separ-length”这列数据的分类分布情况')
print(dataset.groupby('separ-length').size())
print('查看“separ-width”这列数据的分类分布情况')
print(dataset.groupby('separ-width').size())
print('查看“petal-length”这列数据的分类分布情况')
print(dataset.groupby('petal-length').size())
print('查看“petal-width”这列数据的分类分布情况')
print(dataset.groupby('petal-width').size())
print('查看“class”这列数据的分类分布情况')
print(dataset.groupby('class').size())
print()

# 箱线图
# 每列数据表示在不同坐标图中，则每张图纵坐标不同。
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()
# 每列数据表示在相同坐标图中方法1，则共用一个纵坐标值。
dataset.plot(kind='box', subplots=False, layout=(1, 1), sharex=False, sharey=False)
pyplot.show()
# 每列数据表示在相同坐标图中方法2，则共用一个纵坐标值。
dataset.boxplot()
pyplot.show()

# 直方图
# 每列数据表示在不同坐标图中，每张图的横坐标有相同的取值范围，纵坐标的取值范围不同。
dataset.plot(kind='hist', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()
# 每列数据表示在同一坐标图中，即共用同一横纵坐标。
dataset.plot(kind='hist', subplots=False, sharex=False, sharey=False)
pyplot.show()
# 每列数据表示在不同坐标图中，每张图的横、纵坐标取值范围不同。
dataset.hist()
pyplot.show()

# 密度图
# 每列数据表示在不同坐标图中，每张图的横、纵坐标取值范围不同。
dataset.plot(kind='density', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()
# 每列数据表示在同一坐标图中，即共用同一横纵坐标。
dataset.plot(kind='density', subplots=False, sharex=False, sharey=False)
pyplot.show()

# 散点矩阵图
scatter_matrix(dataset)
pyplot.show()

# 分离数据集
array = dataset.values
# print('展示dataset.values：')
# print(array)
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 1/3
seed = 4
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=validation_size, random_state=seed)

# help(train_test_split)

# 审查基本分类算法
models = {'LR': LogisticRegression(), 'LDA': LinearDiscriminantAnalysis(),
          'KNN': KNeighborsClassifier(),
          'CART': DecisionTreeClassifier(), 'NB': GaussianNB(), 'SVM': SVC()}
# 字典生成后其items的顺序就固定了
# print(models)
# print(models)
# print(models)

# 评估算法
print()
print('评估算法')
results = []
mean_std = []
kfold = KFold(n_splits=10, random_state=seed)
for key in models:
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))
    # print(type(mean))
    # print(cv_results.mean())
    # mean_num = float(cv_results.mean())
    # print(type(mean_num))
    mean_std.append(cv_results.mean()-cv_results.std())

print('type of "mean_std" is', type(mean_std))
print('"mean_std" is')
print(mean_std)
max_mean = max(mean_std)
print('max "mean_std" is:', max_mean)
max_mean_index = mean_std.index(max_mean)
print('max_mean_index is:', max_mean_index)

print('type of "models.keys()" is:', type(models.keys()))
print('"models.keys()" are')
print(models.keys())
# print(models.keys())
# print(models.keys())
print('type of "results" is:', type(results))
print('"results" are')
print(results)
# 箱线图比较算法
fig = pyplot.figure()
fig.suptitle('last figure: Algorithm Comparison')
ax = fig.add_subplot(111)
ax.set_xticklabels(models.keys())
pyplot.boxplot(results)
pyplot.show()

# 将之前作的图都显示出来
# pyplot.show()

# 使用评估数据集评估算法
print()
print('使用评估数据集评估算法')
models_values = list(models.values())
print('type of "models_values" is:', type(models_values))
print('"models_values" are')
print(models_values)
best_algorithm = models_values[max_mean_index]
print('"best_algorithm" is:', best_algorithm)
best_algorithm.fit(X=X_train, y=Y_train)
predictions = best_algorithm.predict(X_validation)
print('accuracy_score:', accuracy_score(Y_validation, predictions))
print('confusion_matrix')
print(confusion_matrix(Y_validation, predictions))
print('classification_report')
print(classification_report(Y_validation, predictions))
