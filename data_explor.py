import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# 设置字体为支持中文的字体，例如 SimHei（黑体）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负号显示问题

# import warnings
# warnings.filterwarnings("ignore")

quick_run = 100
if quick_run == 1: # 最快
    two_dim_kde = False
    time_series_heatmap = False
    other_for_loop = False
    pairplot = False
    decision_tree = False
elif quick_run == 10: # 适中
    two_dim_kde = False
    time_series_heatmap = True
    other_for_loop = True
    pairplot = True
    decision_tree = False
elif quick_run == 100: # 最慢
    two_dim_kde = True
    time_series_heatmap = True
    other_for_loop = True
    pairplot = True
    decision_tree = True
else: # 生成two_dim_kde和decision_tree
    two_dim_kde = True
    time_series_heatmap = False
    other_for_loop = False
    pairplot = False
    decision_tree = True

# # 选择需要读取的列
# columns_selected = [
#     "Date Time",
#     "MW.UNIT1@NET1",
#     "10PAB31CT301.UNIT1@NET1",
#     "10PAB31CT302.UNIT1@NET1",
#     "10PAB32CT301.UNIT1@NET1",
#     "10MAG10CT301.UNIT1@NET1",
#     "10MAG10CT302.UNIT1@NET1",
#     "10MAG20CT301.UNIT1@NET1",
#     "10MAG20CT302.UNIT1@NET1",
#     "10MAJ10AP001CE.UNIT1@NET1",
#     "10MAJ20AP001CE.UNIT1@NET1",
#     "10MAJ30AP001CE.UNIT1@NET1",
#     "10PAC10CF101.UNIT1@NET1",
#     "10MAG10CP102.UNIT1@NET1",
#     "10MAG20CP102.UNIT1@NET1",
# ]
# 每个文件从第40行开始读取数据
start_row = 39

# 获取当前文件绝对路径
current_file_path = __file__
# 获取当前路径的上一级目录
parent_folder_path = os.path.dirname(current_file_path)
# 指定文件夹名称
folder_name = "#1机冷端优化"
# 拼接完整的文件夹路径
folder_path = os.path.join(parent_folder_path, folder_name)

# 获取folder_path下所有的 Excel 文件路径，并按原始读取的顺序存储在列表中，而不是按文件名排序
excel_files = sorted(
    glob.glob(os.path.join(folder_path, "*.xls*")), key=os.path.getmtime
)
# # 将最后一个元素放到第一位，因为这张最新的编辑的表在时间上最靠前
# excel_files.insert(0, excel_files.pop())

# 创建文件夹，一定要用绝对路径，以免degub和console运行时路径不同
analysis_folder = f"data_exploration"
os.makedirs(f"{parent_folder_path}/{analysis_folder}", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/violin_boxplots", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/hist_kde", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/autocorrelation", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/lagplots", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/time_heatmap", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/2d_kde", exist_ok=True)
# os.makedirs(f"{parent_folder_path}/{analysis_folder}/violinplots", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/qqplots", exist_ok=True)
os.makedirs(f"{parent_folder_path}/{analysis_folder}/decision_tree", exist_ok=True)

# 读取所有 Excel 文件并存储在一个列表中
try:
    dataframes = [
        pd.read_excel(file, skiprows=start_row, engine="openpyxl")
        for file in excel_files
    ]
except Exception as e:
    print("\nError reading Excel files:", e, "\n使用xlrd进行读取...\n")
    dataframes = [
        pd.read_excel(file, skiprows=start_row, engine="xlrd") for file in excel_files
    ]

# 找出dataframes中空df的索引
empty_indices = [i for i, df in enumerate(dataframes) if df.empty]
# 检查dataframes中每个df的行数是否相同
row_counts = [len(df) for df in dataframes]
max_row_count = max(row_counts)
# 找出行数不等于max_row_count的df的索引
row_count_mismatch_indices = [
    i for i, count in enumerate(row_counts) if count != max_row_count
]
# 构建要保存的信息字符串
info_str = (
    f"\n总表数为：{len(dataframes)}个\n"
    f"行数小于最大行数的表的索引：{row_count_mismatch_indices}\n"
    f"它们的行数分别是：{[row_counts[idx] for idx in row_count_mismatch_indices]}\n"
    f"所以空表索引为：{empty_indices}\n"
    f"所以小于最大行数但不为空的表的索引为：{sorted(list(set(row_count_mismatch_indices) - set(empty_indices)))}\n"
)
# 打印信息
print(info_str)
# 将信息保存到文本文件
with open(f"{parent_folder_path}/{analysis_folder}/miss_data.txt", "w", encoding="utf-8") as file:
    file.write(info_str)


# 拼接成一个 DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.dropna(axis=1, how="all", inplace=True)
combined_df.dropna(axis=0, how="all", inplace=True)
combined_df.sort_values(by="Date Time", inplace=True)
combined_df.reset_index(drop=True, inplace=True)
# print(combined_df)

# df_selected = combined_df[columns_selected]
df_selected = combined_df.copy()
combined_df.to_csv(f"{parent_folder_path}/combined_data_origin.csv", index=False)

# 查看每一列的数据类型
print(f"\n原始数据类型：\n{df_selected.dtypes}\n")
# 查看缺失值个数
print(f"\n缺失值：\n{df_selected.isnull().sum()}\n")
# 将列名中带有.部分分割开，取.前面的部分作为新的列名
df_selected.columns = df_selected.columns.str.split(".").str[0]
df_selected["Date Time"] = pd.to_datetime(
    df_selected["Date Time"], format="%m/%d/%Y %H:%M:%S.%f", errors="coerce"
)
# 将Date Time列的所有日期增加12个月，减少1分钟1秒100毫秒
df_selected["Date Time"] += pd.Timedelta(
    days=365, hours=0, minutes=-1, seconds=-1, milliseconds=-100
)
# 将所有其他列转换为float64
for column in df_selected.columns[1:]:
    df_selected[column] = pd.to_numeric(df_selected[column], errors="coerce")
# 将df_selected中的缺失值用该值所在列的上下的值进行线性插值填充
df_selected.iloc[:, 1:] = df_selected.iloc[:, 1:].interpolate(
    method="linear", limit_direction="both"
)
# 给一个确定的随机数种子
np.random.seed(42)
# 为 df_selected 中的每个数值生成一个 0.95 到 1 之间的随机数
random_factors = np.random.uniform(0.95, 1.0, size=df_selected.iloc[:, 1:].shape)
# 将 df_selected 中的每个数值乘上对应的随机数
df_selected.iloc[:, 1:] = df_selected.iloc[:, 1:] * random_factors
print(f"\n处理后的数据类型：\n{df_selected.dtypes}\n")
# 查看哪些列有缺失值
print(f"\n缺失值：\n{df_selected.isnull().sum()}\n")
df_selected.to_csv(f"{parent_folder_path}/{analysis_folder}/masked_data.csv", index=False)
df_selected.to_excel(f"{parent_folder_path}/{analysis_folder}/masked_data.xlsx", index=False)
df_selected.to_parquet(f"{parent_folder_path}/{analysis_folder}/masked_data.parquet", index=False)


# 开始数据分析
# 创建一个 ExcelWriter 对象
with pd.ExcelWriter(f"{parent_folder_path}/{analysis_folder}/combined_stats.xlsx") as writer:
    # 描述性统计分析
    desc_stats = df_selected.describe()
    desc_stats.to_excel(writer, sheet_name="描述性统计")
    # 中位数
    median = df_selected.iloc[:, 1:].median()
    median.to_frame(name="P50").to_excel(writer, sheet_name="中位数")
    # 众数
    mode = df_selected.iloc[:, 1:].mode().iloc[0]
    mode.to_frame(name="mode").to_excel(writer, sheet_name="众数")
    # 方差
    variance = df_selected.iloc[:, 1:].var()
    variance.to_frame(name="variance").to_excel(writer, sheet_name="方差")
    # 偏度
    skewness = df_selected.iloc[:, 1:].skew()
    skewness.to_frame(name="skewness").to_excel(writer, sheet_name="偏度")
    # 峰度
    kurtosis = df_selected.iloc[:, 1:].kurt()
    kurtosis.to_frame(name="kurtosis").to_excel(writer, sheet_name="峰度")
    # 百分位数
    percentiles = df_selected.iloc[:, 1:].quantile([0.25, 0.5, 0.75])
    percentiles.to_excel(writer, sheet_name="百分位数")
    # 范围
    range_ = df_selected.iloc[:, 1:].max() - df_selected.iloc[:, 1:].min()
    range_.to_frame(name="极差").to_excel(writer, sheet_name="范围")
    # 变异系数
    cv = df_selected.iloc[:, 1:].std() / df_selected.iloc[:, 1:].mean()
    cv.to_frame(name="变异系数").to_excel(writer, sheet_name="变异系数")
    # 自相关系数
    autocorr = df_selected.iloc[:, 1:].apply(lambda x: x.autocorr())
    pd.DataFrame(autocorr, columns=["自相关系数"]).to_excel(
        writer, sheet_name="自相关系数"
    )
    # 相关矩阵
    correlation_matrix = df_selected.iloc[:, 1:].corr()
    correlation_matrix.to_excel(writer, sheet_name="相关矩阵")
    # 协方差矩阵
    covariance_matrix = df_selected.iloc[:, 1:].cov()
    covariance_matrix.to_excel(writer, sheet_name="协方差矩阵")


# 对这些字段分别作图以便分析，并保存
# 以date time的值为横轴，其他字段的值为纵轴，绘制折线图
df_selected.plot(x="Date Time", subplots=True, figsize=(30, 30))
plt.savefig(f"{parent_folder_path}/{analysis_folder}/time_series.png", format="png")
plt.close()

if other_for_loop:
    for column in df_selected.columns[1:]:

        # 小提琴图和箱线图
        # 创建绘图
        plt.figure(figsize=(10, 10))
        # 绘制小提琴图，设置 inner=None 去除内部的箱线图
        sns.violinplot(y=df_selected[column], color="skyblue", inner=None)
        # 在同一张图上绘制箱线图，设置宽度较小便于区分
        sns.boxplot(y=df_selected[column], width=0.1, color="orange", fliersize=2)
        # 添加标题和标签
        plt.title(f"{column} 小提琴图和箱线图")
        plt.xlabel("值")
        plt.ylabel(column)
        plt.savefig(
            f"{parent_folder_path}/{analysis_folder}/violin_boxplots/{column}_violin_boxplot.png",
            format="png",
        )
        plt.close()

        # 绘制直方图和 KDE 图，使用双纵轴
        fig, ax1 = plt.subplots(figsize=(10, 10))
        # 绘制直方图，使用 ax1
        sns.histplot(x=df_selected[column], bins=30, color="skyblue", alpha=0.6, ax=ax1)
        ax1.set_xlabel(column)
        ax1.set_ylabel("频数", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        # 创建第二个 y 轴，共享 x 轴
        ax2 = ax1.twinx()
        # 绘制 KDE 曲线，使用 ax2
        sns.kdeplot(x=df_selected[column], color="red", ax=ax2)
        ax2.set_ylabel("密度", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        # 添加标题
        plt.title(f"{column} 直方图和概率密度图")
        # 保存图像
        plt.savefig(
            f"{parent_folder_path}/{analysis_folder}/hist_kde/{column}_hist_kde_plot.png",
            format="png",
        )
        plt.close()

        # QQ图
        plt.figure()
        stats.probplot(df_selected[column].dropna(), dist="norm", plot=plt)
        plt.title(f"{column} QQ图")
        plt.savefig(f"{parent_folder_path}/{analysis_folder}/qqplots/{column}_qqplot.png", format="png")
        plt.close()

        # 自相关图
        plt.figure()
        autocorrelation_plot(df_selected[column])
        plt.title(f"{column} 自相关图")
        plt.savefig(
            f"{parent_folder_path}/{analysis_folder}/autocorrelation/{column}_autocorrelation_plot.png",
            format="png",
        )
        plt.close()

        # 滞后图
        plt.figure()
        lag_plot(df_selected[column])
        plt.title(f"{column} 滞后图")
        plt.savefig(
            f"{parent_folder_path}/{analysis_folder}/lagplots/{column}_lag_plot.png",
            format="png",
        )
        plt.close()

# 时间序列热力图
if time_series_heatmap:
    df_selected["Hour"] = df_selected["Date Time"].dt.hour
    df_selected["Day"] = df_selected["Date Time"].dt.date
    for column in df_selected.columns[1:-2]:
        pivot_table = df_selected.pivot_table(
            values=column, index="Day", columns="Hour", aggfunc="mean"
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap="viridis")
        plt.title(f"{column} 时间序列热力图")
        plt.savefig(
            f"{parent_folder_path}/{analysis_folder}/time_heatmap/{column}_time_series_heatmap.png",
            format="png",
        )
        plt.close()
    df_selected.drop(columns=["Hour", "Day"], inplace=True)

# 二维核密度图
if two_dim_kde:
    for i in range(1, len(df_selected.columns) - 1):
        for j in range(i + 1, len(df_selected.columns)):
            plt.figure()
            sns.kdeplot(
                x=df_selected[df_selected.columns[i]],
                y=df_selected[df_selected.columns[j]],
                fill=True,
                cmap="Blues",
            )
            plt.xlabel(f"{df_selected.columns[i]}")
            plt.ylabel(f"{df_selected.columns[j]}")
            plt.title(
                f"{df_selected.columns[i]}与{df_selected.columns[j]}的二维核密度图"
            )
            plt.savefig(
                f"{parent_folder_path}/{analysis_folder}/2d_kde/{df_selected.columns[i]}and{df_selected.columns[j]}_2d_kde_plot.png",
                format="png",
            )
            plt.close()

# 相关系数矩阵热力图
plt.figure(figsize=(20, 20))
corr = df_selected[df_selected.columns[1 : len(df_selected.columns) + 1]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("相关系数矩阵热力图")
plt.savefig(f"{parent_folder_path}/{analysis_folder}/correlation_heatmap.png", format="png")
plt.close()

# 成对关系图（散点图）
if pairplot:
    sns.pairplot(
        df_selected[df_selected.columns[1 : len(df_selected.columns) + 1]],
        diag_kind="kde",
        corner=True,
    )
    plt.savefig(f"{parent_folder_path}/{analysis_folder}/pairplot.png", format="png")
    plt.close()

# 主成分分析（PCA）
# 对数据进行标准化
features = df_selected.columns[1 : len(df_selected.columns) + 1]
x = df_selected[features].values
x = StandardScaler().fit_transform(x)

# PCA降维到2个主成分
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["主成分1", "主成分2"])
# 绘制散点图
plt.figure()
plt.scatter(principalDf["主成分1"], principalDf["主成分2"])
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.title("主成分分析（PCA）结果_2d")
plt.savefig(f"{parent_folder_path}/{analysis_folder}/pca_plot_2d.png", format="png")
plt.close()

# 3维主成分分析（PCA）
# PCA降维到3个主成分
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["主成分1", "主成分2", "主成分3"]
)
# 绘制三维散点图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(principalDf["主成分1"], principalDf["主成分2"], principalDf["主成分3"])
ax.set_xlabel("主成分1")
ax.set_ylabel("主成分2")
ax.set(zlabel="主成分3")
ax.set_title("主成分分析（PCA）结果_3d")
plt.savefig(f"{parent_folder_path}/{analysis_folder}/pca_plot_3d.png", format="png")
plt.close()

# 用决策树画出特征重要性
X = df_selected.drop(
    columns=[df_selected.columns[-1], df_selected.columns[-2], df_selected.columns[0]]
)
y_1 = df_selected[df_selected.columns[-1]]
y_2 = df_selected[df_selected.columns[-2]]

clf_1 = DecisionTreeRegressor()
clf_1 = clf_1.fit(X, y_1)
plt.figure(figsize=(10, 10))
importances = clf_1.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(len(indices)), importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title(f"各个自变量对于{df_selected.columns[-1]}的重要性（无对齐）")
plt.savefig(
    f"{parent_folder_path}/{analysis_folder}/decision_tree/{df_selected.columns[-1]}_feature_importance.png",
    format="png",
)
plt.close()

clf_2 = DecisionTreeRegressor()
clf_2 = clf_2.fit(X, y_2)
plt.figure(figsize=(10, 10))
importances = clf_2.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(len(indices)), importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title(f"各个自变量对于{df_selected.columns[-2]}的重要性（无对齐）")
plt.savefig(
    f"{parent_folder_path}/{analysis_folder}/decision_tree/{df_selected.columns[-2]}_feature_importance.png",
    format="png",
)
plt.close()

# 决策树可视化
if decision_tree:
    max_depth = 3

    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf_1,
        max_depth=max_depth,
        filled=True,
        feature_names=X.columns.tolist(),
        rounded=True,
    )
    plt.title(f"{df_selected.columns[-1]} 决策树")
    plt.savefig(
        f"{parent_folder_path}/{analysis_folder}/decision_tree/{df_selected.columns[-1]}_decision_tree.svg",
        format="svg",
    )
    plt.close()

    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf_2,
        max_depth=max_depth,
        filled=True,
        feature_names=X.columns.tolist(),
        rounded=True,
    )
    plt.title(f"{df_selected.columns[-2]} 决策树")
    plt.savefig(
        f"{parent_folder_path}/{analysis_folder}/decision_tree/{df_selected.columns[-2]}_decision_tree.svg",
        format="svg",
    )
    plt.close()

print("\n数据探索结束！\n")
