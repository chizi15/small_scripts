import numpy as np
from scipy import interpolate
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 用二维插值构造函数，----------------------------------------------------------------------------------------------
def s_curve_interp(n, x=(1, 10, 20, 30), y=(1e-5, 0.1, 0.9, 1)):
    """
    n：需要根据构造的插值函数得到对应y值的x坐标
    ratio：用于构造插值函数的点的x坐标，n值最好在x的范围内，因为插值函数不合适做外推
    period：用于构造插值函数的点的y坐标；x和y是成对的坐标，遵循奥卡姆剃刀原则，最少只需四个点，即三段插值函数，就可以构造任意大致规律的全局函数；若点数越多，构造出的函数形态就可以控制得越细致。
    return: 构造出的插值函数的x坐标为n时，对应的一个y坐标值
    """
    if x[0] <= n < x[1]:
        cs1 = interpolate.CubicSpline(x[:2], y[:2], bc_type=((1, y[1] / x[1]**2), (1, y[1] / x[1]**0.5)), extrapolate=False)
        r = cs1(n)
        if r < 0:
            r = cs1(x[0]+1)
    elif x[1] <= n < x[2]:
        cs2 = interpolate.CubicSpline(x[1:3], y[1:3], bc_type=((1, y[1] / x[1]**0.5), (1, (y[2]-y[1]) / (x[2]-x[1])**2)), extrapolate=False)
        r = cs2(n)
    else:
        cs3 = interpolate.CubicSpline(x[-2:], y[-2:], bc_type=((1, (y[2] - y[1]) / (x[2] - x[1]) ** 2), (1, (y[3] - y[2]) / (x[3] - x[2]) ** 2)), extrapolate=False)
        r = cs3(n)
        if r > 1:
            r = cs3(x[-1]-1)
    return float(r)


# 用于构造函数的坐标点
data_x = (1, 10, 20, 30)
data_y = (1e-5, 0.1, 0.9, 1)

for i in range(data_x[0], data_x[-1]):
    if s_curve_interp(i+1, data_x, data_y) - s_curve_interp(i, data_x, data_y) < 0:
        raise Exception('构造出的函数应不严格地单调递增，但此时在第 %s 个点处，s型曲线的值降低' % (i+1))
print('各个y坐标值：')
for i in range(data_x[0], data_x[-1]+1):
    print(s_curve_interp(i, data_x, data_y))

# x坐标间距越小，构造出的曲线就会显示得越光滑；因为配置的插值函数在临界点处原函数的左右极限相等，即连续，
# 左右一阶导数相等，即光滑，左右二阶导数相等，即凹凸性相同，所以函数在整个定义域上连续且光滑
# Array of evenly spaced values. For floating point arguments, the length of the result is `ceil((stop - start)/step)`.
# Because of floating point overflow, this rule may result in the last element of `out` being greater than `stop`.
xnew = np.arange(data_x[0], data_x[-1], 0.01)
ynew = [s_curve_interp(i, x=data_x, y=data_y) for i in xnew]
plt.figure()
plt.plot(data_x, data_y, 'o', xnew, ynew, '-')
plt.title('constructed interpolate points')
plt.show()
# 根据构造的函数生成归一化的权重w。因为每个w的分子与构造曲线的每个y值完全相同，而每个w的分母都是sum(period_new)，
# 所以w的分布完全由其分子确定，而其分子的分布与构造曲线y值的分布相同，所以w的分布特征与构造曲线的分布特征完全相同。
plt.figure()
w = [i/sum(ynew) for i in ynew]
plt.plot(list(range(len(w))), w)
plt.title('weights')
plt.show()


#########################################################################################################
# 用scipy.interpolate进行二维插值
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interpolate.interp1d(x, y, bounds_error=True)
f2 = interpolate.interp1d(x, y, kind='quadratic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.figure()
plt.plot(x, y, 'o', xnew, f(xnew), '--', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'quadratic'], loc='best')
plt.show()


# 用pd.Series.interpolate插值
dta = sm.datasets.co2.load_pandas().data.co2
plt.figure()
co2 = dta.interpolate(inplace=False)  # deal with missing values. see issue
co2.plot(color='r', label='interpolated')
dta.plot(color='g', label='origin')
plt.legend()
plt.show()


# 用三维插值构造函数，unstructured data, no sequenced, random------------------------------------------------------
def bivar_interp(x, y, z, xnew, ynew, xb=None, xe=None, yb=None, ye=None, kx=3, ky=3, s=None):
    """
    根据给定的三维数据点构造曲面，同时传入新的二维自变量得到新的应变量。

    :param x: 用于构造插值曲面的点的x坐标
    :param y: 用于构造插值曲面的点的y坐标
    :param z: 用于构造插值曲面的点的z坐标
    :param xnew: 根据已构造曲面，输入新的点的x坐标
    :param ynew: 根据已构造曲面，输入新的点的y坐标
    :param xb: x最小值，输入的x序列不能低于该下限，用于构成定义域的边界
    :param xe: x最大值，输入的x序列不能高于该上限，用于构成定义域的边界
    :param yb: y最小值，输入的y序列不能低于该下限，用于构成定义域的边界
    :param ye: y最大值，输入的y序列不能高于该上限，用于构成定义域的边界
    :param kx: x维度样条的灵活性，取值1~5，越大越灵活，插值或拟合时对给定点的逼近程度越高，但在非给定点的振荡性越大；
    :param ky: y维度样条的灵活性，取值1~5，越大越灵活，插值或拟合时对给定点的逼近程度越高，但在非给定点的振荡性越大；
        (kx+1)*(ky+1) <= len(x)
    :param s: 非负平滑因子，取值越大曲面越平滑，越近0对给定点的逼近程度越高；当s=0时则为完全插值，不具有拟合特性。
        通常取值为[len(x)-np.sqrt(2*len(x)), len(x)+np.sqrt(2*len(x))]，当kx，ky取值越大时，s当取值越大，反之s应越趋近0.

    :return: 新输入点对应的z值
    """

    try:
        if (kx+1)*(ky+1) > len(x):
            raise Exception('len(x)应 >= (kx+1)*(ky+1)')
        if s is None:
            surf_func = interpolate.bisplrep(x, y, z, xb=xb, xe=xe, yb=yb, ye=ye, kx=kx, ky=ky,
                                             s=len(x) + np.sqrt(2 * len(x)))
        else:
            surf_func = interpolate.bisplrep(x, y, z, xb=xb, xe=xe, yb=yb, ye=ye, kx=kx, ky=ky, s=max(s, 0))
        znew = interpolate.bisplev(xnew, ynew, surf_func)
        return znew
    except Exception as e:
        print(e)


x = 100*np.random.random(30)
y = x
z = 10*np.random.random(30)

xnew = x
ynew = y
znew = []
for i in range(len(xnew)):
    znew.append(bivar_interp(x,y,z,xnew[i],ynew[i], kx=3, ky=3, s=0))

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z, edgecolor='red')
ax.plot3D(xnew,ynew,znew)
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z, edgecolor='red')
xnewmesh, ynewmesh = np.meshgrid(xnew, ynew)
znewmesh = np.meshgrid(znew, znew)
ax.plot_surface(xnewmesh, ynewmesh, znewmesh[0])
plt.show()

print('各点偏差比：', '\n', (z-znew)/z, '\n')
print('平均绝对偏差比：', '\n', sum(abs((z-znew)/z)) / len(z), '\n')


# 用三维插值构造函数，structured data, sequenced------------------------------------------------------------------------------
def Rosenbrock(x):
    x = np.asarray(x)
    r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
    return r


x = np.linspace(-1, 1, 30)
X, Y = np.meshgrid(x, x)  # X,Y是互为转置的二维数组，以铺满三维坐标轴的水平面
interp_func = interpolate.RectBivariateSpline(X[0],X[0],Rosenbrock([X, Y])+1, kx=3, ky=3, s=len(x)**2)

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Rosenbrock([X, Y])+1, edgecolor='red')
ax.plot_surface(X, Y, interp_func(X[0], X[0]))
plt.show()

print('各点偏差比：', '\n', (Rosenbrock([X, Y])+1 - interp_func(X[0], X[0])) / (Rosenbrock([X, Y])+1), '\n')
print('平均绝对偏差比：', '\n', sum(sum(abs((Rosenbrock([X, Y])+1 - interp_func(X[0], X[0])) / (Rosenbrock([X, Y])+1))))
                            / (len(X)*len(Y)), '\n')
