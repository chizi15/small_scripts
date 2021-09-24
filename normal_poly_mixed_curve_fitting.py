"""整合概率密度函数和多项式，进行拟合，可兼顾密度函数的规律性和多项式的灵活性"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def nor_dis(x, mu, sigma, a, b, c):
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)) \
           + a*x**2 + b*x + c


mu, sigma, a, b, c = 0, 1, -0.01, 0.1, 1
xdata = np.linspace(-4, 4, 50)
y = nor_dis(xdata, mu, sigma, a, b, c)
np.random.seed(1729)
y_noise = 0.1 * np.random.random(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data: mu=%.3f, sigma=%.3f, a=%.3f, b=%.3f, c=%.3f'
                                   % (mu, sigma, a, b, c))

results = curve_fit(nor_dis, xdata, ydata)
print('parameters of mu, sigma are:', results[0])
plt.plot(xdata, nor_dis(xdata, *results[0]), 'r-',
         label='fit: mu=%.3f, sigma=%.3f, a=%.3f, b=%.3f, c=%.3f' % tuple(results[0]))

results = curve_fit(nor_dis, xdata, ydata, bounds=(0, [3., 1., 1, 1, 1]))
print('parameters of mu, sigma are:', results[0])
plt.plot(xdata, nor_dis(xdata, *results[0]), 'g--',
         label='fit: mu=%.3f, sigma=%.3f, a=%.3f, b=%.3f, c=%.3f' % tuple(results[0]))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
