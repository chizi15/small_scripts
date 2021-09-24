"""
对没有进行异常值调整的原始销量序列，和进行异常值调整后的销量序列，分别取出这两条序列对应点的值作为amount_origin和amount_normal，
传入动态调值函数dyn_mod，根据基准值和差值的大小，进行分段非线性的自适应调整，使调整后的值更接近正常情况。
amount_origin是有异常点无空值的连续非负序列中的单个值，amount_normal是识别异常点并插值后的连续非负序列中的单个值。
fix_amount是以两条序列中对应各点为基础，进行动态调整后的值，最后将其组成等长序列fix_amount_results。
0 < beta_1,beta_2,beta_3 < 1，alpha_1是较大的倍数，alpha_2是较小的倍数。
"""

import numpy as np
import pandas as pd


def dyn_mod(amount_origin, amount_normal, alpha_1=20, alpha_2=2, beta_1=0.2, beta_2=0.1, beta_3=0.4):
    a, b1, b11, b2, b22, b3, b33, b4, c1, c2 = np.zeros(10)
    delta = amount_origin - amount_normal
    if delta == 0:
        if amount_origin < 0:
            raise ValueError('amount_origin和amount_normal应是非负数')
        a = 1
        fix_amount = amount_normal
    elif delta > 0:
        if amount_origin <= 0:
            raise ValueError('amount_origin应是正数')
        omega_1 = 1 + delta / amount_origin
        if amount_origin >= alpha_1 * amount_normal:
            b1 = 1
            fix_amount = amount_normal * np.exp(omega_1 - 1)
            if fix_amount >= (amount_normal + beta_1 * (amount_origin - amount_normal)):
                b11 = 1
                fix_amount = amount_normal * (np.exp(omega_1 - 1) + omega_1) / 2
        elif alpha_2 * amount_normal <= amount_origin < alpha_1 * amount_normal:
            b2 = 1
            fix_amount = amount_normal * (omega_1 + (1+np.log(omega_1))) / 2
            if fix_amount >= (amount_normal + beta_2 * (amount_origin - amount_normal)):
                b22 = 1
                fix_amount = amount_normal * (omega_1 + 1 + (1+np.log(omega_1))) / 3
        else:
            b3 = 1
            fix_amount = amount_normal * (1 + (1+np.log(omega_1))) / 2
            if fix_amount >= (amount_normal + beta_3 * (amount_origin - amount_normal)):
                b33 = 1
                fix_amount = amount_normal * (1 + 1 + (1+np.log(omega_1))) / 3
        if fix_amount >= amount_origin:
            b4 = 1
            fix_amount = amount_normal
        if fix_amount < 0:
            raise ValueError('amount_origin和/或amount_normal的值异常')
    else:
        if amount_normal <= 0:
            raise ValueError('amount_normal应是正数')
        omega_2 = 1 + delta / amount_normal
        c1 = 1
        fix_amount = amount_normal * (1 + np.exp(omega_2 - 1)) / 2
        if fix_amount <= amount_origin:
            c2 = 1
            fix_amount = amount_normal
        if fix_amount < 0:
            raise ValueError('amount_origin和/或amount_normal的值异常')
    return round(fix_amount, 3), a, b1, b11, b2, b22, b3, b33, b4, c1, c2


length = 10000
normal = pd.Series(np.ones(length)*10 + np.random.normal(0, 1, length))
origin = pd.Series(np.ones(length)*10 + np.random.normal(0, 10, length))
origin[origin < np.zeros(length)] = 0
results = []
for i, j in zip(normal, origin):
    results.append(dyn_mod(j, i))
df = pd.DataFrame(results)
fix_amount_results = df[0]
df.drop(columns=0, inplace=True)
df = df.T
counts = df.apply(lambda x: x.sum(), axis=1)
print('动态调整后的序列为：{0}, 各类调整数量总和为：{1}'.format(fix_amount_results, counts))
