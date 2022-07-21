import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


def safe_inv_adj(stockout, period, inventory, balanced_state=None, t=7, w=(1, 2), bias=1):
    """
    根据历史一段时间的缺货率、周转天数、安全库存，各自的最后一个元素是订货当日的缺货率、周转天数、原始安全库存，
    以及指定的或自动算出的平衡状态的缺货率和周转天数，给出新的动态安全库存。
    当程序开始起作用后，输入的历史安全库存inventory则包含越来越多的新动态安全库存，直到全部取代原有的安全库存。
    stockout, period, inventory不必等长，只要各自都有一定数量的元素能够进行统计即可。

    :param stockout: 一维数组，滑动窗口，如至少最近8周56个数值，缺货率历史值，使用百分数%，即比例×100.
    :param period: 一维数组，滑动窗口，如至少最近8周56个数值，周转天数历史值。
    :param inventory: 一维数组，滑动窗口，如至少最近8周56个数值，安全库存历史值。
    :param balanced_state: 指定的或算出的最优状态，包含两个元素的tuple，list，array，series等（缺货率，周转天数），为None时则自动计算；
        若opt_state为None，则表示该单品的历史缺货率和周转天数中没有符合筛选条件的情况发生。
    :param t: 该单品计算初始安全库存时所用的周期（即订货提前期+订货周期）。
    :param w: 缺货率和周转天数的权重之比。因缺货率为取值有限的离散变量，即表达的状态有限；且是最近一段时间的状态，即及时性不如周转天数；
        所以其权重应低于周转天数。
    :param bias: 缩放公式中带有分式时，为避免出现趋近于0的分母而产生振荡的结果，可将分子分母同时加一个偏置项bias；与做对数指数变换相比，对原分式的改变更小。

    :return:
        inventory_adj: 当前缺货率和周转天数对应的动态缩放后的安全库存。
        kind: 当前缺货率和周转天数对应的状态，
        ‘A’表示“缺货率高于平衡状态缺货率，周转天数低于或等于平衡状态周转天数，应增大安全库存”；
        ‘B’表示“缺货率低于或等于平衡状态缺货率，周转天数高于平衡状态缺货率，应减小安全库存”；
        ‘C’表示“缺货率高于平衡状态缺货率、周转天数也高于平衡状态缺货率，最坏情况，可能也较普遍，增大或减小安全库存视具体情况而定”；
        ‘D’表示“缺货率低于平衡状态缺货率、周转天数低于平衡状态缺货率，最好情况，可能也较罕见，沿用基准安全库存base_inv”；
        ‘E’表示“跟订货周期相比，历史数据较少，暂不进行安全库存缩放，沿用当前安全库存inventory_present”；
        ‘abnormal’表示“调整后安全库存为负，但库存可能不方便退货，则可保持负数，即让预测量减小”。
        base_inv: 根据当前安全库存、前两个订货点的安全库存加权得到的基准安全库存。
        inventory_lb: 由滑动窗口统计得到的安全库存下届。
        inventory_ub: 由滑动窗口统计得到的安全库存上届。
        period_lb: 由滑动窗口统计得到的周转天数下届。
        period_ub: 由滑动窗口统计得到的周转天数上届。
        stockout_lb: 由滑动窗口统计得到的缺货率下届。
        stockout_ub: 由滑动窗口统计得到的缺货率上届。
    """

    # 将对应序列最后一个元素指定当前缺货率、周转天数、安全库存，用于判断当前状态，以此为基础，结合历史统计值，进行安全库存缩放
    stockout_present, period_present, inventory_present = stockout[-1], period[-1], inventory[-1]
    if 2 * t + 1 <= len(inventory):
        # 根据历史数据动态统计缺货率、周转天数、安全库存的上下界
        stockout_ub = np.min([np.mean([np.mean(stockout) + 3 * np.std(stockout, ddof=1),
                                       np.percentile(stockout, 50) + 3 * (
                                               np.percentile(stockout, 68) - np.percentile(stockout, 50))]),
                              max(stockout), 100])
        stockout_lb = 0
        period_ub = np.min([np.mean([np.mean(period) + 3 * np.std(period, ddof=1),
                                     np.percentile(period, 50) + 3 * (
                                             np.percentile(period, 68) - np.percentile(period, 50))]),
                            max(period)])
        period_lb = np.min([max([np.mean([np.mean(period) - 3 * np.std(period, ddof=1),
                                          np.percentile(period, 50) - 3 * (
                                                  np.percentile(period, 50) - np.percentile(period, 32))]),
                                 0]), np.min([period])])
        inventory_ub = max([np.mean([np.mean(inventory) + 3 * np.std(inventory, ddof=1),
                                     np.percentile(inventory, 50) + 3 * (
                                             np.percentile(inventory, 68) - np.percentile(inventory, 50))]),
                            max(inventory)])
        inventory_lb = max([np.min([np.mean([np.mean(inventory) - 3 * np.std(inventory, ddof=1),
                                             np.percentile(inventory, 50) - 3 * (
                                                     np.percentile(inventory, 50) - np.percentile(inventory, 32))]),
                                    np.min([inventory])]), 0])
        base_inv = np.average(
            [inventory_present, np.mean(inventory[-t - 1:-t + 1]), np.mean(inventory[-2 * t - 1: -2 * t + 1])],
            weights=[1, 2, 1])
        if balanced_state is None:  # 该单品的历史缺货率和周转天数中没有符合筛选条件的情况发生
            balanced_state = [1, round(np.min([np.percentile(period, 50), np.mean(period)]), 2)]
        else:
            balanced_state[0] = round(max(balanced_state[0], 1), 2)
            balanced_state[1] = round(balanced_state[1], 2)
        if stockout_present > balanced_state[0] and period_present <= balanced_state[1]:  # 缺货率高，周转天数低于或等于最优周转天数，应增大安全库存
            delta = np.average([(stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias),
                                (balanced_state[1] - period_present + bias) / (balanced_state[1] - period_lb + bias)],
                               weights=w) * (inventory_ub - base_inv)
            inventory_adj = base_inv + delta
            kind = 'A'
        elif stockout_present <= balanced_state[0] and period_present > balanced_state[1]:
            # 缺货率低于或等于最优缺货率，周转天数高，应减小安全库存
            delta = np.average([-(stockout_present - balanced_state[0] - bias) / -(stockout_lb - balanced_state[0] - bias),
                                -(balanced_state[1] - period_present - bias) / -(balanced_state[1] - period_lb - bias)],
                               weights=w) * -(inventory_lb - base_inv)
            inventory_adj = base_inv - delta
            kind = 'B'
        elif stockout_present > balanced_state[0] and period_present > balanced_state[1]:  # 缺货率高、周转天数高，最坏情况，可能也较普遍
            delta_stockout = (stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias) * (
                    inventory_ub - base_inv)  # 为降低缺货率，应增加的安全库存量
            delta_period = (period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias) * (
                    base_inv - inventory_lb)  # 为降低周转天数，应减小的安全库存量
            delta = np.average([delta_stockout, -delta_period], weights=w)
            inventory_adj = base_inv + delta
            kind = 'C'
        else:  # 缺货率低、周转天数低，最好情况，可能也较罕见
            inventory_adj = base_inv
            kind = 'D'
        if inventory_adj < 0:  # 调整后安全库存为负，但库存可能不方便退货，则可保持负数，即让预测量减小
            # inventory_adj = 0
            kind = kind + ' - ' + 'abnormal'
        return round(inventory_adj, 4), kind, round(base_inv, 2), round(inventory_present, 2), round(inventory_lb, 2), \
               round(inventory_ub, 2), round(period_lb, 2), round(period_ub, 2), round(stockout_lb, 2), \
               round(stockout_ub, 2), balanced_state
    else:
        kind = 'E'
        return round(inventory_present, 4), kind, np.nan, round(inventory_present, 2), np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.nan, np.nan  # 相对于订货周期，历史数据较少，暂不进行安全库存缩放


safe_inventory = pd.read_csv('D:/codes/safe-inventory.csv')
info = safe_inventory.groupby(['organ', 'code']).describe()
print(info)
groups = safe_inventory.groupby(['organ', 'code'])
results = pd.DataFrame([])
for key, value in groups:
    print(key)
    result = safe_inv_adj(value['outofstockrate'].values*100, value['turnoverdays'].values, value['new_storage'].values)
    print(result)
    print("")
    results = pd.concat([results, pd.DataFrame([[key[0], key[1], result[0], result[1], result[2], result[3],
                                                result[4], result[5], result[6], result[7], result[8], result[9],
                                                result[10]]],
                                               columns=['organ', 'code', 'inventory_adj', 'kind', 'base_inv',
                                                        'inventory_present',  'inventory_lb', 'inventory_ub',
                                                        'period_lb', 'period_ub', 'stockout_lb', 'stockout_ub',
                                                        'balanced_state'])], ignore_index=True)
print(results.describe())
