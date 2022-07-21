import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


def safe_inv_adj(stockout, period, inventory, balanced_state=None, t=7, w=(1, 2), bias=1):
    stockout_present, period_present, inventory_present = stockout[-1], period[-1], inventory[-1]
    if 2 * t + 1 <= len(inventory):
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
        if balanced_state is None:
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
            delta = np.average([-(stockout_present - balanced_state[0] - bias) / -(stockout_lb - balanced_state[0] - bias),
                                -(balanced_state[1] - period_present - bias) / -(balanced_state[1] - period_lb - bias)],
                               weights=w) * -(inventory_lb - base_inv)
            inventory_adj = base_inv - delta
            kind = 'B'
        elif stockout_present > balanced_state[0] and period_present > balanced_state[1]:
            delta_stockout = (stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias) * (
                    inventory_ub - base_inv)
            delta_period = (period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias) * (
                    base_inv - inventory_lb)
            delta = np.average([delta_stockout, -delta_period], weights=w)
            inventory_adj = base_inv + delta
            kind = 'C'
        else:
            inventory_adj = base_inv
            kind = 'D'
        if inventory_adj < 0:
            # inventory_adj = 0
            kind = kind + ' - ' + 'abnormal'
        return round(inventory_adj, 4), kind, round(base_inv, 2), round(inventory_present, 2), round(inventory_lb, 2), \
               round(inventory_ub, 2), round(period_lb, 2), round(period_ub, 2), round(stockout_lb, 2), \
               round(stockout_ub, 2), balanced_state
    else:
        kind = 'E'
        return round(inventory_present, 4), kind, np.nan, round(inventory_present, 2), np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.nan, np.nan


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
