import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


def safe_inv_adj(stockout, period, inventory, balanced_state=(8, 25), plb=0, t=7, w=(1, 2), bias=1, critical_d=10, ratio=1/3):
    
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
        period_lb = max([np.mean([np.mean(period) - 3 * np.std(period, ddof=1),
                                          np.percentile(period, 50) - 3 * (
                                                  np.percentile(period, 50) - np.percentile(period, 32))]),
                                 plb])
        inventory_ub = min([np.mean([np.mean(inventory) + 3 * np.std(inventory, ddof=1),
                                     np.percentile(inventory, 50) + 3 * (
                                             np.percentile(inventory, 68) - np.percentile(inventory, 50))]),
                            max(inventory)])
        inventory_lb = max([np.min([np.mean([np.mean(inventory) - 3 * np.std(inventory, ddof=1),
                                             np.percentile(inventory, 50) - 3 * (
                                                     np.percentile(inventory, 50) - np.percentile(inventory, 32))]),
                                    np.min([inventory])]), 0])
        if t < critical_d:
            base_inv = np.average(
                [inventory_present, np.mean(inventory[-t - 1:-t + 1]), np.mean(inventory[-2 * t - 1: -2 * t + 1])],
                weights=[1, 2, 1])
        else:
            base_inv = np.average(
                [inventory_present, np.mean(inventory[-t - ratio*t:-t + ratio*t]),
                 np.mean(inventory[-2 * t - ratio*t: -2 * t + ratio*t])], weights=[1, 2, 1])
#         base_inv = np.average(
#             [inventory_present, np.mean(inventory[-t - 1:-t + 1]), np.mean(inventory[-2 * t - 1: -2 * t + 1])],
#             weights=[1, 2, 1])
        if balanced_state is None:
            balanced_state = [1, round(max([np.min([np.percentile(period, 50), np.mean(period)]), plb]), 2)]
        else:
            balanced_state = list(balanced_state)
            balanced_state[0] = round(max(balanced_state[0], 1), 2)
            balanced_state[1] = round(max(balanced_state[1], plb), 2)
        if stockout_present > balanced_state[0] and period_present <= balanced_state[1]:
            if stockout_ub > balanced_state[0] and balanced_state[1] > period_lb:
                # delta = np.average([(stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias),
                #                     (balanced_state[1] - period_present + bias) / (balanced_state[1] - period_lb + bias)],
                #                    weights=w) * max(inventory_ub - base_inv, 0)
                delta = min((stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias), 1)\
                        * max(inventory_ub - base_inv, 0)
                kind = 'A1'
            elif stockout_ub <= balanced_state[0] and balanced_state[1] > period_lb:
                # delta = (balanced_state[1] - period_present + bias) / (balanced_state[1] - period_lb + bias)\
                #         * max(inventory_ub - base_inv, 0)
                delta = min((stockout_present - balanced_state[0] + bias + stockout_present - balanced_state[0]) \
                        / (abs(stockout_ub - balanced_state[0]) + bias + stockout_present - balanced_state[0]), 1) \
                        * max(inventory_ub - base_inv, 0)
                kind = 'A2'
            elif stockout_ub > balanced_state[0] and balanced_state[1] <= period_lb:
                delta = min((stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias), 1)\
                        * max(inventory_ub - base_inv, 0)
                kind = 'A3'
            else:
                delta = 0
                kind = 'A4'
            inventory_adj = base_inv + delta
            inventory_adj = max(inventory_adj, inventory_present)
        elif stockout_present <= balanced_state[0] and period_present > balanced_state[1]:
            if balanced_state[0] - stockout_lb > 0 and period_ub - balanced_state[1] > 0:
                # delta = np.average([(balanced_state[0] - stockout_present + bias) / (balanced_state[0] - stockout_lb + bias),
                #                     (period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias)],
                #                    weights=w) * max(-(inventory_lb - base_inv), 0)
                delta = min((period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias), 1) \
                        * max(-(inventory_lb - base_inv), 0)
                kind = 'B1'
            elif balanced_state[0] - stockout_lb <= 0 and period_ub - balanced_state[1] > 0:
                delta = min((period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias), 1)\
                        * max(-(inventory_lb - base_inv), 0)
                kind = 'B2'
            elif balanced_state[0] - stockout_lb > 0 and period_ub - balanced_state[1] <= 0:
                # delta = (balanced_state[0] - stockout_present + bias) / (balanced_state[0] - stockout_lb + bias)\
                #         * max(-(inventory_lb - base_inv), 0)
                delta = min((period_present - balanced_state[1] + bias + period_present - balanced_state[1]) \
                        / (abs(period_ub - balanced_state[1]) + bias + period_present - balanced_state[1]), 1)\
                        * max(-(inventory_lb - base_inv), 0)
                kind = 'B3'
            else:
                delta = 0
                kind = 'B4'
            delta = - delta
            inventory_adj = base_inv + delta
            inventory_adj = min(inventory_adj, inventory_present)
        elif stockout_present > balanced_state[0] and period_present > balanced_state[1]:
            if stockout_ub - balanced_state[0] > 0 and period_ub - balanced_state[1] > 0:
                delta_stockout = min((stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias), 1) \
                                 * max(inventory_ub - base_inv, 0) 
                delta_period = min((period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias), 1) \
                               * max(base_inv - inventory_lb, 0) 
                delta = np.average([delta_stockout, -delta_period], weights=w)
                kind = 'C1'
            elif stockout_ub - balanced_state[0] > 0 and period_ub - balanced_state[1] <= 0:
                delta_stockout = min((stockout_present - balanced_state[0] + bias) / (stockout_ub - balanced_state[0] + bias), 1) \
                                 * max(inventory_ub - base_inv, 0) 
                delta_period = min((period_present - balanced_state[1] + bias + period_present - balanced_state[1])
                                   / (abs(period_ub - balanced_state[1]) + bias + period_present - balanced_state[1]), 1) \
                               * max(base_inv - inventory_lb, 0)  
                delta = np.average([delta_stockout, -delta_period], weights=w)
                kind = 'C2'
            elif stockout_ub - balanced_state[0] <= 0 and period_ub - balanced_state[1] > 0:
                delta_stockout = min((stockout_present - balanced_state[0] + bias + stockout_present - balanced_state[0])
                                     / (abs(stockout_ub - balanced_state[0]) + bias + stockout_present - balanced_state[0]), 1) \
                                 * max(inventory_ub - base_inv, 0)  
                delta_period = min((period_present - balanced_state[1] + bias) / (period_ub - balanced_state[1] + bias), 1) \
                               * max(base_inv - inventory_lb, 0)  
                delta = np.average([delta_stockout, -delta_period], weights=w)
                kind = 'C3'
            else:
                delta = 0
                kind = 'C4'
            inventory_adj = base_inv + delta
        else: 
            delta = 0
            inventory_adj = base_inv
            kind = 'D'
        if inventory_adj < 0:  
            # inventory_adj = 0
            kind = kind + '-' + 'abnormal'
        return round(inventory_adj, 4), kind, round(base_inv, 2), delta, round(inventory_present, 2), round(inventory_lb, 2), \
               round(inventory_ub, 2), round(period_lb, 2), round(period_ub, 2), round(stockout_lb, 2), \
               round(stockout_ub, 2), balanced_state
    else:
        kind = 'E'
        return round(inventory_present, 4), kind, np.nan, np.nan, round(inventory_present, 2), np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.nan, np.nan  


safe_inventory = pd.read_csv('D:/codes/safe-inventory-2.csv')
info = safe_inventory.groupby(['organ', 'code']).describe()
print(info, '\n')
groups = safe_inventory.groupby(['organ', 'code'])
results = pd.DataFrame([])
for key, value in groups:
    print(key)
    result = safe_inv_adj(value['stockoutrate'].values*100, value['turnoverdays'].values, value['new_storage'].values,
                          plb=value['stkdayslow'].values[0], t=value['ord_day'].values[0])
    print(result)
    print("")
    results = pd.concat([results, pd.DataFrame([[key[0], key[1], result[0], result[1], result[2], result[3],
                                                result[4], result[5], result[6], result[7], result[8], result[9],
                                                result[10], result[11]]],
                                               columns=['organ', 'code', 'inventory_adj', 'kind', 'base_inv', 'delta',
                                                        'inventory_present',  'inventory_lb', 'inventory_ub',
                                                        'period_lb', 'period_ub', 'stockout_lb', 'stockout_ub',
                                                        'balanced_state'])], ignore_index=True)
print(results.describe(), '\n')
print(results[results['inventory_adj'] < 0], '\n')
print(results[results['inventory_adj'] > 20], '\n')
