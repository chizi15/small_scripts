import numpy as np


def filter_data_normal(amount, final_amou):

    result_amou_stde = np.std(np.array(amount))
    result_amou_Q1 = np.percentile(np.array(amount), 10)
    result_amou_median = np.percentile(np.array(amount), 50)
    result_amou_Q3 = np.percentile(np.array(amount), 90)  # 春节、国庆、中秋有大爆发的天数假定为20天，占一年天数5%左右，再考虑其他情况的爆发，总共假定一年有10%的天数有爆发。

    result_amou_under = (result_amou_stde + result_amou_median - result_amou_Q1) / 2  # d下，距离，正数
    result_amou_on = (result_amou_stde + result_amou_Q3 - result_amou_median) / 2  # d上，距离，正数
    result_amou_l = result_amou_median - 3 * result_amou_under  # lb，下界，可正可负；不需要mean，因为拼接的原始序列(amount)是未经过处理的，造成mean偏大，故舍弃。
    result_amou_u = 3 * result_amou_on + result_amou_median  # ub，上界，正数；
    print(result_amou_median, result_amou_u, result_amou_l)

    # 假设final_amou的所需字段名也为amount
    final_amou_result = []
    final_amou_ = final_amou['amount']
    for f_amou in final_amou_:
        if f_amou >= result_amou_u:
            final_r = result_amou_median + result_amou_on + np.log(1 + f_amou - (result_amou_median + result_amou_on))
            print('highout')
            if final_r >= result_amou_u:
                final_r = result_amou_u
                print('highout_limit')
        elif f_amou <= result_amou_l:
            final_r = result_amou_median - result_amou_under - np.log(
                1 + result_amou_median - result_amou_under - f_amou)
            print('lowout')
            if final_r <= result_amou_l:
                final_r = result_amou_l
                print('highout_limit')
        else:
            final_r = f_amou

        final_amou_result.append(final_r)

    return final_amou_result
