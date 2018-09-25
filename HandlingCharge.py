"""
1. 手续费即利息，手续费率即利率。
2. 要以借入方实际付出的手续费作为计算真实利率的基础，不能以借出方显示的手续费率作为计算其他利率的基础。
3. 对于大部分金融机构：（显示的）每期手续费率(%) = 每期手续费 / 本金 * 100，但借入方实际付出的手续费率比这高；
                        总手续费 = 每期手续费 * 期数，该方法没有计算手续费的时间价值。
4. 为什么每期分期手续费和总续费看起来低，而实际付出的利率高？因为还款通常是每期定额还款，
（或者定每期本金变每期手续费和每期总额，或者定总额变本金和手续费，或者三者都固定）
所以越靠前还款的期间本金的借用时间越短，越靠后的还款期间本金的借用时间越长，折算之后的真实利率就高了。
5. 由于日利率较小，通常扩大10000倍表示，即表示为万分数（‱），
月利率和年利率通常扩大100倍表示，即表示为百分数（%）。
6. 以下计算采用支付宝借呗花呗、微信微粒贷、京东金条（按日计息）的原则来统一计算标准。
因为上述三类小额贷款给出的计算方法遵循两大原则：一，借了多少钱借了多久就付多少利息；
二，在遵循第一条原则的前提下，根据还款方式的不同，从借出方的角度出发，
形成了只计算本金的时间价值，即对本金计算利息，不计算手续费的时间价值的计算方法。
"""

"在每个实现单独功能的程序块中要将变量、列表等初始化，以免重名而错误调用。"


def hc2hcr(handling_charge, capital, n):
    """精确，(每期)手续费 转 (每期)手续费率(%)，前提：每期等额偿还本金及利息(即手续费)"""
    """正确算法："""
    m = 0
    for i in range(1, n + 1):
        m += i
    handling_charge_rate = (handling_charge * n ** 2) / (capital * m) * 100
    """错误算法："""
    # handling_charge_rate = handling_charge / capital * 100
    return handling_charge_rate


def hc2dr(handling_charge, capital, n):
    """近似，(每期)手续费 转 日利率(‱)，前提：每期等额偿还本金及利息(即手续费)"""
    day_rate = hc2hcr(handling_charge, capital, n) / 30 * 100
    return day_rate


def hc2ar(handling_charge, capital, n):
    """近似，(每期)手续费 转 年利率(%)，前提：每期等额偿还本金及利息(即手续费)"""
    annual_rate = hc2hcr(handling_charge, capital, n) * 12
    return annual_rate


def hcr2dr(handling_charge_rate):
    """(每期)手续费率(%) 转 日利率(‱)"""
    day_rate = handling_charge_rate / 30 * 100
    return day_rate


def hcr2ar(handling_charge_rate):
    """(每期)手续费率(%) 转 年利率(‱)"""
    annual_rate = handling_charge_rate * 12
    return annual_rate


def dr2ar(day_rate):
    """日利率(‱) 转 年利率(%)"""
    annual_rate = day_rate * 365 / 100
    return annual_rate


print('交通分期，每期等额偿还本金及手续费')
num = [3, 6, 9, 12, 18, 24]
handling_charge = [42.51, 36.57, 32.91, 32.91, 32.91, 32.91]
capital = 4572.04
handling_charge_rate = []
dr = []
ar = []
if len(num) == len(handling_charge):
    for item in num:
        # 属于每期等额偿还本金及手续费的情况，将(每期)分期手续费转化为实际(每期)分期手续费率(%)
        index = num.index(item)
        handling_charge_rate.append(hc2hcr(handling_charge=handling_charge[index], capital=capital, n=item))
    for hcr in handling_charge_rate:
        # 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
        dr.append(hcr2dr(hcr))
        ar.append(hcr2ar(hcr))
    for index in range(len(num)):
        print('若分%s期,则实际每期手续费率为：%.2f%%，日利率为：%.2f‱，年利率为：%.2f%%'
              % (num[index], handling_charge_rate[index], dr[index], ar[index]))
else:
    print('期数有错')
print()

print('招商分期，每期等额偿还本金及手续费')
num = [2, 3, 6, 10, 12, 18, 24, 36]
handling_charge = [51.88, 46.69, 38.91, 36.31, 34.24, 35.28, 35.28, 35.28]
capital = 5187.59
handling_charge_rate = []
dr = []
ar = []
if len(num) == len(handling_charge):
    for item in num:
        # 属于每期等额偿还本金及手续费的情况，将(每期)分期手续费转化为实际(每期)分期手续费率(%)
        index = num.index(item)
        handling_charge_rate.append(hc2hcr(handling_charge=handling_charge[index], capital=capital, n=item))
    for hcr in handling_charge_rate:
        # 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
        dr.append(hcr2dr(hcr))
        ar.append(hcr2ar(hcr))
    for index in range(len(num)):
        print('若分%s期,则实际每期手续费率为：%.2f%%，日利率为：%.2f‱，年利率为：%.2f%%'
              % (num[index], handling_charge_rate[index], dr[index], ar[index]))
else:
    print('期数有错')
print()

print('广发分期，每期等额偿还本金及手续费')
num = 12
handling_charge = 111.36
capital = 19200
# 属于每期等额偿还本金及手续费的情况，将(每期)分期手续费转化为实际(每期)分期手续费率(%)
handling_charge_rate = hc2hcr(handling_charge=handling_charge, capital=capital, n=num)
# 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
dr = hcr2dr(handling_charge_rate)
ar = hcr2ar(handling_charge_rate)
print('若分%s期，则分期手续费率为：%.2f%%，日利率为：%.2f‱，年利率为：%.2f%%' % (num, handling_charge_rate, dr, ar))
print()

print('中信分期，每期等额偿还本金，手续费在第一期末一次性付清')
num = [6, 9, 12, 18, 24, 36]
# 假定资金的年平均收益率取为3.5%
average_annual_rate = 3.5
month_rate = average_annual_rate / 100 / 12

print('单笔分期：')
total_handling_charge_single = [57.55, 82.01, 105.03, 161.87, 215.82, 323.73]
capital_single = 1199
handling_charge_rate_single = []
handling_charge_single = []
dr_single = []
ar_single = []
print('当个人资金年收益率为：%.2f%%，本金为：%.2f时' % (average_annual_rate, capital_single))
if len(num) == len(total_handling_charge_single):
    for item in num:
        index = num.index(item)
        # 因为手续费在第一期末一次性付清，所以应按照复利计息方式将总手续费折算成每期等额手续费
        handling_charge_single.append(total_handling_charge_single[index] * month_rate * (1 + month_rate) ** (item - 1)
                                      / ((1 + month_rate) ** item - 1))
        # 将总手续费折算后即属于每期等额偿还本金及手续费的情况，
        # 可用“hc2hcr”函数将(每期)分期手续费转化为实际(每期)分期手续费率(%)
        handling_charge_rate_single.append(hc2hcr(handling_charge=handling_charge_single[index],
                                                  capital=capital_single, n=item))
    for hcr in handling_charge_rate_single:
        # 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
        dr_single.append(hcr2dr(hcr))
        ar_single.append(hcr2ar(hcr))
    for index in range(len(num)):
        print('若分%s期，则折算每期手续费为：%.2f，实际每期手续费率为：%.2f%%，日利率为：%.2f‱，年利率为：%.2f%%'
              % (num[index], handling_charge_single[index], handling_charge_rate_single[index],
                 dr_single[index], ar_single[index]))
else:
    print('期数有错')

print('账单分期：')
total_handling_charge_bill = [566.62, 807.43, 1034.08, 1593.61, 2124.82, 3187.22]
capital_bill = 11804.53
handling_charge_rate_bill = []
handling_charge_bill = []
dr_bill = []
ar_bill = []
print('当个人资金年收益率为：%.2f%%，本金为：%.2f时' % (average_annual_rate, capital_bill))
if len(num) == len(total_handling_charge_bill):
    for item in num:
        index = num.index(item)
        # 因为手续费在第一期末一次性付清，所以应按照复利计息方式将总手续费折算成每期等额手续费
        handling_charge_bill.append(total_handling_charge_bill[index] * month_rate * (1 + month_rate) ** (item - 1)
                                    / ((1 + month_rate) ** item - 1))
        # 将总手续费折算后即属于每期等额偿还本金及手续费的情况，
        # 可用“hc2hcr”函数将(每期)分期手续费转化为实际(每期)分期手续费率(%)
        handling_charge_rate_bill.append(hc2hcr(handling_charge=handling_charge_bill[index],
                                                capital=capital_bill, n=item))
    for hcr in handling_charge_rate_bill:
        # 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
        dr_bill.append(hcr2dr(hcr))
        ar_bill.append(hcr2ar(hcr))
    for index in range(len(num)):
        print('若分%s期，则折算每期手续费为：%.2f，实际每期手续费率为：%.2f%%，日利率为：%.2f‱，年利率为：%.2f%%'
              % (num[index], handling_charge_bill[index], handling_charge_rate_bill[index],
                 dr_bill[index], ar_bill[index]))
else:
    print('期数有错')
print()

print('京东白条（还款时分期）：')
num = [3, 6, 12, 24]
handling_charge = 11.63
capital = 1550.43
handling_charge_rate = []
dr = []
ar = []
for i in num:
    # 将(每期)分期手续费转化为(每期)分期手续费率(%)
    handling_charge_rate.append(hc2hcr(handling_charge=handling_charge, capital=capital, n=i))
for hcr in handling_charge_rate:
    # 将(每期)分期手续费率(%)转化为日利率(‱)及年利率(%)
    dr.append(hcr2dr(hcr))
    ar.append(hcr2ar(hcr))
for index in range(len(num)):
    print('若分%s期,实际每期手续费率为：%.2f%%，则日利率为：%.2f‱，年利率为：%.2f%%'
          % (num[index], handling_charge_rate[index], dr[index], ar[index]))
print()

print('蚂蚁借呗：')
# 日利率(‱)转年利率(%)
day_rate = 3.5
dr2ar(day_rate)
print('当日利率为：%.2f‱时，年利率为：%.2f%%' % (day_rate, dr2ar(day_rate)))
print()

print('微信微粒贷：')
# 日利率(‱)转年利率(%)
day_rate = 2.5
dr2ar(day_rate)
print('当日利率为：%.2f‱时，年利率为：%.2f%%' % (day_rate, dr2ar(day_rate)))
print()

print('京东金条按日计息：')
# 日利率(‱)转年利率(%)
day_rate = 5 * 0.9
dr2ar(day_rate)
print('当日利率为：%.2f‱时，年利率为：%.2f%%' % (day_rate, dr2ar(day_rate)))

print('京东金条按月计息：')
# 手续费转年利率(%)，不计算手续费的时间价值
handling_charge_rate = hc2hcr(handling_charge=75, capital=10000, n=3)
print('金条按月计息年利率：%.2f%%' % (hcr2ar(handling_charge_rate)))
print()
