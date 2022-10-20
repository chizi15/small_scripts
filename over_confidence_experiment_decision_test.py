demand = 225
perish = 50
plus = list(range(0, 476))   # 476
minus = list(range(0, -204, -1))  # [-204,-205]
state = ['overshoot', 'shortage']
state = state[1]
order, profit = [], []

if state == 'overshoot':
    for _ in plus:
        print(_)
        order.append(demand + perish + _)
    print()
    for _ in range(len(order)):
        print(_)
        profit.append(100 * demand - 30 * order[_])
else:
    for _ in minus:
        print(_)
        order.append(demand + perish + _)
    print()
    for _ in range(len(order)):
        print(_)
        profit.append(70 * order[_] - 100 * perish)
print(profit[0], profit[-1])
