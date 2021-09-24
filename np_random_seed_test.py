import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


freq = 'D'
t0 = '2020-01-01'
data_length = 28
num_ts = 2
period = 14
different_type = 'no_seed'  # one of 'no_seed', 'seed_outside_loop', 'seed_before_every_random'

if different_type == 'no_seed':
    series = []
    for k in range(num_ts):
        level = 10 * np.random.rand()
        print(level/10)
        seas_amplitude = 0.1 + 0.3*np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude*np.sin(time_ticks*(2*np.pi)/period)
        noise = 0.1*np.random.randn(data_length)
        print(noise/0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('no_seed')
        plt.show()

    series = []
    for k in range(num_ts):
        level = 10 * np.random.rand()
        print(level/10)
        seas_amplitude = 0.1 + 0.3*np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude*np.sin(time_ticks*(2*np.pi)/period)
        noise = 0.1*np.random.randn(data_length)
        print(noise/0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('no_seed')
        plt.show()

elif different_type == 'seed_outside_loop':
    np.random.seed(0)
    series = []
    for k in range(num_ts):
        level = 10 * np.random.rand()
        print(level / 10)
        seas_amplitude = 0.1 + 0.3 * np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
        noise = 0.1 * np.random.randn(data_length)
        print(noise / 0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('seed_outside_loop')
        plt.show()

    np.random.seed(0)
    series = []
    for k in range(num_ts):
        level = 10 * np.random.rand()
        print(level / 10)
        seas_amplitude = 0.1 + 0.3 * np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
        noise = 0.1 * np.random.randn(data_length)
        print(noise / 0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('seed_outside_loop')
        plt.show()

elif different_type == 'seed_before_every_random':
    series = []
    for k in range(num_ts):
        np.random.seed(0)
        level = 10 * np.random.rand()
        print(level / 10)
        np.random.seed(0)
        seas_amplitude = 0.1 + 0.3 * np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
        np.random.seed(0)
        noise = 0.1 * np.random.randn(data_length)
        print(noise / 0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('seed_before_every_random')
        plt.show()

    series = []
    for k in range(num_ts):
        np.random.seed(0)
        level = 10 * np.random.rand()
        print(level / 10)
        np.random.seed(0)
        seas_amplitude = 0.1 + 0.3 * np.random.rand()
        print((seas_amplitude - 0.1) / 0.3)
        time_ticks = np.array(range(data_length))
        source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
        np.random.seed(0)
        noise = 0.1 * np.random.randn(data_length)
        print(noise / 0.1)
        data = source + noise
        index = pd.date_range(t0, periods=data_length, freq='D')
        series.append(pd.Series(data=data, index=index))
        series[k].plot()
        plt.title('seed_before_every_random')
        plt.show()

else:
    print('different_type must be one of \'no_seed\', \'seed_outside_loop\', \'seed_before_every_random\'')
