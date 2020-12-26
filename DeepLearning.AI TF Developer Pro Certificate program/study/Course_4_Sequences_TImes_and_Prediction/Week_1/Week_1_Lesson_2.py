import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def trend(time, slope=0):
    return slope*time


time = np.arange(4*365+1)
baseline = 10
series = trend(time, 0.1)


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.4,
        np.cos(season_time*2*np.pi), 1 / np.exp(3*season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time+phase)%period)/period
    return amplitude * seasonal_pattern(season_time)


baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


noise_level = 5
noise = white_noise(time, noise_level, seed=42)


series += noise

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    pi_1 = 0.5
    pi_2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += pi_1 * ar[step-50]
        ar[step] += pi_2 * ar[step-33]
    return ar[50:]*amplitude

pass
