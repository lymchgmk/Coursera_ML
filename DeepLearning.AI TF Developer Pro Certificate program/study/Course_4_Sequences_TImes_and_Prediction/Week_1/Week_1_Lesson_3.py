import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_series(time, series, format='-', start=0, end=None):
    pass


def trend(time, slope=0):
    return slope*time


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.4,
        np.cos(season_time*2*np.pi),
        1/np.exp(3*season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time+phase)%period)/period
    return amplitude*seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arrange(4*365+1, dytpe='float32')
baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

naive_forecast = series[split_time-1: -1]

