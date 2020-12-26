import numpy as np
import matplotlib.pyplot as plt


def plot_series():
    pass


def trend(time, slope=0):
    return time*slope


time = np.arange(4*365+1)
baseline = 10
series = trend(time, 0.1)
plot_series(time, series)


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.4,
        np.cos(
            season_time*2*np.pi
        ),
        1 / np.exp(3*season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time+phase)%period)%period
    return amplitude*seasonal_pattern(season_time)


