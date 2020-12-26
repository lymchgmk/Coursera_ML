import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series():
    pass


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
)

history = model.fit(
    dataset,
    epochs=100,
    verbose=0
)

forecast = []
for time in range(len(serires) - window_size):
    forecast.append(model.predict(series[time: time+window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()