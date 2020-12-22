import tensorflow as tf
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.99:
            print('\naccuracy가 99%에 도달하여 학습을 종료합니다!')
            self.model.stop_training = True


MNIST = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = MNIST.load_data()
x_train, x_test = x_train/255, x_test/255

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=100, callbacks=[callbacks])