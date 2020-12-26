import tensorflow as tf
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            print('\naccuracy가 99.8%에 도달하여 학습을 종료합니다!')
            self.model.stop_training = True


callbacks = myCallback()
MNIST = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = MNIST.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(train_images, train_labels, epochs = 10, callbacks=[callbacks])