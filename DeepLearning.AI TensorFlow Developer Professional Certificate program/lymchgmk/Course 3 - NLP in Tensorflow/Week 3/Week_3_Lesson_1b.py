from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds


dataset, info = tfds.load('', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = train_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),
        tf.keras.Dense(64, activation='relu'),
        tf.keras.Dense(1, activation='sigmoid')
    ]
)

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metics=['accuracy']
)

NUM_EPOCHS = 10
history = model.fit(
    x=train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS
)
