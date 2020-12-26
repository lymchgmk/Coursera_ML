import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


'''!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt \
    -O /tmp/sonnets.txt'''


tokenizer = Tokenizer()

data = open('', 'r')
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequences_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequences_len, padding='pre'))

predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequences_len-1),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(150, return_sequences=True)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(total_words/2,
                              activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)
                              ),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ]
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    predictors,
    label,
    epochs=100,
    verbose=1
)

seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequences_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = predicted
            break
    seed_text += ' ' + output_word
print(seed_text)