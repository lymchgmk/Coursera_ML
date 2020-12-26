import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

'''
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
    -O /tmp/irish-lyrics-eof.txt
'''

tokenizer = Tokenizer()

data = open('').read()
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sentences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sentences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sentences])
input_sentences = np.array(pad_sequences(input_sentences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sentences[:, :-1], input_sentences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_word)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(150)
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
    x=xs,
    y=ys,
    epochs=100,
    verbose=1
)

seed_text = "I've got a bad feeling about this"
next_word = 100

for _ in range(next_word):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(token_list, maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = predicted
            break
    seed_text += ' ' + output_word
print(seed_text)