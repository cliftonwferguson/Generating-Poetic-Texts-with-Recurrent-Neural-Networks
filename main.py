import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Activation
from tensorflow.python.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() 

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i)for i, c in enumerate(characters))
index_to_char = dict((i, c)for i, c in enumerate(characters))

SEQ_Length = 40
Step_Size = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_Length, Step_Size):
    sentences.append(text[i: i+SEQ_Length])
    next_characters.append(text[i: i+SEQ_Length])

    x = np.zeros((len(sentences), SEQ_Length, len(characters)), dtype=np.bool)
    y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, character in enumerate(sentences):
            x[i, t, char_to_index[character]] = 1
        y[i, char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_Length, len(next_characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model.fit(x,y, batch_size=256, epochs=4)

model.save('textgenerator.model')