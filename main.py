import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense, Activation
from tensorflow.python.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() 

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i)for i, c in enumerate(characters))
index_to_char = dict((i, c)for i, c in enumerate(characters))

sentences = []
next_characters = []