import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.datasets import load_breast_cancer
from tensorflow.python.client import device_lib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


MAX_VOCAB_SIZE = 20000


def preprocess_text(text, padding='post'):
    # tokeniz e
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    # padding
    sequences = pad_sequences(sequences, padding=padding)

    return sequences, len(tokenizer.word_index)


data = pd.read_csv("spam.csv")

target = data['v1'].map({'ham': 0, 'spam': 1}).values
input = data['v2'].values
input, V = preprocess_text(input)

X_train, X_test, Y_train, Y_test = train_test_split(input, target, test_size=0.33, random_state=42)

T = X_train.shape[1]
# embedding dimentionality
D = 20

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(V + 1, D, input_shape=(T,)))
model.add(tf.keras.layers.LSTM(15, return_sequences=True))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

h = model.fit(X_train,
              Y_train,
              validation_data=(X_test, Y_test),
              epochs=10)
