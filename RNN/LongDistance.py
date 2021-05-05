import tensorflow as tf
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


def get_taget(X, i1, i2, i3):
    sum = 0

    if X[i1] < 0:
        sum += 1
    if X[i2] < 0:
        sum += 1
    if X[i3] < 0:
        sum += 1
    return sum % 2 != 0


def get_series(N, T, long_short):
    X = []
    Y = []
    for i in range(N):
        x = np.random.randn(T)
        X.append(x)

        if long_short == 'L':
            y = get_taget(x, 0, 1, 2)
        else:
            y = get_taget(x, -3, -2, -1)

        Y.append(y)

    return np.array(X).reshape((N, T, 1)), to_categorical(np.array(Y))


def get_model_rnn(T, X, Y):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(10, input_shape=(T, 1)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X, Y, epochs=30, validation_split=0.4)

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

def get_model_lstm(T, X, Y):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(10, input_shape=(T, 1)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X, Y, epochs=200, validation_split=0.4)

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

def get_model_lstm_max_pool(T, X, Y):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(10, input_shape=(T, 1), return_sequences=True))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X, Y, epochs=200, validation_split=0.4)

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

N = 3000
T = 20
long_short = 'L'
X, Y = get_series(N, T, long_short)

# long distance rnn vs lstm
get_model_rnn(T, X, Y)
get_model_lstm(T, X, Y)



N = 3000
T = 35
long_short = 'L'
X, Y = get_series(N, T, long_short)
# even longer distance lstm vs lstm max pooling
get_model_lstm(T, X, Y)
get_model_lstm_max_pool(T, X, Y)