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

def sin_sequence_rnn(add_noise=False, N_1 = 300, T = 51, prediction_no=300):
    # create sin sequence
    series = np.sin(0.1 * np.arange(N_1))
    if add_noise:
        series += np.random.randn(N_1)*0.1

    # create data set
    X = []
    Y = []
    N = series.shape[0] - (T + 1) + 1
    for i in range(N):
        X.append(series[i:(i + T - 1)])
        Y.append(series[i + T])
    X = np.array(X, dtype='float32').reshape((N, T-1, 1))
    Y = np.array(Y, dtype='float32')

    # create model
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.SimpleRNN(5, input_shape=(T - 1,1)))
    model.add(tf.keras.layers.LSTM(5, input_shape=(T - 1,1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='mse')
    h = model.fit(X, Y, epochs=50)

    # predict the future
    future_x = []
    future_y = []
    current_x = series[series.shape[0] - T + 1:, ]

    for x in range(prediction_no):
        pred = model.predict(current_x.reshape((1, -1,1)))[0][0]
        future_y.append(pred)
        future_x.append(current_x)

        current_x = np.roll(current_x, -1)
        current_x[-1] = pred

    # plot
    plt.plot(np.arange(N_1), series, label='actual')
    plt.plot(np.arange(N_1 + 1, (N_1 + 1 + len(future_y))), future_y, label='predicted')
    plt.legend()
    plt.show()


def complex_sin_sequence_rnn(add_noise=False, N_1 = 300, T = 51, prediction_no=300):
    # create sin sequence
    series = np.sin((0.1 * np.arange(N_1))**2)
    if add_noise:
        series += np.random.randn(N_1)*0.1

    # create data set
    X = []
    Y = []
    N = series.shape[0] - (T + 1) + 1
    for i in range(N):
        X.append(series[i:(i + T - 1)])
        Y.append(series[i + T])
    X = np.array(X, dtype='float32').reshape((N, T-1, 1))
    Y = np.array(Y, dtype='float32')

    # create model
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.SimpleRNN(5, input_shape=(T - 1,1)))
    model.add(tf.keras.layers.LSTM(20, input_shape=(T - 1,1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='mse')
    h = model.fit(X, Y, epochs=200)

    # predict the future
    future_x = []
    future_y = []
    current_x = series[series.shape[0] - T + 1:, ]

    for x in range(prediction_no):
        pred = model.predict(current_x.reshape((1, -1,1)))[0][0]
        future_y.append(pred)
        future_x.append(current_x)

        current_x = np.roll(current_x, -1)
        current_x[-1] = pred

    # plot
    plt.plot(np.arange(N_1), series, label='actual')
    plt.plot(np.arange(N_1 + 1, (N_1 + 1 + len(future_y))), np.sin((0.1 * np.arange(N_1+1,(N_1 + 1 + len(future_y)) ))**2), label='actual-future')
    plt.plot(np.arange(N_1 + 1, (N_1 + 1 + len(future_y))), future_y, label='predicted')
    plt.legend()
    plt.show()

complex_sin_sequence_rnn(True, 500, 100, 150)