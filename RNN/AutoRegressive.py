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


def sin_sequence_ann(add_noise=False, N_1 = 300, T = 51, prediction_no=300):
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
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(50, input_shape=(T - 1,), activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9),
                  loss='mse')
    h = model.fit(X, Y, epochs=100)

    # predict the future
    future_x = []
    future_y = None
    current_x = series[series.shape[0] - T + 1:, ]

    for x in range(prediction_no):
        if future_y is None:
            future_y = model.predict(current_x.reshape(1, -1))[0]
        else:
            pred = model.predict(current_x.reshape(1, -1))[0]
            future_y = np.concatenate((future_y, pred), axis=0)
        future_x.append(current_x)
        current_x = current_x[1:, ]
        current_x = np.hstack((current_x, future_y[future_y.shape[0] - 1,]))

    # plot
    plt.plot(np.arange(N_1), series, label='actual')
    plt.plot(np.arange(N_1 + 1, (N_1 + 1 + future_y.shape[0])), future_y, label='predicted')
    plt.legend()
    plt.show()
sin_sequence_ann(True, 200,10, 100 )