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

def get_data(T):
    df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')
    df['prevClose'] = df['close'].shift(1)
    df['return'] = (df['close'] - df['prevClose']) / df['prevClose']
    print(df.head())

    # df['return'].hist()
    # plt.show()

    series = df['return'].values[1:].reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(series[:len(series) // 2])
    series = scaler.transform(series).flatten()

    # create data set
    X = []
    Y = []
    N = series.shape[0] - (T + 1) + 1
    for i in range(N):
        X.append(series[i:(i + T - 1)])
        Y.append(series[i + T])
    X = np.array(X, dtype='float32').reshape((N, T - 1, 1))
    Y = np.array(Y, dtype='float32')

    return X, Y, series

def predict_stock_classification(T):
    df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')
    input_data = df[['open', 'high', 'low', 'close']].values

    df['prevClose'] = df['close'].shift(1)
    df['return'] = (df['close'] - df['prevClose']) / df['prevClose']
    targets = df['return'].values

    D = input_data.shape[1]
    N = len(input_data) - T
    n_train = len(input_data) * 2 // 3

    scaler = StandardScaler()
    scaler.fit(input_data[:n_train])
    series = scaler.transform(input_data)

    X_train = np.zeros((n_train, T, D))
    Y_train = np.zeros(n_train)

    for t in range(n_train):
        X_train[t, :, :] = input_data[t:t + T]
        Y_train[t] = (targets[t + T] > 0)


    n_test = N - n_train
    X_test = np.zeros((n_test, T, D))
    Y_test = np.zeros(n_test)

    for t in range(n_test):
        input_idx = t + n_train
        X_test[t, :, :] = input_data[input_idx:input_idx + T]
        Y_test[t] = (targets[input_idx + T] > 0)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(T, D)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),batch_size=32, epochs=300)

def build_model(X, Y, T, series, prediction_no):
    # create model
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.SimpleRNN(5, input_shape=(T - 1,1)))
    model.add(tf.keras.layers.LSTM(5, input_shape=(T - 1, 1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='mse')
    h = model.fit(X, Y, epochs=50)

    # predict the future
    future_x = []
    future_y = []
    current_x = series[series.shape[0] - T + 1:, ]

    for x in range(prediction_no):
        pred = model.predict(current_x.reshape((1, -1, 1)))[0][0]
        future_y.append(pred)
        future_x.append(current_x)

        current_x = np.roll(current_x, -1)
        current_x[-1] = pred

    # plot
    N_1 = len(series)
    plt.plot(np.arange(N_1 // 2), series[:N_1 // 2], label='actual')
    plt.plot(np.arange(N_1 // 2 + 1, (N_1 // 2 + 1 + len(future_y))), series[N_1 // 2 + 1:N_1 // 2 + 1 + len(future_y)],
             label='actual-future')
    plt.plot(np.arange(N_1 // 2 + 1, (N_1 // 2 + 1 + len(future_y))), future_y, label='predicted')
    plt.legend()
    plt.show()

# T = 10
# x, y, series = get_data(T)
# build_model(x, y, T, series, 100)

predict_stock_classification(10)