import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from tensorflow.python.client import device_lib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def linear_classification():
    data = load_breast_cancer()
    # print(data.keys())
    # print(data.data.shape)
    # print(data.target)
    # print(data.target_names)
    # print(data.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    N, D = x_train.shape

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(D,), activation="sigmoid")
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    print("Train score: ", model.evaluate(x_train, y_train))
    print("Test score: ", model.evaluate(x_test, y_test))

    plt.plot(h.history['loss'], label='loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


def linear_regression():
    data = pd.read_csv("moore.csv", header=None).values
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1]
    Y = np.log(Y)
    X = X - X.mean()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9),
                  loss='mse')

    h = model.fit(X, Y, validation_split=0.33, epochs=100)

    # plt.plot(h.history['loss'], label='loss')
    # plt.plot(h.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.show()

    slope = model.layers[0].get_weights()[0][0, 0]
    intercept = model.layers[0].get_weights()[1][0]

    print(np.exp(slope), np.exp(intercept))

    plt.scatter(X, Y)

    x = np.linspace(X[0], X[-1], 100)
    y = (slope * x) + intercept
    plt.plot(x, y)
    plt.show()


linear_regression()
