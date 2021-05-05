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


def ann_image_classification():
    data = mnist.load_data()
    x_train, y_train, x_test, y_test = data[0][0], data[0][1], data[1][0], data[1][1]
    x_train = x_train / 255
    x_test = x_test / 255

    index = y_train.shape[0]
    Y = np.concatenate((y_train, y_test), axis=0)
    Y = to_categorical(Y)
    y_train = Y[0:index, ]
    y_test = Y[index:, ]

    N, D1, D2 = x_train.shape

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(D1, D2)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)


def ann_regression():
    N = 1000
    X = np.random.random((N, 2)) * 6 - 3
    Y = np.cos(2 * X[:, 0]) + np.cos(3 * X[:, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    # plt.show()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(140, input_shape=(2,), activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mse')

    h = model.fit(X, Y, epochs=300)

    line = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(line, line)
    x_grid = np.vstack((xx.flatten(), yy.flatten())).T
    y_hat = model.predict(x_grid).flatten()
    ax.plot_trisurf(x_grid[:, 0], x_grid[:, 1], y_hat, linewidth=0.2, antialiased=True)
    plt.show()


ann_regression()
