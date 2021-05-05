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


def rnn_image_classification():
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
    model.add(tf.keras.layers.LSTM(5,input_shape=(D1, D2)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

def rnn_max_pooling_image_classification():
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
    model.add(tf.keras.layers.LSTM(5, input_shape=(D1, D2), return_sequences=True))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    plt.plot(h.history['accuracy'], label='acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


# rnn_image_classification()
rnn_max_pooling_image_classification()
