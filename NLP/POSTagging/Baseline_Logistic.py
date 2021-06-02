import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def get_data(filename, test_filename):
    X = []
    Y = []

    word2indx = {}
    indx2word = []
    i = 0
    labels = []
    label2idx = {}
    j = 0
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line != "\n":
                x_y = line.split()
                if x_y[0] not in [",", ".", "/", "!", ":", "'", "\"", " ", "?", ">", "<", ";", "[", "]", "{", "}", "(",
                                  ")", "$", "#", "`", "\\", "'"]:

                    Y.append(x_y[1])
                    if x_y[1] not in labels:
                        labels.append(x_y[1])
                        label2idx[x_y[1]] = j
                        j += 1

                    word = x_y[0]
                    if word not in word2indx:
                        word2indx[word] = i
                        indx2word.append(word)
                        i += 1

                    X.append(word2indx[word])
    L = []
    l_count = len(labels)
    for y in Y:
        l = np.zeros(l_count)
        l[label2idx[y]] = 1
        L.append(l)

    X_t = []
    Y_t = []
    with open(test_filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line != "\n":
                x_y = line.split()
                if x_y[0] in word2indx and x_y[1] in labels:
                    X_t.append(word2indx[x_y[0]])
                    t = np.zeros(l_count)
                    t[label2idx[x_y[1]]] = 1
                    Y_t.append(t)

    Y = np.array(L)
    X = np.array(X)
    X_t = np.array(X_t)
    Y_t = np.array(Y_t)
    print("found {} words".format(len(word2indx)))
    print("X dimentions : {}".format(X.shape))
    print("Y dimentions : {}".format(Y.shape))

    print("X test dimentions : {}".format(X_t.shape))
    print("Y test dimentions : {}".format(Y_t.shape))
    return X, Y, X_t, Y_t, word2indx


def train(X, Y, V, D, epochs=200, lr=0.1):
    # init weights
    W1_init = np.random.randn(V, D) / np.sqrt(V)
    W = tf.Variable(W1_init.astype(np.float32))

    for e in range(epochs):
        def loss():
            vectors = tf.nn.embedding_lookup(W, X)
            return tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=vectors)

        optim = tf.keras.optimizers.Adam(lr=lr)
        optim.minimize(loss, [W])

        loss = loss().numpy().sum()
        print("Epoch {} : {}".format(e + 1, loss))

    return W.numpy()


def test(X, Y, W):
    predictions = softmax(W[X])
    chosen = np.argmax(predictions, axis=1)
    targets = np.argmax(Y, axis=1)
    correct = targets[targets == chosen]
    accuracy = correct.shape[0] / Y.shape[0]
    print("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    X, Y, X_t, Y_t, word2indx = get_data("train.txt", "test.txt")
    V = len(word2indx)
    D = 41


    # W = train(X, Y, V, D, epochs=10)
    # test(X_t, Y_t, W)
