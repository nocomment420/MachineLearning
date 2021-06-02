import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch


def one_hot_encode_y(label2idx, Y, T=None):
    targets = []
    label_count = len(label2idx) + 1

    for sentence in Y:
        one_hot_sentence = []
        for y in sentence:
            one_hot = np.zeros(label_count)
            one_hot[label2idx[y]] = 1
            one_hot_sentence.append(one_hot)
        targets.append(one_hot_sentence)

    return pad_sequences(targets, maxlen=T)


def get_pos_data(filename="train.txt", test_filename="test.txt"):
    punctuation = [",", "/", "!", ":", "'", "\"", " ", "?", ">", "<", ";", "[", "]", "{", "}", "(",
                   ")", "$", "#", "`", "\\", "'", "--"]

    word2indx = {}
    word_idx = 1

    label2idx = {}
    label_idx = 1

    # Train set
    print("Getting Train set...")

    X = []
    Y = []
    curr_x = []
    curr_y = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:

            # End of line
            if line == "\n":
                X.append(curr_x)
                Y.append(curr_y)
                curr_x = []
                curr_y = []

            # same line
            else:
                x, y, _ = line.split()
                if x not in punctuation and y not in punctuation and _ != "O":

                    if y not in label2idx:
                        label2idx[y] = label_idx
                        label_idx += 1
                    curr_y.append(label2idx[y])

                    if x not in word2indx:
                        word2indx[x] = word_idx
                        word_idx += 1
                    curr_x.append(word2indx[x])

    X = pad_sequences(X)
    Y = pad_sequences(Y)

    assert (X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1])

    print("Sucessfully loaded Train set!\n")
    print("found {} words".format(len(word2indx)))
    print("found {} labels".format(len(label2idx)))
    print("X dimentions : {}".format(X.shape))
    print("Y dimentions : {}".format(Y.shape))
    print("\n")

    # Test set
    print("Getting Test set...")

    X_t = []
    Y_t = []
    curr_x = []
    curr_y = []

    with open(test_filename, "r") as file:
        lines = file.readlines()
        for line in lines:

            # End of line
            if line == "\n":
                X_t.append(curr_x)
                Y_t.append(curr_y)
                curr_x = []
                curr_y = []

            # Same line
            else:
                x, y, _ = line.split()

                if x in word2indx and y in label2idx:
                    curr_x.append(word2indx[x])
                    curr_y.append(label2idx[y])

    X_t = pad_sequences(X_t)
    Y_t = pad_sequences(Y_t)

    assert (X_t.shape[0] == Y_t.shape[0] and X_t.shape[1] == Y_t.shape[1])

    print("Sucessfully loaded Test set!\n")
    print("X test dimentions : {}".format(X_t.shape))
    print("Y test dimentions : {}".format(Y_t.shape))
    print("\n")

    return X, Y, X_t, Y_t, word2indx, label2idx


def get_ner_data(filename="ner.txt", test_size=0.3):
    word2indx = {}
    word_idx = 0

    label2idx = {}
    label_idx = 0

    # Train set
    print("Getting Train set...")

    X = []
    Y = []
    curr_x = []
    curr_y = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            # End of line
            if line == "\t\n":
                X.append(curr_x)
                Y.append(curr_y)
                curr_x = []
                curr_y = []

            # same line
            else:
                x, y = line.strip("\n").split(sep="\t")

                if y not in label2idx:
                    label2idx[y] = label_idx
                    label_idx += 1
                curr_y.append(label2idx[y])

                if x not in word2indx:
                    word2indx[x] = word_idx
                    word_idx += 1
                curr_x.append(word2indx[x])

    X = pad_sequences(X)
    Y = pad_sequences(Y)

    assert (X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1])

    n_test = int(test_size * X.shape[0])

    X_t = X[:n_test]
    Y_t = Y[:n_test]
    X = X[n_test:]
    Y = Y[n_test:]

    assert (X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1])
    assert (X_t.shape[0] == Y_t.shape[0] and X_t.shape[1] == Y_t.shape[1])

    print("Sucessfully loaded data set!\n")
    print("found {} words".format(len(word2indx)))
    print("found {} labels".format(len(label2idx)))
    print("X dimentions : {}".format(X.shape))
    print("Y dimentions : {}".format(Y.shape))
    print("X test dimentions : {}".format(X_t.shape))
    print("Y test dimentions : {}".format(Y_t.shape))
    print("\n")

    return X, Y, X_t, Y_t, word2indx, label2idx


def train(X, Y, V, D, K, epochs=10, lr=0.01, batch_size=512, X_t=None, Y_t=None):
    # Dimentions
    T = Y.shape[1]
    N = Y.shape[0]
    # K = Y.shape[2]

    # init weights
    We_init = np.random.randn(V, D) / np.sqrt(V)
    We = tf.Variable(We_init.astype(np.float32))

    Wo_init = np.random.randn(D, K) / np.sqrt(D)
    Wo = tf.Variable(Wo_init.astype(np.float32))

    bo_init = np.zeros(K)
    bo = tf.Variable(bo_init.astype(np.float32))

    rnn = tf.keras.layers.LSTM(D, return_sequences=True)
    print("Starting training for {} epochs".format(epochs))

    optim = tf.keras.optimizers.Adam(lr=lr)
    n_batches = N // batch_size

    for e in range(epochs):

        for batch in range(n_batches):
            x_batch = X[batch * batch_size: (batch + 1) * batch_size]
            y_batch = Y[batch * batch_size: (batch + 1) * batch_size]

            def loss():
                vector = tf.nn.embedding_lookup(We, x_batch)  # N x T x D

                h = rnn(vector)  # T x N x D
                h = tf.reshape(h, (T * batch_size, D))  # NT x D

                Z = tf.matmul(h, Wo) + bo  # NT x K

                flat_labels = tf.reshape(y_batch, [-1])  # NT

                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_labels, logits=Z))

            optim.minimize(loss, [We, Wo, bo] + rnn.trainable_weights)

            if batch == n_batches - 1:
                print("Epoch {} : {}".format(e + 1, loss().numpy()))

                # accuracy
                if X_t is not None and Y_t is not None:
                    test(We, Wo, bo, rnn, D, K, X_t, Y_t)

    return We, Wo, bo, rnn


def test(We, Wo, bo, rnn, D, K, X_t, Y_t):
    # Dimentions
    T = Y_t.shape[1]
    N = Y_t.shape[0]

    vector = tf.nn.embedding_lookup(We, X_t)  # N x T x D

    h = rnn(vector)  # T x N x D
    h = tf.reshape(h, (T * N, D))  # NT x D

    Z = tf.nn.softmax(tf.matmul(h, Wo) + bo)  # NT x K

    prediction = tf.argmax(Z, axis=1).numpy()
    labels = Y_t.reshape((T * N))

    correct = labels[labels == prediction]
    accuracy = correct.shape[0] / labels.shape[0]

    print("Accuracy: {}%".format(round(accuracy * 100, 2)))


def predict(scentence, We, Wo, bo, rnn, D, word2indx, label2idx):
    X = [word2indx[word] for word in scentence.split()]
    X = np.array([X])
    vector = tf.nn.embedding_lookup(We, X)  # N x T x D
    T = X.shape[1]

    h = rnn(vector)  # T x N x D
    h = tf.reshape(h, (T, D))  # NT x D

    Z = tf.nn.softmax(tf.matmul(h, Wo) + bo)  # NT x K

    prediction = tf.argmax(Z, axis=1).numpy()

    idx2label = list(label2idx.keys())
    preds = [idx2label[label] for label in prediction]

    for (word, pos) in zip(scentence.split(), preds):
        print("{} : {}".format(word, pos))


def save_model(We, Wo, bo, rnn, name):
    tf.keras.models.save_model(rnn, )


def load_model(name):
    pass


def run_pos():
    X, Y, X_t, Y_t, word2indx, label2idx = get_pos_data()
    V = len(word2indx) + 1
    K = len(label2idx) + 1
    D = 50
    We, Wo, bo, rnn = train(X, Y, V, D, K, lr=0.1, epochs=10)
    test(We, Wo, bo, rnn, D, K, X_t, Y_t)
    predict("The louder the rant of the traffic offenders the more acute are the wardens feelings of pleasure that they the stakeless the outcasts the niggers are a valued part of the empire of law and in a position to chastise the arrogance and selfishness of the indigenous people", We, Wo, bo, rnn, D, word2indx, label2idx)


def run_ner():
    X, Y, X_t, Y_t, word2indx, label2idx = get_ner_data()
    V = len(word2indx)
    K = len(label2idx) + 1
    D = 50
    We, Wo, bo, rnn = train(X, Y, V, D, K, lr=0.1, epochs=20)
    # test(We, Wo, bo, rnn, D, K, X_t, Y_t)


if __name__ == "__main__":
    # run_ner()
    run_pos()
