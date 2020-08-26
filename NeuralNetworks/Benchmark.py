import numpy as np
from NeuralNetworks.DataGenerator import get_minst_data
from sklearn.utils import shuffle
import datetime
import matplotlib.pyplot as plt

# ----------- utility methods --------------
"""Feed forward for NN (uses softmax activation)"""
def forwards(x, W, b):
    u = x.dot(W) + b
    expu = np.exp(u)

    return expu / expu.sum(axis=1, keepdims=True)


"""Returns the predicted class from 1-hot encoded y"""
def predict(cat_y):
    return np.argmax(cat_y, axis=1)


"""Averages the error values to have a smoother curve"""
def smooth_history_curve(history, step_size=10):
    l = len(history) // step_size
    smoothed_history = []
    for i in range(l):
        total = 0
        for j in range(step_size):
            total += history[(step_size * i) + j]
        mean = total / step_size
        smoothed_history.append(mean)
    return smoothed_history


"""Calculates the accuracy """
def error_rate(cat_y, t):
    prediction = predict(cat_y)
    actual = predict(t)

    incorrect = 0
    total = len(cat_y)
    for i in range(total):
        if prediction[i] != actual[i]:
            incorrect += 1

    return incorrect / total


"""Catagorical cross entropy loss"""
def cost(cat_y, t):
    tot = t * np.log(cat_y)
    return -tot.sum()


"""calculates gradient for weights"""
def gradW(t, y, X):
    return X.T.dot(t - y)


"""calculates gradient for bias"""
def gradb(t, y):
    return (t - y).sum(axis=0)


# Hyper parameters
LR = 0.0001
EPOCHS = 50
REG = 0.01


# ----- Optimisers ----------
def full_gradient_descent(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS):
    # init dimentions and weights
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # feed forwards
        prediction = forwards(x_train, W, b)

        # gradient descent
        W += LR * (gradW(y_train, prediction, x_train) - REG * W)
        b += LR * (gradb(y_train, prediction) - REG * b)

        cost_history.append(cost(prediction, y_train) )

    # validation + print results
    prediction = forwards(x_test, W, b)
    print("Full Gradient Descent")
    print("Final error rate:", error_rate(prediction, y_test))
    print("Time taken: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def stochastic_gradient_descent(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS):
    # init dimentions and weights
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)

    # training loop
    cost_history = []
    t0 = datetime.datetime.now()
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)
        for n in range(min(N, 500)):
            # Randomly chose an observation
            x = x_temp[n, :].reshape(1, D)
            y = y_temp[n, :].reshape(1, 10)

            # feed forward
            predictions = forwards(x, W, b)

            # back propagation
            W += LR * (gradW(y, predictions, x) - REG * W)
            b += LR * (gradb(y, predictions) - REG * b)

            cost_history.append(cost(predictions, y) )

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("Stochastic Gradient Descent")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for stochastic GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def batch_gradient_descent(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS):
    # init dimentions and weights
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    batch_sz = 500
    n_batches = N // batch_sz

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)

        for n in range(n_batches):
            # Select batch
            x = x_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, D)
            y = y_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, 10)

            # feed forwards
            pred = forwards(x, W, b)

            # gradient descent
            W += LR * (gradW(y, pred, x) - REG * W)
            b += LR * (gradb(y, pred) - REG * b)

            cost_history.append(cost(pred, y) )

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("Batch Gradient Descent")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for batch GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def batch_gradient_descent_with_momentum(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS, mu=0.90):
    # init dimentions and weights and momentums
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    Vw = 0
    b = np.zeros(10)
    Vb = 0

    batch_sz = 500
    n_batches = N // batch_sz

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)

        for n in range(n_batches):
            # select batch
            x = x_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, D)
            y = y_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, 10)

            # feed forwards
            pred = forwards(x, W, b)

            # gradient descent using momentum
            Vw = (mu * Vw) + LR * (gradW(y, pred, x) - REG * W)
            W += Vw

            Vb = (mu * Vb) + LR * (gradb(y, pred) - REG * b)
            b += Vb

            cost_history.append(cost(pred, y))

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("Batch with momentum")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for batch GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def batch_gradient_descent_with_nesterov_momentum(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS, mu=0.90):
    # init dimentions and weights and momentums
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    Vw = 0
    b = np.zeros(10)
    Vb = 0

    batch_sz = 500
    n_batches = N // batch_sz

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)

        for n in range(n_batches):
            # select batch
            x = x_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, D)
            y = y_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, 10)

            # feed forwwards
            pred = forwards(x, W, b)

            # gradient descent using nesterov momentum
            Vw = (mu * Vw) + LR * (gradW(y, pred, x) - REG * W)
            W += (mu * Vw) + LR * (gradW(y, pred, x) - REG * W)

            Vb = (mu * Vb) + LR * (gradb(y, pred) - REG * b)
            b += (mu * Vb) + LR * (gradb(y, pred) - REG * b)

            cost_history.append(cost(pred, y))

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("Batch with nestrov monmentum")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for batch GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def batch_gradient_descent_RMS(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS, mu=0.90):
    # init dimentions and weights and momentums
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    cacheW = 1
    b = np.zeros(10)
    cacheB = 1

    # init hyper params
    batch_sz = 500
    n_batches = N // batch_sz
    decay_rate = 0.999
    epsilon = 0.000000001
    lr0 = 0.01

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)

        for n in range(n_batches):
            # select batch
            x = x_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, D)
            y = y_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, 10)

            # feed forwards
            pred = forwards(x, W, b)

            # gradient descent using RMS algorithm
            gW = gradW(y, pred, x)
            cacheW = decay_rate * cacheW + (1 - decay_rate) * gW * gW
            ada = lr0 / np.sqrt(cacheW + epsilon)
            W += ((gW - REG * W) * ada)

            gb = gradb(y, pred)
            cacheB = decay_rate * cacheB + (1 - decay_rate) * gb * gb
            ada = lr0 / np.sqrt(cacheB + epsilon)
            b += ((gb - REG * b) * ada)

            cost_history.append(cost(pred, y))

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("RMS")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for batch GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)

def batch_gradient_descent_ADAM(x_train, x_test, y_train, y_test, EPOCHS=EPOCHS, mu=0.9):
    # init dimentions and weights and momentums
    N, D = x_train.shape
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    mW = 0
    vW = 0
    mb = 0
    vb = 0

    # init hyperparams
    t = 1
    batch_sz = 500
    n_batches = N // batch_sz
    epsilon = 0.000000001
    lr0 = 0.01

    # training loop
    t0 = datetime.datetime.now()
    cost_history = []
    for i in range(EPOCHS):
        # shuffle data
        x_temp, y_temp = shuffle(x_train, y_train)

        for n in range(n_batches):
            # select batch
            x = x_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, D)
            y = y_temp[n * batch_sz:(n * batch_sz + batch_sz), :].reshape(batch_sz, 10)

            # feed forwards
            pred = forwards(x, W, b)

            # gradient descent using ADAM algorithm
            gW = -gradW(y, pred, x) - REG * W
            mW = (0.9 * mW) + (1 - 0.9) * gW
            vW = (0.999 * vW) + (1 - 0.999) * gW * gW
            mW_ = mW / (1 - (0.9 ** t))
            vW_ = vW / (1 - (0.999 ** t))
            W += -lr0 * mW_ / np.sqrt(vW_ + epsilon)

            gb = -gradb(y, pred) - REG * b
            mb = (0.9 * mb) + (1 - 0.9) * gb
            vb = (0.999 * vb) + (1 - 0.999) * gb * gb
            mb_ = mb / (1 - (0.9 ** t))
            vb_ = vb / (1 - (0.999 ** t))
            b += -lr0 * mb_ / np.sqrt(vb_ + epsilon)

            t += 1

            cost_history.append(cost(pred, y))

    # validation + print results
    p_y = forwards(x_test, W, b)
    print("Adam optimser")
    print("Final error rate:", error_rate(p_y, y_test))
    print("time for batch GD: {}".format(datetime.datetime.now() - t0))
    print("\n")

    return smooth_history_curve(cost_history)


def run():
    x_train, x_test, y_train, y_test = get_minst_data(True)
    h0 = full_gradient_descent(x_train, x_test, y_train, y_test, 30)
    h1 = stochastic_gradient_descent(x_train, x_test, y_train, y_test, 30)
    h2 = batch_gradient_descent(x_train, x_test, y_train, y_test, 30)
    h3 = batch_gradient_descent_with_momentum(x_train, x_test, y_train, y_test, 30)
    h4 = batch_gradient_descent_with_nesterov_momentum(x_train, x_test, y_train, y_test, 30)
    h5 = batch_gradient_descent_RMS(x_train, x_test, y_train, y_test, 30)
    h6 = batch_gradient_descent_ADAM(x_train, x_test, y_train, y_test, 30)

    x0 = np.linspace(0, 1, len(h0))
    # plt.plot(x0, h0, label="full_gradient_descent")

    x1 = np.linspace(0, 1, len(h1))
    # plt.plot(x1, h1, label="stochastic_gradient_descent")

    x2 = np.linspace(0, 1, len(h2))
    plt.plot(x2, h2, label="batch_gradient_descent")

    x3 = np.linspace(0, 1, len(h3))
    plt.plot(x3, h3, label="batch_gradient_descent_with_momentum")

    x4 = np.linspace(0, 1, len(h4))
    plt.plot(x4, h4, label="batch_gradient_descent_with_nesterov_momentum")

    x5 = np.linspace(0, 1, len(h5))
    plt.plot(x5, h5, label="batch_gradient_descent_RMS")

    x6 = np.linspace(0, 1, len(h6))
    plt.plot(x6, h6, label="batch_gradient_descent_ADAM")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
