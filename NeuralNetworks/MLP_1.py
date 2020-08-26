import numpy as np
import tensorflow as tf
from NeuralNetworks.DataGenerator import get_minst_data
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split

"""
    Multi layered perceptron using tensorflow 2.0 
"""
class MLP:
    def __init__(self, nodes, dropouts=None):
        # no dropouts -> p(keep) = 1 for all
        if dropouts is None:
            dropouts = [1 for _ in range(len(nodes) - 1)]
        assert len(nodes) == len(dropouts) + 1

        self.nodes = nodes
        self.dropouts = dropouts

        # initialise weights and biases
        self.W = []
        self.b = []
        for i in range(len(nodes) - 1):
            node_in = nodes[i]
            node_out = nodes[i + 1]

            weight_init = np.random.randn(node_in, node_out) / np.sqrt(node_in)
            W = tf.Variable(weight_init.astype(np.float32))
            self.W.append(W)

            bias_init = np.zeros(node_out)
            b = tf.Variable(bias_init.astype(np.float32))
            self.b.append(b)

        # initialise momentum
        self.VW = [0 for _ in self.W]
        self.Vb = [0 for _ in self.b]

    def loss(self, predicted, actual):
        target = tf.convert_to_tensor(actual)
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        return loss_func(y_true=target, y_pred=predicted)

    def fit(self, X, Y, batch_size=500, lr=0.01, epochs=100, mu=0.9, verbose=False, graph_out=False, validation_split=None):
        N, D = X.shape
        n_batches = N // batch_size

        # split test and train set
        if validation_split is not None:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_split, random_state=42)
        else:
            X_train, X_test, Y_train, Y_test = X, 0, Y, 0

        # initialise tracking variables
        count = 0
        loss_history = []
        t0 = dt.datetime.now()

        # training loop
        for i in range(epochs):
            for j in range(n_batches):
                # batch
                Xbatch = X_train[j * batch_size:(j * batch_size + batch_size), ]
                Ybatch = Y_train[j * batch_size:(j * batch_size + batch_size), ]

                self.train_function(Xbatch, Ybatch, lr, mu)

                # validation
                if count % 50 == 0 and validation_split is not None:
                    l, e = self.validate_loss_err(X_test, Y_test)
                    loss_history.append(l)
                    if verbose:
                        print("loss/err at iteration i=%d, j=%d: %.3f/%.5s" % (i, j, l, e))

                count += 1

        # communicate results
        if verbose:
            print("Completed in {}".format(dt.datetime.now() - t0))

        if graph_out:
            plt.plot(loss_history)
            plt.show()

    def feed_forward(self, X):
        # feed-forward
        Z = X
        for i in range(len(self.W) - 1):
            W, b = self.W[i], self.b[i]
            Z = tf.multiply(Z, self.dropouts[i])
            Z = tf.nn.relu(tf.matmul(Z, W) + b)

        Z = tf.nn.softmax(tf.matmul(Z, self.W[-1]) + self.b[-1])

        return Z

    def validate_loss_err(self, X, Y):
        predicted = self.feed_forward(X)
        m = tf.keras.metrics.CategoricalAccuracy()
        _ = m.update_state(Y, predicted)

        e = m.result().numpy()
        l = self.loss(predicted, Y)

        return l, e

    def train_function(self, X, Y, lr, mu):
        with tf.GradientTape() as g:
            # instruct tf to watch weights and balances for gradients
            for W, b in zip(self.W, self.b):
                g.watch(W)
                g.watch(b)

            # feed-forward
            Z = X
            for i in range(len(self.W) - 1):
                W, b = self.W[i], self.b[i]
                p_keep = self.dropouts[i]
                Z = tf.nn.dropout(Z, rate=1-p_keep)
                Z = tf.nn.relu(tf.matmul(Z, W) + b)

            Z = tf.nn.softmax(tf.matmul(Z, self.W[-1]) + self.b[-1])

            # calculate loss and gradients
            loss_val = self.loss(Z, Y)
            grads = g.gradient(loss_val, self.W + self.b)
            W_grads = grads[0:len(self.W)]
            b_grads = grads[len(self.W):]

            assert len(self.W) == len(self.b) == len(W_grads) == len(b_grads)

            # gradient descent
            for i in range(len(self.W)):
                W_g = W_grads[i]
                self.VW[i] = (mu * self.VW[i]) + (lr * W_g)
                self.W[i].assign_sub(self.VW[i])

                b_g = b_grads[i]
                self.Vb[i] = (mu * self.Vb[i]) + (lr * b_g)
                self.b[i].assign_sub(self.Vb[i])


    def predict(self, X):
        Z = self.feed_forward(X)
        return tf.argmax(Z, 1).numpy()

    def score(self, X, Y):
        l, e = self.validate_loss_err(X, Y)
        return e




if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest = get_minst_data(True)
    m = MLP([300, 100, 10],[0.8, 0.6])
    m.fit(Xtrain, Ytrain, epochs=100, verbose=True, lr=0.1)
    print(m.score(Xtest, Ytest))
