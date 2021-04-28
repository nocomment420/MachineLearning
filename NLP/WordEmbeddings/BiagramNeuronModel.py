import numpy as np
import tensorflow as tf
from WordEmbedding import WordEmbeddingModel
import random


class BiagramNeronModel(WordEmbeddingModel.WordEmbeddingModel):
    def __init__(self, V, D):
        super().__init__(V, D, savePath="BiagramNeuronModel")

    def softmax(self, a):
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    def save(self):
        np.save("{}/logX.npy".format(self.save_path), self.logX)
        np.save("{}/Fx.npy".format(self.save_path), self.Fx)
        super().save()

    def load(self):
        try:
            self.Fx = np.load("{}/Fx.npy".format(self.save_path))
            self.logX = np.load("{}/logX.npy".format(self.save_path))
            super().load()
        except:
            print("error loading model")

    def loss(self, predicted, target):
        return -np.sum(np.log(predicted[np.arange(predicted.shape[0]), target])) / predicted.shape[0]

    def train_numpy(self, epochs=1, file_count=2, lr=0.0001):
        sentences, word2indx = self.get_wiki_data([], file_count=file_count)
        self.V = len(word2indx)

        V = self.V
        D = self.D

        W1 = np.random.randn(V, D) / np.sqrt(V)
        W2 = np.random.randn(D, V) / np.sqrt(D)

        losses = []
        for e in range(epochs):
            random.shuffle(sentences)
            j = 0
            for sentence in sentences:
                sentence = [0] + sentence + [1]
                n = len(sentence)
                x = sentence[:n - 1]
                y = sentence[1:]

                h = np.tanh(W1[x])
                prediction = self.softmax(h.dot(W2))

                step_loss = self.loss(prediction, y)
                losses.append(step_loss)

                dout = prediction
                dout[np.arange(n - 1), y] -= 1
                g2 = h.T.dot(dout)
                W2 = W2 - lr * g2

                g1 = (dout).dot(W2.T) * (1 - h ** 2)

                np.subtract.at(W1, x, lr * g1)

                if j % 100 == 0:
                    print("epoch {}({}) - {}".format(e, j, step_loss))

                j += 1

        self.We = (W1 + W2.T) / 2

        self.save()

    def train_tf(self, epochs=10, file_count=2, lr=0.001):
        sentences, word2indx = self.get_wiki_data([], file_count=file_count)
        self.V = len(word2indx)

        V = self.V
        D = self.D

        W1_init = np.random.randn(V, D) / np.sqrt(V)
        W1 = tf.Variable(W1_init.astype(np.float32))

        W2_init = np.random.randn(D, V) / np.sqrt(D)
        W2 = tf.Variable(W2_init.astype(np.float32))

        losses = []
        optim = tf.keras.optimizers.SGD(lr=lr)
        for e in range(epochs):
            random.shuffle(sentences)
            j = 0
            for sentence in sentences:
                sentence = [0] + sentence + [1]
                n = len(sentence)
                x = sentence[:n - 1]
                y = sentence[1:]

                def loss():
                    h = tf.nn.tanh(tf.nn.embedding_lookup(W1, x))
                    prediction = tf.nn.softmax(tf.matmul(h, W2))
                    out = tf.nn.embedding_lookup(tf.transpose(prediction), y)
                    return -tf.reduce_sum(tf.math.log(out)) / (n - 1)

                optim.minimize(loss, [W1, W2])

                if j % 1000 == 0:
                    step_loss = loss().numpy()
                    print("epoch {}({}) - {}".format(e, j, step_loss))

                j += 1

        self.We = (W1.numpy() + W2.numpy().T) / 2

        self.save()


if __name__ == '__main__':
    model = BiagramNeronModel(2000, 100)
    model.train_tf(epochs=1, lr=0.01)
    model.generate_analogies()
