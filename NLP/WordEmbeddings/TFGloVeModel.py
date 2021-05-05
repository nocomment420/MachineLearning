import numpy as np
import tensorflow as tf
from WordEmbeddingModel import WordEmbeddingModel


class TFGloVeModel(WordEmbeddingModel):
    def __init__(self, V, D, from_file=True):
        super().__init__(V, D, savePath="TFGloveModel")
        self.logX = None
        self.Fx = None

        if from_file:
            try:
                self.load()
            except:
                print("could not load model")
                self.load_data()


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

    def train(self, epochs=10, lr_max=0.1, lr_min=0.01):
        Fx = self.Fx
        logX = self.logX
        V = self.V
        D = self.D

        # init weights
        W_init = np.random.randn(V, D) / np.sqrt(V)
        W = tf.Variable(W_init.astype(np.float32))

        U_init = np.random.randn(V, D) / np.sqrt(V)
        U = tf.Variable(U_init.astype(np.float32))

        B_init = np.zeros(V).reshape(V, 1)
        B = tf.Variable(B_init.astype(np.float32))

        C_init = np.zeros(V).reshape(1, V)
        C = tf.Variable(C_init.astype(np.float32))

        mu = tf.convert_to_tensor(logX.mean().astype(np.float32))
        logX = tf.convert_to_tensor(logX.astype(np.float32))
        Fx = tf.convert_to_tensor(Fx.astype(np.float32))
        lr_delta = (lr_max - lr_min) / epochs
        losses = []
        for e in range(epochs):
            lr = lr_max - (epochs * lr_delta)
            def loss():
                difference = tf.matmul(W, tf.transpose(U)) + B + C + mu - logX
                return tf.reduce_sum(difference * difference * Fx)

            optim = tf.keras.optimizers.Adam(lr=lr)
            a = optim.minimize(loss, [W, U, B, C])

            loss_summ = loss().numpy()

            print("Epoch {}: Loss: {}".format(e, loss_summ))

        self.We = W.numpy().dot(U.numpy().T) + B.numpy().reshape(V, 1) + C.numpy().reshape(1, V) + mu.numpy()

        self.save()

    def load_data(self, alpha=0.75, x_max=100, context_size=5, file_count=2):

        sentences, word2indx = self.get_wiki_data([], file_count=file_count)
        self.V = len(word2indx)
        V = self.V

        print("building matricies")

        X = np.zeros((V, V))

        for sentence in sentences:
            n = len(sentence)
            for start in range(n - 1):
                i = sentence[start]
                for end in range(start + 1, min(start + 1 + context_size, n)):
                    j = sentence[end]
                    X[i, j] += 1 / abs(start - end)
                for end in range(start - 1, max(start - 1 - context_size, 0), -1):
                    j = sentence[end]
                    X[i, j] += 1 / abs(start - end)

        Fx = X
        Fx[Fx > x_max] = 1
        Fx[Fx < x_max] = (Fx[Fx < x_max] / x_max) ** alpha

        logX = np.log(X + 1)

        self.Fx = Fx
        self.logX = logX

        self.save()

if __name__ == '__main__':
    model = TFGloVeModel(10000,300)
    model.load_data( context_size=10, file_count=31)
    model.train(epochs=200)
    model.generate_analogies()