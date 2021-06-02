import numpy as np
import tensorflow as tf
from TreeNeuralNetworks.RecursiveTree import get_train_test_trees

def train_model(roots, V, D, K, epochs=10, lr=0.001):
    # init weights
    We_init = np.random.randn(V, D) / np.sqrt(V)
    We = tf.Variable(We_init.astype(np.float32))

    Wl_init = np.random.randn(D, D) / np.sqrt(D)
    Wl = tf.Variable(Wl_init.astype(np.float32))

    Wr_init = np.random.randn(D, D) / np.sqrt(D)
    Wr = tf.Variable(Wr_init.astype(np.float32))

    b_init = np.zeros(D)
    b = tf.Variable(b_init.astype(np.float32))

    Wo_init = np.random.randn(D, K) / np.sqrt(D)
    Wo = tf.Variable(Wo_init.astype(np.float32))

    bo_init = np.zeros(K)
    bo = tf.Variable(bo_init.astype(np.float32))

    optim = tf.keras.optimizers.Adam(lr=lr)

    print("Start training for {} Epochs".format(epochs))
    for e in range(epochs):
        for (i, root) in enumerate(roots):
            def loss():
                targets = []
                logits = []
                root.forwards(We, Wl, Wr, b, Wo, bo, logits, targets)

                targets = np.array(targets)
                logits = tf.reshape(logits, (targets.shape[0], 5))
                return tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=targets)
                )

            optim.minimize(loss, [We, Wl, Wr, b, Wo, bo])
            if i % 500 == 0:
                print("Epoch {} | Tree {}/{} : {}".format(e + 1, i, len(roots), loss().numpy()))


if __name__ == "__main__":
    train_roots, test_roots, word2idx = get_train_test_trees()
    V = len(word2idx)
    D = 50
    train_model(train_roots, V, D, 5)
