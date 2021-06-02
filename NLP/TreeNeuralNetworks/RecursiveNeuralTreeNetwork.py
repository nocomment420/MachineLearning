from TreeNeuralNetworks.RecursiveNeuralTree import get_train_test_sequences
import numpy as np
import torch
import datetime
from sklearn.utils import shuffle
import stanfordnlp

def forwards(left, right, words, weights, T, D, N):
    [We, All, Alr, Arr, Wl, Wr, b, Wo, bo] = weights

    # recursive steps
    hidden_init = np.zeros((N, T, D)).astype(np.float32)
    hidden = torch.from_numpy(hidden_init)  # N x T x D
    for i in range(T):
        hidden = recursive(i, hidden, left, right, words, [We, All, Alr, Arr, Wl, Wr, b])

    Z = torch.matmul(hidden, Wo) + bo  # N x T x K
    Z = torch.flatten(Z, start_dim=0, end_dim=1)  # NT x K
    return torch.log_softmax(Z, -1)


def recursive(i, hidden, left, right, words, weights):
    [We, All, Alr, Arr, Wl, Wr, b] = weights

    w = words[:, i]  # 1 x N

    # leaf node
    l_i = w > -1
    hidden[l_i, i] = We[w][l_i]

    # parent node
    p_i = w == -1
    if len(p_i.nonzero()) != 0:
        xl = hidden[p_i, left[p_i, i]]  # N x D
        xr = hidden[p_i, right[p_i, i]]  # N x D
        hidden[p_i, i] = torch.relu(torch.einsum("nd,ddd,nd->nd", xl, All, xl) +
                                    torch.einsum("nd,ddd,nd->nd", xl, Alr, xr) +
                                    torch.einsum("nd,ddd,nd->nd", xr, Arr, xr) +
                                    torch.einsum("nd,dd->nd", xr, Wr) +
                                    torch.einsum("nd,dd->nd", xl, Wl) +
                                    b).data

    return hidden


def train_model(left, right, words, labels, N, V, D, K, T, epochs=5, batch_size=512, lr_max=0.05, lr_min=0.005,
                test=None):
    # init weights
    We_init = np.random.randn(V, D) / np.sqrt(V)
    We = torch.tensor(We_init.astype(np.float32), requires_grad=True)

    A_l_l_init = np.random.randn(D, D, D) / np.sqrt(3 * D)
    All = torch.tensor(A_l_l_init.astype(np.float32), requires_grad=True)

    A_l_r_init = np.random.randn(D, D, D) / np.sqrt(3 * D)
    Alr = torch.tensor(A_l_r_init.astype(np.float32), requires_grad=True)

    A_r_r_init = np.random.randn(D, D, D) / np.sqrt(3 * D)
    Arr = torch.tensor(A_r_r_init.astype(np.float32), requires_grad=True)

    W_l_init = np.random.randn(D, D) / np.sqrt(D)
    Wl = torch.tensor(W_l_init.astype(np.float32), requires_grad=True)

    W_r_init = np.random.randn(D, D) / np.sqrt(D)
    Wr = torch.tensor(W_r_init.astype(np.float32), requires_grad=True)

    b_init = np.zeros(D)
    b = torch.tensor(b_init.astype(np.float32), requires_grad=True)

    W_o_init = np.random.randn(D, K) / np.sqrt(K)
    Wo = torch.tensor(W_o_init.astype(np.float32), requires_grad=True)

    bo_init = np.zeros(K)
    bo = torch.tensor(bo_init.astype(np.float32), requires_grad=True)

    # init variables
    left = left.astype(np.long)
    right = right.astype(np.long)
    words = words.astype(np.long)
    labels = labels.astype(np.long)

    n_batches = N // batch_size
    lr_delta = (lr_max - lr_min) / epochs

    weights = [We, All, Alr, Arr, Wl, Wr, b, Wo, bo]
    criterion = torch.nn.CrossEntropyLoss()

    # train loop
    print("Start training for {} epochs".format(epochs))
    for e in range(epochs):
        start = datetime.datetime.now()
        left, right, words, labels = shuffle(left, right, words, labels)

        lr = lr_max - (e * lr_delta)
        optim = torch.optim.Adam(weights, lr=lr, weight_decay=1e-4)

        for batch in range(n_batches):
            left_batch = torch.tensor(left[batch_size * batch:batch_size * (batch + 1)], dtype=torch.long)
            right_batch = torch.tensor(right[batch_size * batch:batch_size * (batch + 1)], dtype=torch.long)
            words_batch = torch.tensor(words[batch_size * batch:batch_size * (batch + 1)], dtype=torch.long)
            labels_batch = torch.tensor(labels[batch_size * batch:batch_size * (batch + 1)], dtype=torch.long)

            optim.zero_grad()
            prediction = forwards(left_batch, right_batch, words_batch, weights, T, D, batch_size)

            loss = criterion(prediction, torch.flatten(labels_batch))

            loss.backward()

            optim.step()

            if batch == n_batches - 1:
                time_taken = (datetime.datetime.now() - start).seconds

                # test accuracy
                if test is not None:
                    accuracy = test_model(test, weights, verbose=False)
                    print("Epoch {} - Loss: {} | Accuracy: {}% | Time: {} s".format(e + 1, loss.item(), accuracy,
                                                                                    time_taken))
                else:
                    print("Epoch {} - Loss: {} | Time: {} s".format(e + 1, loss.item(), time_taken))

    save_model(weights)


def save_model(weights):
    [We, All, Alr, Arr, Wl, Wr, b, Wo, bo] = weights
    torch.save(We, "trees/We.pt")
    torch.save(All, "trees/All.pt")
    torch.save(Alr, "trees/Alr.pt")
    torch.save(Arr, "trees/Arr.pt")
    torch.save(Wl, "trees/Wl.pt")
    torch.save(Wr, "trees/Wr.pt")
    torch.save(b, "trees/b.pt")
    torch.save(Wo, "trees/Wo.pt")
    torch.save(bo, "trees/bo.pt")


def load_model():
    return [torch.load("trees/We.pt"),
            torch.load("trees/All.pt"),
            torch.load("trees/Alr.pt"),
            torch.load("trees/Arr.pt"),
            torch.load("trees/Wl.pt"),
            torch.load("trees/Wr.pt"),
            torch.load("trees/b.pt"),
            torch.load("trees/Wo.pt"),
            torch.load("trees/bo.pt")]


def test_model(test, weights=None, verbose=True):
    lefts, rights, words, labels = test
    T = lefts.shape[1]
    N = lefts.shape[0]
    D = 50

    if weights is None:
        weights = load_model()

    with torch.no_grad():
        prediction = forwards(lefts, rights, words, weights, T, D, N)  # NT x K
        pred = torch.argmax(prediction, 1).detach().numpy()  # NT

        targets = labels.flatten()  # NT
        correct = targets[targets == pred]

        if verbose:
            print("Accuracy : {}%".format(round((correct.shape[0] * 100) / (N * T), 2)))
        else:
            return round((correct.shape[0] * 100) / (N * T), 2)

def test():
    stanfordnlp.download('en')

    sentence = "The argument is used to specify the task All five processors are taken by default if no argument is passed Here is a quick overview of the processors"
    # weights = load_model()

    nlp = stanfordnlp.Pipeline(processors="tokenize,depparse")
    f = nlp(sentence)
    m = 0


if __name__ == "__main__":
    # train, test, word2idx = get_train_test_sequences()
    #
    # left, right, words, labels = train
    # V = len(word2idx) + 1
    # N = left.shape[0]
    # T = left.shape[1]
    # K = labels.max() + 1
    # D = 50
    #
    # train_model(left, right, words, labels, N, V, D, K, T, batch_size=1000, epochs=5, test=test,lr_max=0.05, lr_min=0.01)
    # test_model(test)
    test()
