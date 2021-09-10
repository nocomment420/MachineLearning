import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import json

IGNORTOKENS = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
               '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']


def load_data(max_seq_len=100, max_vocab=20000):
    df = pd.read_csv("train.csv/train.csv")
    data = df.to_numpy()
    rows = data[:, 1]
    Y = np.array(data[:, 2:7], dtype=np.int32)
    X = []

    # tokenise scentences
    word2idx = {}
    word_freq = {}
    idx2words = ["PAD"]
    idx = 1
    largest = 0
    for row in rows:
        words = row.split()
        x = []
        for word in words:
            word = word.lower()
            if word not in IGNORTOKENS:
                if word not in word2idx:
                    idx2words.append(word)
                    word2idx[word] = idx
                    word_freq[word] = 0
                    idx += 1
                x.append(word2idx[word])
                word_freq[word] += 1
        if len(x) > largest:
            largest = len(x)
        X.append(x)

    # pad sentences
    if largest > max_seq_len:
        largest = max_seq_len

    for i, x in enumerate(X):
        remainder = largest - len(x)
        if remainder > 0:
            X[i] = [0 for _ in range(remainder)] + x
        else:
            X[i] = x[-largest:]

    # restrict vocab
    word2idx2 = {}
    old_to_new = {}
    idx2word2 = ["PAD"]
    idx = 0
    vocab = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[0:max_vocab]
    for word, count in vocab:
        word2idx2[word] = idx
        idx2word2.append(word)
        idx += 1
        old_to_new[word2idx[word]] = idx
    X2 = []
    for x in X:
        x2 = []
        for t in x:
            if t in old_to_new:
                x2.append(old_to_new[t])
            else:
                x2.append(0)
        X2.append(x2)

    return np.array(X2), Y, word2idx2, idx2word2


def load_embeddings():
    word2vec = {}
    with open("glove.6B.100d.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            word2vec[line[0]] = line[1:]
    return word2vec


def create_embedding_matrix(idx2word, emb_size):
    try:
        embedding = np.load("data/embedding.npy")
    except:
        print("Failed to laod embedding...")
        embedding = np.zeros((len(idx2word), emb_size))
        word2vec = load_embeddings()

        for i in range(1, len(idx2word)):
            word = idx2word[i]
            if word in word2vec:
                embedding[i] = word2vec[word]

        np.save("data/embedding.npy", embedding)
    return embedding


class CNNModel(nn.Module):
    def __init__(self, embedding, in_channels, out_channels1, out_channels2, classes):
        super(CNNModel, self).__init__()
        self.embedding = embedding
        self.conv1 = nn.Conv1d(in_channels, out_channels1, (3,))
        self.max1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(out_channels1, out_channels2, (3,))
        self.max2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(out_channels2, out_channels2, (3,))
        self.max3 = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(256, 100)
        self.linear2 = nn.Linear(100, classes)

    def forward(self, x):
        h = self.embedding[x]
        h = torch.Tensor(h)

        h = self.conv1(h)
        h = self.relu(h)
        h = self.max1(h)

        h = self.conv2(h)
        h = self.relu(h)
        h = self.max2(h)

        h = self.conv3(h)
        h = self.relu(h)
        h = self.max3(h)

        h = self.flat(h)
        h = self.linear1(h)
        h = self.relu(h)

        h = self.linear2(h)
        act = nn.Sigmoid()
        h = act(h)
        return h
class RNNModel(nn.Module):
    def __init__(self, embedding, hidden_size, classes):
        super(RNNModel, self).__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(100, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 100)
        self.linear2 = nn.Linear(100, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.embedding[x]
        h = torch.Tensor(h)

        output, (hn, cn) = self.rnn(h)

        h = self.linear1(output[:,-1,:])
        h = self.relu(h)

        h = self.linear2(h)
        act = nn.Sigmoid()
        h = act(h)
        return h

class BiRNNModel(nn.Module):
    def __init__(self, embedding, hidden_size, classes, seq_len, batch_size):
        super(BiRNNModel, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(100, hidden_size, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, 100)
        self.linear2 = nn.Linear(100, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.embedding[x]
        h = torch.Tensor(h)

        output, (hn, cn) = self.rnn(h)
        output = output.view(output.shape[0],output.shape[1], 2, int(output.shape[2]/2))
        output = torch.cat((output[:,-1,0,:],output[:,-1,1,:]), dim=1)
        h = self.linear1(output)
        h = self.relu(h)

        h = self.linear2(h)
        act = nn.Sigmoid()
        h = act(h)
        return h

def evaluate_model(X_test, Y_test, model, e):
    with torch.no_grad():
        out = model(X_test)
        out = out.numpy()
        out = out.round()

        correct = 0
        for y_h, y_a in zip(out, Y_test):
            for a, b in zip(y_h, y_a):
                if a == b:
                    correct += 1

        acc = correct * 100 / (Y_test.shape[0] * Y_test.shape[1])
        print("epoch {}, acc: {}%".format(e + 1, round(acc, 2)))

def get_data():
    path = "data"
    try:
        X_train = np.load("{}/x_train.npy".format(path))
        X_test= np.load("{}/x_test.npy".format(path))
        Y_train= np.load("{}/y_train.npy".format(path))
        Y_test= np.load("{}/y_test.npy".format(path))

        f = open("{}/word2idx.json".format(path), 'r')
        word2idx = json.load(f)

        f = open("{}/idx2word.json".format(path), 'r')
        idx2word = json.load(f)

    except:
        print("Failed loading from file...")
        x, y, word2idx, idx2word = load_data()

        val_index = np.random.randint(0, x.shape[0], (int(0.2 * x.shape[0])))
        mask = np.ones((x.shape[0],), dtype=bool)
        mask[val_index] = False
        X_train = x[mask]
        X_test = x[val_index]
        Y_train = y[mask]
        Y_test = y[val_index]

        np.save("{}/x_train.npy".format(path), X_train)
        np.save("{}/x_test.npy".format(path), X_test)
        np.save("{}/y_train.npy".format(path), Y_train)
        np.save("{}/y_test.npy".format(path), Y_test)

        with open("{}/word2idx.json".format(path), 'w') as f:
            json.dump(word2idx, f)

        with open("{}/idx2word.json".format(path), 'w') as f:
            json.dump(idx2word, f)

    print("Loaded {} comments, max sequence length : {}, vocabulary size: {}, number of classes: {}".format(X_train.shape[0] + X_test.shape[0],
                                                                                                            X_train.shape[1],
                                                                                                            len(idx2word),
                                                                                                            Y_test.shape[1]))
    print("Train set: {}, Test set: {}".format(X_train.shape[0], X_test.shape[0]))

    return X_train, Y_train, X_test, Y_test, word2idx, idx2word

if __name__ == "__main__":
    # load data and split into train and test
    X_train, Y_train, X_test, Y_test, word2idx, idx2word = get_data()

    # create embedding and model
    embedding = create_embedding_matrix(idx2word, 100)
    # model = CNNModel(embedding, X_train.shape[1], 128, 128, Y_train.shape[1])
    # model = RNNModel(embedding, 128, Y_test.shape[1])
    model = BiRNNModel(embedding, 128, Y_test.shape[1], X_train.shape[1], 128)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    cret = nn.BCELoss()

    # train loop
    for e in range(10000):
        # mini batch sample
        idx = np.random.randint(0, X_train.shape[0], (128,))
        x_b = X_train[idx]
        y_b = Y_train[idx]

        # gradient descent
        optim.zero_grad()
        out = model(x_b)
        loss = cret(out, torch.FloatTensor(y_b))
        loss.backward()
        optim.step()

        # evaluation
        if e % 100 == 0:
            evaluate_model(X_test, Y_test, model, e)

