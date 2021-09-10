import numpy as np
import torch
import torch.nn as nn
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
IGNORTOKENS = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
               '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']


def load_data(max_seq_len=100, max_vocab=20000):
    # get lines from file
    inputs = []
    outputs = []
    with open("robert_frost.txt") as f:
        lines = f.readlines()
        for line in lines:
            if not line or line == "\n":
                continue
            line = line.replace("\n", "")
            inputs.append("<sos> " + line)
            outputs.append(line + " <eos>")

    # tokenize sentences
    word2idx = {"PAD": 0, "<sos>": 1, "<eos>": 2}
    word_freq = {"PAD": 10000000000, "<sos>": 100000000, "<eos>": 10000000}
    idx2words = ["PAD", "<sos>", "<eos>"]
    idx = 3
    largest = 0

    # inputs
    X = []
    for row in inputs:
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

    # ouputs
    Y = []
    for row in outputs:
        words = row.split()
        y = []
        for word in words:
            word = word.lower()
            if word not in IGNORTOKENS:
                if word not in word2idx:
                    idx2words.append(word)
                    word2idx[word] = idx
                    word_freq[word] = 0
                    idx += 1
                y.append(word2idx[word])
                word_freq[word] += 1
        if len(y) > largest:
            largest = len(y)
        Y.append(y)

    # pad sentences
    if largest > max_seq_len:
        largest = max_seq_len

    for i, x in enumerate(X):
        remainder = largest - len(x)
        if remainder > 0:
            X[i] = x + [0 for _ in range(remainder)]
        else:
            X[i] = x[-(largest + 1):]

    for i, y in enumerate(Y):
        remainder = largest - len(y)
        if remainder > 0:
            Y[i] = y + [0 for _ in range(remainder)]
        else:
            Y[i] = y[-largest:]

    # # restrict vocab
    # word2idx2 = {}
    # idx2word2 = []
    # old_to_new = {}
    # idx = 0
    # vocab = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[0:max_vocab]
    # for word, count in vocab:
    #     word2idx2[word] = idx
    #     idx2word2.append(word)
    #     old_to_new[word2idx[word]] = idx
    #     idx += 1
    #
    # X2 = []
    # for x in X:
    #     x2 = []
    #     for t in x:
    #         if t in old_to_new:
    #             x2.append(old_to_new[t])
    #         else:
    #             x2.append(0)
    #     X2.append(x2)
    #
    # Y2 = []
    # for y in Y:
    #     y2 = []
    #     for t in y:
    #         if t in old_to_new:
    #             y2.append(old_to_new[t])
    #         else:
    #             y2.append(0)
    #     Y2.append(y2)
    #
    # return np.array(X2), np.array(Y2), word2idx2, idx2word2
    return np.array(X), np.array(Y), word2idx, idx2words


def get_data():
    path = "data"
    try:
        X_train = np.load("{}/x_train.npy".format(path))
        X_test = np.load("{}/x_test.npy".format(path))
        Y_train = np.load("{}/y_train.npy".format(path))
        Y_test = np.load("{}/y_test.npy".format(path))

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

    print("Loaded {} lines, max sequence length : {}, vocabulary size: {}".format(X_train.shape[0] + X_test.shape[0],
                                                                                  X_train.shape[1],
                                                                                  len(idx2word)))
    print("Train set: {}, Test set: {}".format(X_train.shape[0], X_test.shape[0]))

    return X_train, Y_train, X_test, Y_test, word2idx, idx2word


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


class Model2(nn.Module):
    def __init__(self, embedding, hidden_size, num_classes):
        super(Model2, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding))
        self.rnn = nn.LSTM(embedding.shape[1], hidden_size)  # , batch_first=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        h = torch.Tensor(x).long().to(device)

        h = self.embedding.forward(h)  # h = N x T x D
        h = h.view(h.shape[1], h.shape[0], h.shape[2])
        h0 = torch.randn(1, x.shape[0], self.hidden_size).to(device)

        c0 = torch.randn(1, x.shape[0], self.hidden_size).to(device)

        h, hidden = self.rnn(h, (h0, c0))
        # h= self.tanh(h)
        h = h.view(h.shape[0] * h.shape[1], -1)  # output = NT x H
        # h = self.dropout(h)
        h = self.linear(h)  # h = N x T x K
        # h = h.view(h.shape[0] * h.shape[1], h.shape[2])     # h = NT x K
        # h = self.softmax(h)
        # r = nn.ReLU()
        # h = r(h)
        return h


def generate_sentence(model, word2idx, idx2word):
    sentence = [word2idx["<sos>"]]
    sentence_string = ""
    for i in range(12):
        output = model(np.array([sentence]))
        s = nn.Softmax(dim=1)
        output = s(output)
        output = output.detach().cpu().numpy()
        probs = output[-1]
        #probs[0] = 0
        # probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        sentence.append(idx)
        sentence_string += " {}".format(idx2word[idx])
        if idx2word[idx] == "<eos>":
            break
    print(sentence_string)


def test():
    # get lines from file
    inputs = []
    outputs = []
    for line in open('robert_frost.txt', encoding="utf-8"):
        line = line.rstrip()
        if not line:
            continue

        inputs.append( '<sos> ' + line)
        outputs.append(line + ' <eos>')

    all_lines = inputs + outputs

    # convert the sentences (strings) into integers
    tokenizer = Tokenizer(num_words=3000, filters='')
    tokenizer.fit_on_texts(all_lines)
    input_sequences = tokenizer.texts_to_sequences(inputs)
    target_sequences = tokenizer.texts_to_sequences(outputs)

    # find max seq length
    max_sequence_length_from_data = max(len(s) for s in input_sequences)
    print('Max sequence length:', max_sequence_length_from_data)

    # get word -> integer mapping
    word2idx = tokenizer.word_index
    print('Found %s unique tokens.' % len(word2idx))
    assert ('<sos>' in word2idx)
    assert ('<eos>' in word2idx)

    # pad sequences so that we get a N x T matrix
    max_sequence_length = min(max_sequence_length_from_data, 100)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
    print('Shape of data tensor:', input_sequences.shape)

    print('Loading word vectors...')
    word2vec = {}
    with open("glove.6B.100d.txt", encoding="utf-8") as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(3000, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, 100))
    for word, i in word2idx.items():
        if i < 3000:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    model = Model2(embedding_matrix, 75, 3000).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    cret = nn.CrossEntropyLoss(ignore_index=0)

    # train loop
    for e in range(200000):
        # mini batch sample
        idx = np.random.randint(0, input_sequences.shape[0], (1,))
        x_b = input_sequences[idx]
        y_b = target_sequences[idx]

        # gradient descent
        # out = out.view(out.shape[1], out.shape[2],out.shape[0],1)
        y_b = torch.LongTensor(y_b).to(device)
        y_b = y_b.view(-1)
        optim.zero_grad()
        out = model(x_b)

        loss = cret(out, y_b)
        loss.backward()
        optim.step()

        # evaluation
        if e % 1000 == 0:
            print(loss)
            with torch.no_grad():
                # out = model(X_test)
                # out = out.detach().numpy()
                # out = out.argmax(axis=1)
                #
                # label = Y_test.reshape((Y_test.shape[0] * Y_test.shape[1]))
                #
                # correct = 0
                # for p, t in zip(out, label):
                #     if p == t:
                #         correct += 1
                # print("epoch {} acc: {}%".format(e + 1, round(correct * 100 / label.shape[0], 2)))
                idx2word = {v: k for k, v in word2idx.items()}
                generate_sentence(model, word2idx, idx2word)
                # generate_sentence(model, word2idx, idx2word)
                # generate_sentence(model, word2idx, idx2word)
                print("\n\n\n")
    print("Finished training, generating 20 sentences...")
    for i in range(20):
        generate_sentence(model, word2idx, idx2word)


if __name__ == "__main__":
    test()
    # X_train, Y_train, X_test, Y_test, word2idx, idx2word = get_data()
    # embedding = create_embedding_matrix(idx2word, 100)
    # model = Model(embedding, 25, len(idx2word))
    # optim = torch.optim.Adam(model.parameters(), lr=0.01)
    # cret = nn.CrossEntropyLoss()
    #
    # # train loop
    # for e in range(200000):
    #     # mini batch sample
    #     idx = np.random.randint(0, X_train.shape[0], (128,))
    #     x_b = X_train[idx]
    #     y_b = Y_train[idx]
    #
    #     # gradient descent
    #     out = model(x_b)
    #     # out = out.view(out.shape[1], out.shape[2],out.shape[0],1)
    #     y_b = torch.LongTensor(y_b)
    #     y_b = y_b.view(-1)
    #     optim.zero_grad()
    #
    #     loss = cret(out, y_b)
    #     loss.backward()
    #     optim.step()
    #
    #     # evaluation
    #     if e % 1000 == 0:
    #         print(loss)
    #         with torch.no_grad():
    #             out = model(X_test)
    #             out = out.detach().numpy()
    #             out = out.argmax(axis=1)
    #
    #             label = Y_test.reshape((Y_test.shape[0] * Y_test.shape[1]))
    #
    #             correct = 0
    #             for p, t in zip(out, label):
    #                 if p == t:
    #                     correct += 1
    #             print("epoch {} acc: {}%".format(e + 1, round(correct * 100 / label.shape[0], 2)))
    #             generate_sentence(model, word2idx, idx2word)
    #             generate_sentence(model, word2idx, idx2word)
    #             generate_sentence(model, word2idx, idx2word)
    #             print("\n\n\n")
    # print("Finished training, generating 20 sentences...")
    # for i in range(20):
    #     generate_sentence(model, word2idx, idx2word)
