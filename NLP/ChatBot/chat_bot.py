import json

import numpy as np
import torch
import torch.nn as nn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
M1 = 400
M2 = 400
ATTENTIION_DIM = 10
MAX_VOCAB = 30000
MAX_SENTENCES = 20000
VAL_P = 0.15
EPOCHS = 50000
MAX_SEQ_LEN = 10

class Encoder(nn.Module):
    def __init__(self, embedding):
        super(Encoder, self).__init__()
        self.embedding_input = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))

        self.encode_lstm = nn.LSTM(embedding.shape[1], M1, bidirectional=True)

        self.hidden_size = M1

        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        # encode
        indexes = torch.Tensor(x).long().to(device)
        embeddings = self.dropout(self.embedding_input.forward(indexes))  # N x T x H

        hiddens = torch.zeros((embeddings.shape[1], embeddings.shape[0], 2 * self.hidden_size)).to(
            device)  # T x N x 2 M1
        for t in range(embeddings.shape[1]):
            current = embeddings[:, t, :].unsqueeze(0)  # 1 x N x H
            out, (h, c) = self.encode_lstm(current)  # h = 2 x N x M1
            h = h.view(h.shape[1], -1)  # h = N x 2 M1
            hiddens[t] = h

        hiddens = hiddens.view(hiddens.shape[1], hiddens.shape[0], hiddens.shape[2])  # N x T x 2 M1

        return hiddens


class Decoder(nn.Module):
    def __init__(self, embedding, num_words):
        super(Decoder, self).__init__()
        self.embedding_target = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))

        self.decode_lstm = nn.LSTM(embedding.shape[1] + 2 * M1, M2)
        self.linear_attention_1 = nn.Linear(2 * M1 + M2, ATTENTIION_DIM)
        self.linear_attention_2 = nn.Linear(ATTENTIION_DIM, 1)
        self.softmax_over_time = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(M2, num_words)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.35)

    def forward(self, encoder_hidden, decoder_state, decoder_c, decorder_input):
        Tx = encoder_hidden.shape[1]
        N = encoder_hidden.shape[0]

        # concat state and hidden
        s = decoder_state.repeat((Tx, 1, 1)).view(N, Tx, -1)  # 1 x N x M2
        attention_in = torch.cat((s, encoder_hidden), dim=2)  # N x Tx x 2M1 + M2

        # pass through attention network
        z = self.linear_attention_1(attention_in)  # 1 x N x AD
        z = self.tanh(z)
        attention = self.linear_attention_2(z)  # N x Tx x 1

        # softmax over time and dot with hiddens
        alphas = self.softmax_over_time(attention).view(N, 1, -1)  # N x 1 x Tx
        context = torch.bmm(alphas, encoder_hidden).view(N, -1)  # N x 2M1

        # embed target words
        indexes = torch.Tensor(decorder_input).long().to(device)
        embeddings = self.dropout(self.embedding_target.forward(indexes))  # N x E

        # catenate embedded targets and context
        decoder_input = torch.cat((embeddings, context), dim=1)  # N x 1 x 2M1 + E

        # pass through decoder and dense layers
        out, (h, c) = self.decode_lstm(decoder_input.unsqueeze(0), (decoder_state, decoder_c))  # out = N x M2
        out = self.linear_out(out.view(N, -1))  # out = N x num_words

        return out, (h, c)


def load():
    path = "data"

    try:
        input_sentences_train = np.load("{}/input_sentences_train.npy".format(path))
        input_sentences_test = np.load("{}/input_sentences_test.npy".format(path))
        target_sentences_train = np.load("{}/target_sentences_train.npy".format(path))
        target_sentences_test = np.load("{}/target_sentences_test.npy".format(path))
        input_target_sentences_train = np.load("{}/input_target_sentences_train.npy".format(path))
        input_target_sentences_test = np.load("{}/input_target_sentences_test.npy".format(path))
        embedding_matrix = np.load("{}/embedding_matrix.npy".format(path))

        f = open("{}/word2idx.json".format(path), 'r')
        word2idx = json.load(f)

        f = open("{}/settings.json".format(path), 'r')
        settings = json.load(f)

        return (settings,
                word2idx,
                input_sentences_train,
                input_sentences_test,
                target_sentences_train,
                target_sentences_test,
                input_target_sentences_train,
                input_target_sentences_test,
                embedding_matrix)


    except Exception as e:

        print("Failed loading from file...")

        # get inputs from file
        input_sentences = []
        target_sentences = []
        input_target_sentences = []

        questions = np.load("questions.npy")
        answers = np.load("answers.npy")
        for i in range(questions.shape[0]):
            if len(questions[i]) > 0 and len(answers[i]) > 0:
                if len(questions[i].split()) < MAX_SEQ_LEN and len(answers[i].split()) < MAX_SEQ_LEN:
                    input_sentences.append("<sos> " + questions[i].replace("\n",""))
                    target_sentences.append(answers[i].replace("\n","") + " <eos>")
                    input_target_sentences.append("<sos> " + answers[i].replace("\n",""))

        # tokenize input sentences
        tokenizer = Tokenizer(num_words=MAX_VOCAB, filters='')
        tokenizer.fit_on_texts(input_sentences + target_sentences)
        word2idx = tokenizer.word_index
        print('Found %s unique origin tokens.' % len(word2idx))
        assert ('<sos>' in word2idx)
        assert ('<eos>' in word2idx)

        input_sentences = tokenizer.texts_to_sequences(input_sentences)
        target_sentences = tokenizer.texts_to_sequences(target_sentences)
        input_target_sentences = tokenizer.texts_to_sequences(input_target_sentences)

        with open("{}/word2idx.json".format(path), 'w') as f:
            json.dump(word2idx, f)



        # pad origin sentences
        max_input = max(len(s) for s in input_sentences)
        print('Max input sequence length:', max_input)
        max_sequence_length = min(max_input, MAX_SEQ_LEN)
        num_trunc = sum([1 if len(s) > max_sequence_length else 0 for s in input_sentences])
        print("Truncated {} input sequences to have len {}".format(num_trunc, max_sequence_length))
        input_sentences = pad_sequences(input_sentences, maxlen=max_sequence_length, padding='pre')

        # pad target sentences
        max_output = max(len(s) for s in target_sentences)
        print('Max target sequence length:', max_output)
        max_sequence_length = min(max_output, MAX_SEQ_LEN)
        num_trunc = sum([1 if len(s) > max_sequence_length else 0 for s in target_sentences])
        print("Truncated {} target sequences to have len {}".format(num_trunc, max_sequence_length))

        target_sentences = pad_sequences(target_sentences, maxlen=max_sequence_length, padding='post')
        input_target_sentences = pad_sequences(input_target_sentences, maxlen=max_sequence_length, padding='post')

        print('Shape of inputs:', input_sentences.shape)
        print('Shape of targets:', target_sentences.shape)
        print('Shape of input targets:', input_target_sentences.shape)

        # split into train and test
        val_index = np.random.randint(0, input_sentences.shape[0], (int(VAL_P * input_sentences.shape[0])))
        mask = np.ones((input_sentences.shape[0],), dtype=bool)
        mask[val_index] = False

        input_sentences_train = input_sentences[mask]
        input_sentences_test = input_sentences[val_index]

        target_sentences_train = target_sentences[mask]
        target_sentences_test = target_sentences[val_index]

        input_target_sentences_train = input_target_sentences[mask]
        input_target_sentences_test = input_target_sentences[val_index]

        # load input word2vec
        print('Loading input word vectors...')
        word2vec = {}
        with open("../PoetryGeneration/glove.6B.100d.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print('Found %s word vectors.' % len(word2vec))

        # prepare input embedding matrix
        print('Filling pre-trained embeddings...')
        num_words = min(MAX_VOCAB, len(word2idx) + 1)
        embedding_matrix = np.zeros((num_words, 100))
        for word, i in word2idx.items():
            if i < MAX_VOCAB:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector


        np.save("{}/input_sentences_train.npy".format(path), input_sentences_train)
        np.save("{}/input_sentences_test.npy".format(path), input_sentences_test)
        np.save("{}/target_sentences_train.npy".format(path), target_sentences_train)
        np.save("{}/target_sentences_test.npy".format(path), target_sentences_test)
        np.save("{}/input_target_sentences_train.npy".format(path), input_target_sentences_train)
        np.save("{}/input_target_sentences_test.npy".format(path), input_target_sentences_test)
        np.save("{}/embedding_matrix.npy".format(path), embedding_matrix)

        settings = {"num_words": num_words}
        with open("{}/settings.json".format(path), 'w') as f:
            json.dump(settings, f)

        return (settings,
                word2idx,
                input_sentences_train,
                input_sentences_test,
                target_sentences_train,
                target_sentences_test,
                input_target_sentences_train,
                input_target_sentences_test,
                embedding_matrix)


def idx_to_sentence(indexes, idx2word):
    sentence = ""
    for index in indexes:
        if index in idx2word:
            sentence += " {}".format(idx2word[index])
    return sentence


def translate(encoder, decoder, input, actual, word2idx_input, word2idx_target):
    idx2word_input = {v: k for k, v in word2idx_input.items()}
    idx2word_target = {v: k for k, v in word2idx_target.items()}

    output = [word2idx_target["<sos>"]]
    output_string = ""
    h = encoder([input])
    c = torch.zeros(1, 1, M2).to(device)
    s = torch.zeros(1, 1, M2).to(device)
    for i in range(MAX_SEQ_LEN):
        out, (s, c) = decoder(h, s, c, [output[i]])

        softmax = nn.Softmax(dim=1)
        out = softmax(out)
        out = out.detach().cpu().numpy()
        probs = out[0]
        # idx = probs.argmax()
        idx = np.random.choice(len(probs), p=probs)

        output.append(idx)
        # output.append(actual[i])

        if idx > 0:
            output_string += " {}".format(idx2word_target[idx])
            if idx2word_target[idx] == "<eos>":
                break

    print("Input: " + idx_to_sentence(input, idx2word_input))
    print("Translation: " + output_string)
    print("Actual: " + idx_to_sentence(actual, idx2word_target))
    print("\n")


(settings,
word2idx,
input_sentences_train,
input_sentences_test,
target_sentences_train,
target_sentences_test,
input_target_sentences_train,
input_target_sentences_test,
embedding_matrix) = load()

num_words = settings["num_words"]

encoder = Encoder(embedding_matrix).to(device)
decoder = Decoder(embedding_matrix, num_words).to(device)

optim_enc = torch.optim.Adam(encoder.parameters())
optim_dec = torch.optim.Adam(decoder.parameters())

criteria = nn.CrossEntropyLoss(ignore_index=0)

tloss = 0
loss_history = []
for e in range(EPOCHS):
    # gradient descent
    optim_enc.zero_grad()
    optim_dec.zero_grad()

    # sample mini batch
    idx = np.random.randint(0, input_sentences_train.shape[0], (64,))
    loss = 0

    x_orgin = input_sentences_train[idx]
    x_target = input_target_sentences_train[idx]
    y = torch.LongTensor(target_sentences_train[idx]).to(device)

    N = x_orgin.shape[0]
    Tx = x_orgin.shape[1]
    Ty = x_target.shape[1]

    h = encoder(x_orgin).to(device)  # N x T x 2 M1
    s = torch.zeros((1, N, M2)).to(device)  # 1 x N x M2
    c = torch.zeros((1, N, M2)).to(device)  # 1 x N x M2

    for t in range(Ty):
        out, (s, c) = decoder(h, s, c, x_target[:, t])
        loss += criteria(out, y[:, t])

    loss.backward()

    optim_enc.step()
    optim_dec.step()

    tloss += loss.item()

    if e % 400 == 0:
        print("Epoch {}".format(e + 1))
        print("Loss: {}".format(tloss))

        loss_history.append(tloss)
        tloss = 0

        # ganerate translations
        with torch.no_grad():
            idxs = np.random.randint(0, input_target_sentences_test.shape[0], (2,))
            for idx in idxs:
                input = input_sentences_test[idx]
                actual = target_sentences_test[idx]

                translate(encoder, decoder, input, actual, word2idx, word2idx)
            print("\n\n")

import matplotlib.pyplot as plt
plt.plot(loss_history, label='acc')
plt.show()
