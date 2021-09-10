import numpy as np
import torch
import torch.nn as nn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random

# data from https://www.manythings.org/anki
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
MAX_VOCAB = 20000
HIDDEN_DIM = 256
EPOCHS = 2000000
BATCH_SIZE = 1
EVAL_STEP = 1000
MAX_SENTENCES = 10000
VAL_P = 0.15
VAL_SIZE = 2
TEACHER_PROB = 0.6

# get inputs from file
input_sentences = []
target_sentences = []
input_target_sentences = []

n = 0
for line in open('fra.txt', encoding="utf-8"):
    line = line.rstrip()
    if not line:
        continue
    all = line.split("\t")
    input_sentences.append('<sos> ' + all[0])
    target_sentences.append(all[1] + ' <eos>')
    input_target_sentences.append('<sos> ' + all[1])
    n += 1
    if n >= MAX_SENTENCES:
        break

# tokenize input sentences
origin_tokenizer = Tokenizer(num_words=MAX_VOCAB, filters='')
origin_tokenizer.fit_on_texts(input_sentences)
word2idx_origin = origin_tokenizer.word_index
print('Found %s unique origin tokens.' % len(word2idx_origin))
assert ('<sos>' in word2idx_origin)

input_sentences = origin_tokenizer.texts_to_sequences(input_sentences)

# tokenize output sentences
target_tokenizer = Tokenizer(num_words=MAX_VOCAB, filters='')
target_tokenizer.fit_on_texts(target_sentences + input_target_sentences)
word2idx_target = target_tokenizer.word_index
print('Found %s unique target tokens.' % len(word2idx_target))
assert ('<sos>' in word2idx_target)
assert ('<eos>' in word2idx_target)

target_sentences = target_tokenizer.texts_to_sequences(target_sentences)
input_target_sentences = target_tokenizer.texts_to_sequences(input_target_sentences)

# pad origin sentences
max_input = max(len(s) for s in input_sentences)
print('Max input sequence length:', max_input)
max_sequence_length = min(max_input, 100)

input_sentences = pad_sequences(input_sentences, maxlen=max_sequence_length, padding='pre')

# pad target sentences
max_output = max(len(s) for s in target_sentences)
print('Max target sequence length:', max_output)

max_sequence_length = min(max_output, 100)
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
num_words_input = min(MAX_VOCAB, len(word2idx_origin) + 1)
embedding_matrix_input = np.zeros((num_words_input, 100))
for word, i in word2idx_origin.items():
    if i < MAX_VOCAB:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix_input[i] = embedding_vector

# prepare target embedding matrix
num_words_output = min(MAX_VOCAB, len(word2idx_target) + 1)
embedding_matrix_output = np.zeros((num_words_output, 100))


class Encoder(nn.Module):
    def __init__(self, embedding):
        super(Encoder, self).__init__()
        self.embedding_input = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))

        self.encode_lstm = nn.LSTM(embedding.shape[1], HIDDEN_DIM)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # encode
        indexes = torch.Tensor(x).long().to(device)

        embeddings = self.embedding_input.forward(indexes)
        # embeddings = self.dropout(embeddings)

        embeddings = embeddings.view(embeddings.shape[1], embeddings.shape[0], embeddings.shape[2])

        out, (h, c) = self.encode_lstm(embeddings)

        return (h, c)


class Decoder(nn.Module):
    def __init__(self, embedding):
        super(Decoder, self).__init__()
        self.embedding_target = nn.Embedding(embedding.shape[0], embedding.shape[1])

        self.decode_lstm = nn.LSTM(embedding.shape[1], HIDDEN_DIM)

        self.linear = nn.Linear(HIDDEN_DIM, num_words_output)


    def forward(self, x, hidden):
        # decode
        indexes = torch.Tensor(x).long().to(device)

        embeddings = self.embedding_target.forward(indexes)
        # embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(1, embeddings.shape[0], embeddings.shape[1])

        out, (h, c) = self.decode_lstm(embeddings, hidden)

        out = self.linear(out)

        return out, (h, c)


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
    (h, c) = encoder([input])
    for i in range(max_output):
        out, (h, c) = decoder([output[i]], (h, c))

        softmax = nn.Softmax(dim=2)
        out = softmax(out)
        out = out.detach().cpu().numpy()
        probs = out[0, 0]
        idx = probs.argmax()

        output.append(idx)

        if idx > 0:
            output_string += " {}".format(idx2word_target[idx])
            if idx2word_target[idx] == "<eos>":
                break

    print("Input: " + idx_to_sentence(input, idx2word_input))
    print("Translation: " + output_string)
    print("Actual: " + idx_to_sentence(actual, idx2word_target))
    print("\n")


encoder = Encoder(embedding_matrix_input)
encoder.to(device)
decoder = Decoder(embedding_matrix_output)
decoder.to(device)

optim_enc = torch.optim.Adam(encoder.parameters())
optim_dec = torch.optim.Adam(decoder.parameters())

criteria = nn.CrossEntropyLoss(ignore_index=0)

tloss = 0

for e in range(EPOCHS):
    # gradient descent
    optim_enc.zero_grad()
    optim_dec.zero_grad()

    # sample mini batch
    idx = np.random.randint(0, input_sentences_train.shape[0], (1,))
    x_orgin = input_sentences_train[idx]
    x_target = input_target_sentences_train[idx]
    y = torch.LongTensor(target_sentences_train[idx]).to(device)

    loss = 0

    (h, c) = encoder(x_orgin)

    # use teacher forcing
    if random.random() < TEACHER_PROB:
        for i in range(x_target.shape[1]):
            out, (h, c) = decoder(x_target[:, i], (h, c))
            loss += criteria(out[0], y[:, i])

    # dont use teacher forcing
    else:
        target = x_target[:, 0]
        for i in range(x_target.shape[1]):
            out, (h, c) = decoder(target, (h, c))
            loss += criteria(out[0], y[:, i])

            softmax = nn.Softmax(dim=2)
            out = softmax(out)
            out = out.detach().cpu().numpy()
            probs = out[0]
            idx = probs.argmax(axis=1)

            target = idx

    loss.backward()
    tloss += loss.item()
    optim_enc.step()
    optim_dec.step()

    if e % EVAL_STEP == 0:
        print("Epoch {}".format(e + 1))
        print("Loss: {}".format(tloss))
        tloss = 0

        # ganerate translations
        with torch.no_grad():
            idxs = np.random.randint(0, input_target_sentences_test.shape[0], (VAL_SIZE,))
            for idx in idxs:
                input = input_sentences_test[idx]
                actual = target_sentences_test[idx]

                translate(encoder, decoder, input, actual, word2idx_origin, word2idx_target)
            print("\n\n")
