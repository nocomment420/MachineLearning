import numpy as np
import tensorflow as tf
from WordEmbeddingModel import WordEmbeddingModel
import random


class Word2VecModel(WordEmbeddingModel):
    def __init__(self, V, D):
        super().__init__(V, D, savePath="Word2VecModel")

    def get_p_neg(self, sentences):
        V = self.V

        word_count = np.ones(V)
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1

        p_neg = word_count ** 0.75

        p_neg = p_neg / p_neg.sum()

        assert (np.all(p_neg > 0))

        return p_neg

    def drop_common_words(self, sentence, p_neg, threshold=10e-5):
        drop_index = []
        for (i, word) in enumerate(sentence):
            p_drop = 1 - np.sqrt(threshold / p_neg[word])
            if random.random() < p_drop:
                drop_index.append(i)
        return np.delete(sentence, drop_index).tolist()

    def save_skip_grams(self, context_words, pos_words, neg_words, word2indx, context_size):
        np.save("{}/pos_words.npy".format(self.save_path), pos_words)
        np.save("{}/neg_words.npy".format(self.save_path), neg_words)
        np.save("{}/context_words.npy".format(self.save_path), context_words)
        self.save_dic(word2indx, "{}/word2idx.json".format(self.save_path))
        self.save_dic({"context_size": context_size}, "{}/info.json".format(self.save_path))
        self.word2idx = word2indx
    def load_skip_grams(self):
        pos_words = np.load("{}/pos_words.npy".format(self.save_path))
        neg_words = np.load("{}/neg_words.npy".format(self.save_path))
        congtext_words = np.load("{}/context_words.npy".format(self.save_path))
        word2indx = self.load_dic("{}/word2idx.json".format(self.save_path))
        info = self.load_dic("{}/info.json".format(self.save_path))
        self.V = len(word2indx)
        self.word2idx = word2indx
        print("Finished loading skipgrams:")
        print("Positive words: {}".format(pos_words.shape))
        print("Negative words: {}".format(neg_words.shape))
        print("Context Words: {}".format(congtext_words.shape))
        print("\n\n")
        assert (pos_words.shape == neg_words.shape == congtext_words.shape)
        return congtext_words, pos_words, neg_words, word2indx, info["context_size"]

    def generate_skip_grams(self, context_size=10, file_count=2, verbose=False):
        sentences, word2indx = self.get_wiki_data([], file_count=file_count)

        self.V = len(word2indx)
        V = self.V

        print("Generating skipgrams")
        middle_index = int(context_size / 2)
        p_neg = self.get_p_neg(sentences)

        poswords = []
        targets = []
        negwords = []
        for (i, sentence) in enumerate(sentences):
            # for each sentence -> drop common words and pad with START and END tokens
            ajusted_sentence = self.drop_common_words(sentence, p_neg)
            ajusted_sentence = [0] + ajusted_sentence + [1]

            # only train if sentence is large enough to accommodate context size
            n = len(ajusted_sentence)

            if n > context_size + 1:

                # find each skip grams combination in sentence
                for k in range(n - (context_size + 1)):
                    targets += ajusted_sentence[k: k + middle_index] + ajusted_sentence[
                                                                       k + middle_index + 1: k + context_size + 1]
                    poswords += [ajusted_sentence[k + middle_index]] * context_size
                    negwords += [np.random.choice(V, p=p_neg)] * context_size

            if verbose and i % 10000 == 0:
                print("Done sentence {}".format(i))

        poswords = np.array(poswords)
        targets = np.array(targets)
        negwords = np.array(negwords)

        print("Finished generating skipgrams:")
        print("Positive words: {}".format(poswords.shape))
        print("Negative words: {}".format(negwords.shape))
        print("Context Words: {}".format(targets.shape))
        print("\n\n")
        assert (poswords.shape == negwords.shape == targets.shape)

        self.save_skip_grams(context_words=targets, pos_words=poswords, neg_words=negwords, word2indx=word2indx, context_size=context_size)

        return targets, poswords, negwords, word2indx, context_size

    def train(self, epochs=10, lr_max=0.025, lr_min=0.001, batch_size=50):
        # Load skip-grams
        try:
            print("Loading Skip-Grams...")
            context_words, pos_words, neg_words, word2indx, context_size = self.load_skip_grams()
        except Exception:
            print("Error Loading Skip-Grams!")
            context_words, pos_words, neg_words, word2indx, context_size = self.generate_skip_grams()
        print("")

        # init variables
        N = pos_words.shape[0]
        V = len(word2indx)
        D = self.D
        lr_delta = (lr_max - lr_min) / epochs

        # init weights
        W1_init = np.random.randn(V, D) / np.sqrt(V)
        W1 = tf.Variable(W1_init.astype(np.float32))

        W2_init = np.random.randn(D, V) / np.sqrt(D)
        W2 = tf.Variable(W2_init.T.astype(np.float32))

        # Train
        print("Starting training for {} epochs".format(epochs))

        for e in range(epochs):
            lr = lr_max - (lr_delta * e)
            optim = tf.keras.optimizers.Adam(lr=lr)

            total_batches = int(N / (context_size * batch_size))
            for batch in range(total_batches):
                # batch_index = np.random.choice(N, size=batch_size, replace=False)
                start_index = batch * (context_size * batch_size)
                end_index = (batch * (context_size * batch_size)) + (context_size * batch_size)

                pos_batch = pos_words[start_index:end_index]
                neg_batch = neg_words[start_index:end_index]
                context_batch = context_words[start_index:end_index]

                def loss():
                    outer = tf.nn.embedding_lookup(W2, context_batch)

                    inner_pos = tf.nn.embedding_lookup(W1, pos_batch)
                    Z_pos = tf.reduce_sum(inner_pos * outer, axis=1)
                    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_pos,
                                                                       labels=tf.ones(
                                                                           tf.shape(input=Z_pos)))
                    inner_neg = tf.nn.embedding_lookup(W1, neg_batch)
                    Z_neg = tf.reduce_sum(inner_neg * outer, axis=1)
                    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_neg,
                                                                       labels=tf.zeros(
                                                                           tf.shape(input=Z_neg)))

                    return tf.reduce_mean(input_tensor=pos_loss) + tf.reduce_mean(input_tensor=neg_loss)

                optim.minimize(loss, [W1, W2])

                if batch == total_batches - 1:
                    loss = loss().numpy().sum()
                    print("epoch {} - {}".format(e+1, loss))

        self.We = (W1.numpy() + W2.numpy()) / 2
        self.save()

if __name__ == '__main__':
    model = Word2VecModel(10000, 300)
    model.generate_skip_grams(verbose=True, file_count=30)
    # model.train(lr_max=0.2, lr_min=0.01, epochs=50, batch_size=1000)
    model.load()
    model.generate_analogies()


# remove start and end token - DONE
# randomise order the batches are in
# add more negative words than positive
# expand vocab - DONE
# expand file count - DONE