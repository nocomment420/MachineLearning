from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import json
import os

class WordEmbeddingModel:

    def __init__(self, V, D, savePath=None):
        self.V = V
        self.D = D
        self.word2idx = None
        self.idx2word = None
        self.We = None
        self.save_path = savePath

    def fit(self):
        pass

    def generate_analogies(self):

        self.idx2word = {i: w for w, i in self.word2idx.items()}

        print("***** GENRATING ANALOGIES *****")

        self.test_analogy('king', 'man', 'queen', 'woman')
        self.test_analogy('king', 'prince', 'queen', 'princess')
        self.test_analogy('miami', 'florida', 'dallas', 'texas')
        self.test_analogy('einstein', 'scientist', 'picasso', 'painter')
        self.test_analogy('japan', 'sushi', 'germany', 'bratwurst')
        self.test_analogy('man', 'woman', 'he', 'she')
        self.test_analogy('man', 'woman', 'uncle', 'aunt')
        self.test_analogy('man', 'woman', 'brother', 'sister')
        self.test_analogy('man', 'woman', 'husband', 'wife')
        self.test_analogy('man', 'woman', 'actor', 'actress')
        self.test_analogy('man', 'woman', 'father', 'mother')
        self.test_analogy('heir', 'heiress', 'prince', 'princess')
        self.test_analogy('nephew', 'niece', 'uncle', 'aunt')
        self.test_analogy('france', 'paris', 'japan', 'tokyo')
        self.test_analogy('france', 'paris', 'china', 'beijing')
        self.test_analogy('february', 'january', 'december', 'november')
        self.test_analogy('france', 'paris', 'germany', 'berlin')
        self.test_analogy('week', 'day', 'year', 'month')
        self.test_analogy('week', 'day', 'hour', 'minute')
        self.test_analogy('france', 'paris', 'italy', 'rome')
        self.test_analogy('paris', 'france', 'rome', 'italy')
        self.test_analogy('france', 'french', 'england', 'english')
        self.test_analogy('japan', 'japanese', 'china', 'chinese')
        self.test_analogy('china', 'chinese', 'america', 'american')
        self.test_analogy('japan', 'japanese', 'italy', 'italian')
        self.test_analogy('japan', 'japanese', 'australia', 'australian')
        self.test_analogy('walk', 'walking', 'swim', 'swimming')

    def test_analogy(self, pos1, neg1, pos2, neg2):
        W = self.We
        V, D = W.shape
        word2idx = self.word2idx

        print("testing: {} - {} = {} - {}".format(pos1, neg1, pos2, neg2))
        for w in (pos1, neg1, pos2, neg2):
            if w not in word2idx:
                print("{} not in word2idx\n\n".format(w))
                return

        p1 = W[word2idx[pos1]]
        n1 = W[word2idx[neg1]]
        p2 = W[word2idx[pos2]]
        n2 = W[word2idx[neg2]]

        vec = p1 - n1 + n2

        distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
        idx = distances.argsort()[:10]

        # pick best word
        best_idx = -1
        keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break

        print("got: {} - {} = {} - {}".format(pos1, neg1, self.idx2word[best_idx], neg2))
        print("closest 10 words:")
        for i in idx:
            print(self.idx2word[i], distances[i])

        print("dist to {}: {}".format(pos2, self.cos_dist(p2, vec)))
        print("\n\n")

    def cos_dist(self, a, b):
        return 1 - (a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save(self):
        self.save_dic(self.word2idx, "{}/word2idx.json".format(self.save_path))
        np.save("{}/We.npy".format(self.save_path), self.We)

    def load(self):
        self.word2idx = self.load_dic("{}/word2idx.json".format(self.save_path))
        self.We = np.load("{}/We.npy".format(self.save_path))

    def save_dic(self, dic, file_name):
        a_file = open(file_name, "w")
        json.dump(dic, a_file)
        a_file.close()

    def load_dic(self, filename):
        a_file = open(filename, "r")
        output = a_file.read()
        return json.loads(output)

    def get_wiki_data(self, keep_words, file_count=2):

        # prepare file names
        numbers = []
        for i in range(file_count):
            if (i + 1) < 10:
                numbers.append("0{}".format(i + 1))
            else:
                numbers.append("{}".format(i + 1))
        files = ["../enwiki-preprocessed/enwiki-20180401-pages-articles1.xml-p10p30302-{}.txt".format(no) for no in
                 numbers]

        word2indx = {'START': 0, 'END': 1}
        indx2word = ['START', 'END']
        word_counts = {
            0: float('inf'),
            1: float('inf')
        }
        sentences = []
        i = 2

        print("starting to read {} files".format(len(files)))

        for file in files:
            file_path = os.path.dirname(__file__)
            if file_path != "":
                os.chdir(file_path)
            with open(file, encoding='utf-8') as f:
                for line in f:
                    current_sentence = []
                    for word in line.split():
                        word = word.lower()

                        if word not in word2indx:
                            word2indx[word] = i
                            word_counts[i] = 1
                            i += 1
                            indx2word.append(word)
                        else:
                            word_counts[word2indx[word]] += 1

                        current_sentence.append(word)
                    sentences.append(current_sentence)

        print("found {} tokens".format(len(word2indx)))

        for word in keep_words:
            word_counts[word2indx[word]] = float('inf')

        sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

        word2indx_small = {}
        new_indx = 0
        old_index_new_index_map = {}
        for index, count in sorted_word_counts[:(self.V - 1)]:
            word = indx2word[index]
            word2indx_small[word] = new_indx
            old_index_new_index_map[index] = new_indx
            new_indx += 1

        word2indx_small['UNKNOWN'] = new_indx
        unknown = new_indx

        print("kept {} tokens".format(len(word2indx_small)))

        sentences_small = []
        for sentence in sentences:
            if len(sentence) > 1:
                new_sentence = [old_index_new_index_map[word2indx[current_word]]
                                if word2indx[current_word]
                                   in old_index_new_index_map
                                else unknown
                                for current_word in sentence]
                sentences_small.append(new_sentence)

        print("Got {} sentences".format(len(sentences_small)))

        self.word2idx = word2indx_small

        return sentences_small, word2indx_small
