import datetime
import numpy as np
import json

class RecursiveNeuralTree:
    def __init__(self):
        self.left = None
        self.right = None
        self.word = None
        self.score = None

    def validate(self):
        # All nodes must have score
        if self.score is None:
            return "Score is none"

        # leaf nodes must have a word
        if self.left is None and self.right is None:
            if self.word is None:
                return "Leaf node with no word"

        # parents nodes must have 2 children
        elif self.left is None or self.right is None or self.word is not None:
            return "Parent node with only 1 child or has a word"

        return None

    def convert_to_sequence(self, left, right, words, labels):

        if self.left is not None:
            left_index = self.left.convert_to_sequence(left, right, words, labels)
        else:
            left_index = None

        if self.right is not None:
            right_index = self.right.convert_to_sequence(left, right, words, labels)
        else:
            right_index = None

        index = len(words)

        # set word for this tree
        if self.word is None:
            words.append(-1)
        else:
            words.append(self.word)

        # set label for this tree
        if self.score is None:
            labels.append(-1)
        else:
            labels.append(self.score)

        # set left child for this tree
        if left_index is not None:
            left.append(left_index)
        else:
            left.append(-1)

        # set right child for this tree
        if right_index is not None:
            right.append(right_index)
        else:
            right.append(-1)

        return index


def is_number(word):
    try:
        return True, int(word)
    except:
        return False, None


def parse_line(i, line):
    tree = RecursiveNeuralTree()
    while i <= len(line) - 1:
        char = line[i]
        if char != " ":
            # new node
            if char == "(":
                if tree.left is None:
                    tree.left, i = parse_line(i + 1, line)
                    errors = tree.left.validate()
                elif tree.right is None:
                    tree.right, i = parse_line(i + 1, line)
                    errors = tree.right.validate()
                else:
                    errors = "New node but left and right nodes are taken"
                if errors is not None:
                    raise Exception("Error: {} - \"{}\"".format(errors, line[i - 10:i]))


            # end of node
            elif char == ")":
                return tree, i

            else:
                is_num, number = is_number(char)

                # rating
                if is_num and i < len(line) - 1 and line[i + 1] == " ":
                    tree.score = number

                # word
                else:
                    if tree.word is None:
                        tree.word = char
                    else:
                        tree.word += char
        i += 1
    return tree


def build_trees(tree_strings):
    tree_roots = []
    for tree_string in tree_strings:
        try:
            root, i = parse_line(1, tree_string)
            assert i == len(tree_string) - 1
            tree_roots.append(root)
        except Exception as e:
            print("{}\n".format(str(e)))

    return tree_roots


def load_trees_from_file(filename):
    tree_strings = []

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line != "\n":
                tree_string = line.rstrip()
                tree_strings.append(tree_string)

    return build_trees(tree_strings)


def convert_trees_to_word2idx(tree, word2idx, word_idx, can_add):
    if tree.left is not None:
        word2idx, word_idx = convert_trees_to_word2idx(tree.left, word2idx, word_idx, can_add)

    if tree.right is not None:
        word2idx, word_idx = convert_trees_to_word2idx(tree.right, word2idx, word_idx, can_add)

    if tree.word is not None:
        if tree.word not in word2idx:
            if can_add:
                word2idx[tree.word] = word_idx
                word_idx += 1
                tree.word = word2idx[tree.word]
            else:
                tree.word = word_idx
        else:
            tree.word = word2idx[tree.word]

    return word2idx, word_idx


def get_train_test_sequences(train_filename="trees/train.txt", test_filename="trees/test.txt", save=True, from_save=True):

    if from_save:
        try:
            train_left = np.load("trees/train_left.npy")
            train_right = np.load("trees/train_right.npy")
            train_words = np.load("trees/train_words.npy")
            train_labels = np.load("trees/train_labels.npy")
            test_left = np.load("trees/test_left.npy")
            test_right = np.load("trees/test_right.npy")
            test_words = np.load("trees/test_words.npy")
            test_labels = np.load("trees/test_labels.npy")
            with open("trees/word2idx.json", "r") as f:
                word2idx = json.loads(f.readline())

            return (train_left, train_right, train_words, train_labels), (test_left, test_right, test_words, test_labels), word2idx

        except Exception as e:
            print("error loading")

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print("Parsing Training trees")

    start = datetime.datetime.now()
    train_roots = load_trees_from_file(train_filename)
    time_taken = (datetime.datetime.now() - start).microseconds
    print("Found {} trees in {} ms".format(len(train_roots), time_taken / 1000))

    word2idx = {"BUFFER" : 0}
    word_idx = 1

    train_left = []
    train_right = []
    train_words = []
    train_labels = []

    for root in train_roots:
        word2idx, word_idx = convert_trees_to_word2idx(root, word2idx, word_idx, can_add=True)

        # convert to sequences
        left = []
        right = []
        words = []
        labels = []
        root.convert_to_sequence(left, right, words, labels)
        train_left.append(left)
        train_right.append(right)
        train_words.append(words)
        train_labels.append(labels)

    train_left = pad_sequences(train_left)
    train_right = pad_sequences(train_right)
    train_words = pad_sequences(train_words)
    train_labels = pad_sequences(train_labels)

    assert train_left.shape[0] == train_words.shape[0] == train_right.shape[0] == train_labels.shape[0]
    assert train_left.shape[1] == train_words.shape[1] == train_right.shape[1] == train_labels.shape[1]

    print("Tokenized train trees, found {} words".format(len(word2idx)))
    print("Converted trees to sequences:")
    print("Left childeren sequence dimentions: {}".format(train_left.shape))
    print("Right childeren dimentions: {}".format(train_right.shape))
    print("Words dimentions: {}".format(train_words.shape))
    print("Label dimentions: {}\n".format(train_labels.shape))

    print("Parsing Test trees")

    start = datetime.datetime.now()
    test_roots = load_trees_from_file(test_filename)
    time_taken = (datetime.datetime.now() - start).microseconds
    print("Found {} trees in {} ms".format(len(test_roots), time_taken / 1000))

    test_left = []
    test_right = []
    test_words = []
    test_labels = []

    for root in test_roots:
        word2idx, word_idx = convert_trees_to_word2idx(root, word2idx, word_idx, can_add=False)

        # convert to sequences
        left = []
        right = []
        words = []
        labels = []
        root.convert_to_sequence(left, right, words, labels)
        test_left.append(left)
        test_right.append(right)
        test_words.append(words)
        test_labels.append(labels)

    test_left = pad_sequences(test_left)
    test_right = pad_sequences(test_right)
    test_words = pad_sequences(test_words)
    test_labels = pad_sequences(test_labels)

    assert test_left.shape[0] == test_words.shape[0] == test_right.shape[0] == test_labels.shape[0]
    assert test_left.shape[1] == test_words.shape[1] == test_right.shape[1] == test_labels.shape[1]

    print("Tokenized test trees".format(len(word2idx)))
    print("Converted test trees to sequences:")
    print("Left childeren sequence dimentions: {}".format(test_left.shape))
    print("Right childeren dimentions: {}".format(test_right.shape))
    print("Words dimentions: {}".format(test_words.shape))
    print("Labels dimentions: {}\n".format(test_labels.shape))

    if save:
        np.save("trees/train_left.npy", train_left)
        np.save("trees/train_right.npy", train_right)
        np.save("trees/train_words.npy", train_words)
        np.save("trees/train_labels.npy", train_labels)
        np.save("trees/test_left.npy", test_left)
        np.save("trees/test_right.npy", test_right)
        np.save("trees/test_words.npy", test_words)
        np.save("trees/test_labels.npy", test_labels)
        with open("trees/word2idx.json", "w") as f:
            f.write(json.dumps(word2idx))

    return (train_left, train_right, train_words, train_labels), (test_left, test_right, test_words, test_labels), word2idx


if __name__ == "__main__":
    train, test, word2idx = get_train_test_sequences(from_save=False)
